"""Outcome-independent Phase 5B data preparation; no model execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .phase5_data import recipe, sha, write

ROOT = Path('transformer-efficiency')
DATA = ROOT / 'data/phase5b-extended-v3/manifest.json'


def prepare(output, tokenizer):
    output.mkdir(parents=True, exist_ok=False)
    old = json.loads((ROOT / 'data/phase5-short-v1/manifest.json').read_text())
    source = ROOT / 'data/phase5-short-v1/dev-v1.1.json'
    excluded = {r['group'].removeprefix('squad/') for r in old['rows']}
    corpus = json.loads(source.read_text())
    encode = lambda text: tokenizer(text, add_special_tokens=False).input_ids
    rows = []

    def add(name, group, split, kind, context, question, answers, limit, generation=16, **extra):
        prefix = 'Read the documents and answer the question with a short phrase.\nDocuments:\n' + context + '\n'
        suffix = 'Question: ' + question + '\nAnswer:'
        p, s, a = encode(prefix), encode(suffix), encode(' ' + answers[0])
        if len(p) + len(s) + generation > 8192 or not a or len(a) > 12:
            raise ValueError('Unsupported complete input/answer; never truncate')
        rows.append(dict(id=name, group=group, split=split, kind=kind,
            prefix_text=prefix, suffix_text=suffix, prefix_ids=p, suffix_ids=s,
            prefix_recipe=recipe(p), answers=answers, answer_ids=a,
            input_tokens=len(p)+len(s), target_context_bucket=limit,
            generation_tokens=generation, **extra))

    candidates = []
    for article in sorted(corpus['data'], key=lambda x: hashlib.sha256(x['title'].encode()).hexdigest()):
        if article['title'] in excluded:
            continue
        eligible = []
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                answers = list(dict.fromkeys(x['text'] for x in qa['answers']))
                if len(encode(paragraph['context'])) <= 300 and 1 <= len(encode(' '+answers[0])) <= 12:
                    eligible.append((qa, paragraph, answers))
        if eligible:
            qa, paragraph, answers = min(eligible, key=lambda x: hashlib.sha256(x[0]['id'].encode()).hexdigest())
            candidates.append((article, qa, paragraph, answers))
    # Separate article/session groups from every prior phase and each other.
    for i, (article, qa, paragraph, answers) in enumerate(candidates[:5]):
        split = 'validation' if i < 2 else 'test'
        bucket = [512, 1024, 512, 1024, 2048][i]
        position = ['start', 'end', 'start', 'middle', 'end'][i]
        # Complete additional paragraphs from THIS source article, never clipped.
        distractors = [p['context'] for p in article['paragraphs'] if p['context'] != paragraph['context']]
        filler = ''
        for text in distractors * 10:
            if len(encode(filler + paragraph['context'])) >= bucket - 100:
                break
            filler += text + '\n\n'
        chunks = filler.split('\n\n')
        middle = len(chunks)//2
        if position == 'start':
            context = paragraph['context'] + '\n\n' + filler
        elif position == 'end':
            context = filler + paragraph['context']
        else:
            context = '\n\n'.join(chunks[:middle])+'\n\n'+paragraph['context']+'\n\n'+'\n\n'.join(chunks[middle:])
        add('squad5b/'+qa['id'], 'squad/'+article['title'], split,
            'adapted-public-qa-'+position, context, qa['question'], answers, bucket,
            generation=32 if i == 4 else 16, source_title=article['title'],
            source_qa_id=qa['id'], original_paragraph=paragraph['context'],
            relevant_paragraph_sha256=hashlib.sha256(paragraph['context'].encode()).hexdigest(),
            relevant_character_start=context.index(paragraph['context']),
            original_paragraph_untruncated=True, adaptation='Added complete same-article distractor paragraphs and relocated target paragraph')

    filler = 'The shipping ledger lists unused cardboard crates. The afternoon report describes mild winds. A spare desk holds paper clips. '
    def padded(first, last, bucket):
        text = first
        while len(encode(text+last)) < bucket-100:
            text += filler
        return text + last
    add('session5b/delayed', 'session5b/delayed', 'test', 'delayed-relevance',
        padded('The expedition access word is cobalt. Remember it for the final question. ',
            'An unrelated conversation discussed the empty crates. Now return to the expedition. ',1024),
        'What is the expedition access word?', ['cobalt'], 1024, authored_probe=True)
    add('session5b/multidoc', 'session5b/multidoc', 'test', 'multi-document-reasoning',
        padded('Document A: Farah manages the observatory in Quito. ',
            'Document B: Nabil reports to Farah. ',1024),
        'In which city is the observatory managed by Nabil\'s supervisor?', ['Quito'],1024, authored_probe=True)
    shared = padded('The earlier project identifier is 68. The supply office is in Bergen. ',
        'User: Discuss the weather. Assistant: Mild winds were reported. ',512)
    add('session5b/switch-office','session5b/switch','test','task-switch-current',shared,
        'Which city contains the supply office?', ['Bergen'],512, authored_probe=True)
    add('session5b/switch-history','session5b/switch','test','task-switch-history',shared,
        'Return to the earlier project. What is its identifier?', ['68'],512, authored_probe=True)
    add('session5b/revisit','session5b/revisit','test','revisited-historical-information',
        padded('Historical record: the retired beacon name was saffron. ',
            'Current record: the active beacon is named copper. Revisit the retired beacon, not the active one. ',2048),
        'What was the retired beacon name?', ['saffron'],2048,generation=32,authored_probe=True)
    cache = Path('.model-cache/phase5b-longbench')
    acquisition = json.loads((ROOT/'results/phase5b/official-acquisition-success.json').read_text())
    prompts = json.loads((cache/'dataset2prompt.json').read_text(encoding='utf-8'))
    limits = json.loads((cache/'dataset2maxlen.json').read_text())
    used_titles = {r.get('source_title','').replace('_',' ') for r in rows} | {t.replace('_',' ') for t in excluded}
    official = []
    for dataset in ['hotpotqa','2wikimqa']:
        release = cache/(dataset+'.jsonl')
        candidates = []
        for line in release.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            titles = set(re.findall(r'Passage \d+:\n([^\n]+)',item['context']))
            if titles & used_titles:continue
            prompt = prompts[dataset].format(context=item['context'],input=item['input'])
            ids = encode(prompt);answer = encode(' '+item['answers'][0])
            if 512<=len(ids)<=2400 and 1<=len(answer)<=12:
                candidates.append((item,prompt,ids,answer,titles))
        if not candidates:raise ValueError('No complete compatible official '+dataset+' case; record infeasibility')
        item,prompt,ids,answer,titles = min(candidates,key=lambda x:hashlib.sha256(x[0]['_id'].encode()).hexdigest())
        used_titles.update(titles)
        # Tokenize the WHOLE official prompt once; split at an existing token
        # boundary so prefix/suffix concatenation exactly preserves official IDs.
        offsets = tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True).offset_mapping
        context_start = prompt.index(item['context']);context_end = context_start+len(item['context'])
        cut = max(i+1 for i,(start,end) in enumerate(offsets) if end<=context_end)
        text_cut = offsets[cut][0]
        record = dict(id='longbench/'+dataset+'/'+item['_id'],group='longbench/'+dataset+'/'+item['_id'],split='test',
            kind='official-source-'+dataset,prefix_text=prompt[:text_cut],suffix_text=prompt[text_cut:],
            prefix_ids=ids[:cut],suffix_ids=ids[cut:],prefix_recipe=recipe(ids[:cut]),
            input_tokens=len(ids),answers=item['answers'],answer_ids=answer,generation_tokens=limits[dataset],
            original_context=item['context'],source_titles=sorted(titles),official_prompt=prompt,
            official_prompt_ids=ids,official_context_untruncated=True,official_generation_limit=limits[dataset],
            relevant_token_start=sum(end<=context_start for start,end in offsets),relevant_token_end=cut,
            coverage_definition='Entire official multi-document context; not annotated answer-support span',
            source_dataset_revision=acquisition['dataset_revision'],source_repository_revision=acquisition['repository_revision'])
        assert record['prefix_ids']+record['suffix_ids']==encode(prompt)
        rows.append(record);write(output/(dataset+'-original-selected.json'),item)
        official.append(dict(dataset=dataset,id=item['_id'],release_sha256=sha(release),selected_original_sha256=sha(output/(dataset+'-original-selected.json')),
            eligible_cases=len(candidates),input_tokens=len(ids),generation_tokens=limits[dataset],source_titles=sorted(titles)))
    for row in rows:
        if row.get('official_context_untruncated'):
            continue
        elif row.get('original_paragraph'):
            offset = row['prefix_text'].index(row['original_paragraph'])
            row['relevant_token_start'] = len(encode(row['prefix_text'][:offset]))
            row['relevant_token_end'] = len(encode(row['prefix_text'][:offset]+row['original_paragraph']))
        else:
            row['relevant_token_start'] = len(encode('Read the documents and answer the question with a short phrase.\nDocuments:\n'))
            row['relevant_token_end'] = row['relevant_token_start']+len(encode(row['prefix_text'].split('Documents:\n')[1].split('. ')[0]+'.'))
    val = {r['group'] for r in rows if r['split']=='validation'}
    test = {r['group'] for r in rows if r['split']=='test'}
    assert val.isdisjoint(test) and not {r['group'] for r in rows} & {r['group'] for r in old['rows']}
    manifest = dict(experiment='phase5b-extended-v3', rows=rows,official_longbench_subset=official,
        official_acquisition=acquisition,
        source_path=source.as_posix(), source_sha256=sha(source),
        source_url='https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json',
        source_license='CC BY-SA 4.0', previous_data_sha256=sha(ROOT/'data/phase5-short-v1/manifest.json'),
        selection='SHA256 article, then QA ordering; first five eligible unused articles; first two validation, next three test',
        eligibility='Unclipped original paragraph <=300 model tokens; first answer 1..12 tokens. Complete distractor paragraphs may overshoot nominal length buckets.',
        label='Adapted SQuAD/authored sessions plus two complete LongBench v1 official-source cases with official prompts/32-token limits; not a full benchmark leaderboard score',
        limitations=['Ten held-out cases in nine source/session clusters; length and workload partly confounded.',
            'Known public development answers may have appeared in foundation-model pretraining; untouched refers to this project\'s training/calibration.',
            'Independent shared-prefix task queries, not an approximate multi-turn serving implementation.'])
    write(output/'manifest.json', manifest)
    (output/'NOTICE.md').write_text('Adapted SQuAD v1.1 development paragraphs (CC BY-SA 4.0), from the already checksummed public release. Original target and distractor paragraphs are complete. Added authored prompts/session probes are explicitly marked. Two complete LongBench v1 official-source HotpotQA/2WikiMultihopQA cases use official prompts and 32-token limits; release and source repository revisions are pinned. LongBench distribution attribution: https://github.com/THUDM/LongBench/tree/main/LongBench ; underlying data rights and licenses remain with their original datasets. This tiny eligibility-selected subset is not a full benchmark/leaderboard result. See manifest.json for source, selection and attribution.\n',encoding='utf-8')
    return manifest
