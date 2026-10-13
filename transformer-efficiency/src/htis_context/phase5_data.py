"""Frozen short-context evaluation; official paragraphs are never truncated."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

URL = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def recipe(ids):
    return {'tokens': [ids], 'positions': [list(range(len(ids)))], 'mask': [[1]*len(ids)]}


def prepare(output, tokenizer, context_limit=256, answer_limit=6):
    if output.exists():
        raise FileExistsError('Evaluation data is immutable; use a new version')
    output.mkdir(parents=True)
    request = urllib.request.Request(URL, headers={'User-Agent':'HTIS-Context-research/0.2'})
    raw = urllib.request.urlopen(request, timeout=60).read()
    (output/'dev-v1.1.json').write_bytes(raw)
    data = json.loads(raw)
    candidates = []
    # Stable article/QA hash ordering, before any model execution or outcome.
    for article in sorted(data['data'], key=lambda a:hashlib.sha256(a['title'].encode()).hexdigest()):
        eligible = []
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                prefix = 'Read the passage and answer with a short phrase.\nPassage: '+paragraph['context']+'\n'
                suffix = 'Question: '+qa['question']+'\nAnswer:'
                ids = tokenizer(prefix, add_special_tokens=False).input_ids
                tail = tokenizer(suffix, add_special_tokens=False).input_ids
                answers = list(dict.fromkeys(a['text'] for a in qa['answers']))
                answer = tokenizer(' '+answers[0], add_special_tokens=False).input_ids
                if len(ids)+len(tail)+answer_limit<=context_limit and 1<=len(answer)<=answer_limit:
                    eligible.append({'id':qa['id'], 'group':'squad/'+article['title'], 'kind':'official-short-qa',
                        'prefix_text':prefix,'suffix_text':suffix,'prefix_ids':ids,'suffix_ids':tail,
                        'answers':answers,'answer_ids':answer,'paragraph_sha256':hashlib.sha256(paragraph['context'].encode()).hexdigest(),
                        'official_paragraph_untruncated':True})
        if eligible:
            candidates.append(sorted(eligible,key=lambda x:hashlib.sha256(x['id'].encode()).hexdigest())[0])
    rows = []
    for i, row in enumerate(candidates[:4]):
        rows.append({**row,'split':'validation' if i<2 else 'test'})
    filler = 'The log records cloudy weather. The spare boxes contain pencils. The meeting ended at noon. '
    def probe(name, split, context, question, answer, group=None):
        prefix = 'Read the record and answer with a short phrase.\nRecord: '+context+'\n'
        suffix = 'Question: '+question+'\nAnswer:'
        ids=tokenizer(prefix,add_special_tokens=False).input_ids
        tail=tokenizer(suffix,add_special_tokens=False).input_ids
        target=tokenizer(' '+answer,add_special_tokens=False).input_ids
        if len(ids)+len(tail)+answer_limit>context_limit or len(target)>answer_limit:
            raise ValueError('Authored probe exceeds preregistered limit; never truncate')
        rows.append({'id':'probe/'+name,'group':group or 'session/'+name,'kind':name,'split':split,
            'prefix_text':prefix,'suffix_text':suffix,'prefix_ids':ids,'suffix_ids':tail,'answers':[answer],
            'answer_ids':target,'authored_probe':True})
    probe('validation-opening','validation','The vault color is blue. '+filler*3,'What is the vault color?','blue')
    probe('validation-update','validation',filler+'The old code was 12. The revised code is 29. '+filler,'What is the revised code?','29')
    probe('beginning','test','The locker color is green. '+filler*4,'What is the locker color?','green')
    probe('middle','test',filler*2+'The locker color is green. '+filler*2,'What is the locker color?','green')
    probe('end','test',filler*4+'The locker color is green.','What is the locker color?','green')
    probe('delayed-revisited-task-switch','test','The archive code is 47. '+filler*3+'User: Describe the weather. Assistant: It was cloudy. User: Return to the archive code. ', 'What is the archive code?','47')
    probe('multidocument-distractors','test','Document A: Iris works in Lima. '+filler*2+'Document B: Omar works with Iris. '+filler,'In which city does Omar\'s colleague work?','Lima')
    probe('rare-code-context','test','def uncommon_key():\n    return "violet"\n'+filler*3,'What string does uncommon_key return?','violet')
    if len({r['group'] for r in rows})!=len(rows):
        raise ValueError('Validation and test groups must be disjoint')
    for row in rows:
        row['prefix_recipe']=recipe(row['prefix_ids'])
        row['input_tokens']=len(row['prefix_ids'])+len(row['suffix_ids'])
    manifest={'experiment':'phase5-short-context-cpu-pilot-v1','rows':rows,'official_release':'SQuAD v1.1 development',
        'dataset_url':URL,'dataset_sha256':hashlib.sha256(raw).hexdigest(),'license':'CC BY-SA 4.0',
        'license_reference':'https://rajpurkar.github.io/SQuAD-explorer/',
        'selection':'SHA256 article then QA ordering; first eligible QA per distinct article; first two validation, next two held-out',
        'eligibility':{'max_complete_input_plus_generation_tokens':context_limit,'max_first_answer_tokens':answer_limit},
        'generation':{'greedy':True,'max_new_tokens':answer_limit,'fixed_length':True,'eos_stopping':False},
        'metrics':'whole six-token output; SQuAD normalization EM and token F1; no answer extraction tuned on outputs',
        'limitations':['Tiny short-context subset, not a SQuAD leaderboard score.',
            'Authored probes are not RULER or LongBench reproductions. No official long-context benchmark executed.',
            'Full source paragraphs retained. Short-answer and length eligibility imposed before model evaluation.',
            'SmolLM2 base model is not instruction tuned. Natural high-value/low-attention examples not mined; Phase 4 supplied-KV stress remains separate.']}
    write(output/'manifest.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('.model-cache/smollm2-135m',local_files_only=True,trust_remote_code=False)
    result=prepare(a.output,tok)
    print(json.dumps({'rows':len(result['rows']),'lengths':[x['input_tokens'] for x in result['rows']], 'dataset_sha256':result['dataset_sha256']}))


if __name__=='__main__':main()
