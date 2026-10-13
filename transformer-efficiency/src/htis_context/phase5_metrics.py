"""Answer metrics and paired, group-level bootstrap; no token pseudo-replication."""
from collections import Counter
import re
import string
import numpy as np


def normalize(text):
    text=''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())


def answer_metrics(output, answers):
    pred=normalize(output).split()
    values=[]
    for answer in answers:
        target=normalize(answer).split()
        overlap=sum((Counter(pred)&Counter(target)).values())
        f1=2*overlap/(len(pred)+len(target)) if pred and target else float(pred==target)
        values.append((float(normalize(output)==normalize(answer)),f1))
    return {'em':max(v[0] for v in values),'f1':max(v[1] for v in values)}


def paired_bootstrap(pairs, *, repetitions=2000, seed=1729):
    """pairs: (document/session group, paired delta). Seeds averaged within group."""
    groups={}
    for group, delta in pairs:
        groups.setdefault(group,[]).append(delta)
    if not groups:
        raise ValueError('Need paired groups')
    values=np.array([np.mean(v) for v in groups.values()])
    rng=np.random.default_rng(seed)
    boot=values[rng.integers(0,len(values),(repetitions,len(values)))].mean(1)
    return {'groups':len(values),'mean_delta':float(values.mean()),'median_group_delta':float(np.median(values)),
        'percentile_95_ci':[float(x) for x in np.quantile(boot,[.025,.975])], 'resampling_unit':'document/session',
        'repetitions':repetitions,'seed':seed,'interpretation':'Exploratory tiny-sample interval; no significance claim.'}


def dominated(rows, quality='mean_f1', cost='mean_decode_ms', memory='max_total_storage_bytes'):
    result={}
    for a in rows:
        result[a['arm']]=[b['arm'] for b in rows if b is not a and b[quality]>=a[quality] and b[cost]<=a[cost] and b[memory]<=a[memory]
            and (b[quality]>a[quality] or b[cost]<a[cost] or b[memory]<a[memory])]
    return result
