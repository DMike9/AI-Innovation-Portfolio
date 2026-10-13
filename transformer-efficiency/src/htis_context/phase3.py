"""Phase 3 only: corpus, traces, fitting, held-out learned-policy evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["acquire","trace","train","evaluate"])
    parser.add_argument("--config",type=Path,default=Path("transformer-efficiency/configs/phase3-small.json"))
    parser.add_argument("--model-config",type=Path,default=Path("transformer-efficiency/configs/smollm2-135m.json"))
    parser.add_argument("--model-dir",type=Path,default=Path(".model-cache/smollm2-135m"))
    parser.add_argument("--corpus",type=Path,default=Path("transformer-efficiency/data/phase3-public-v1/corpus.json"))
    parser.add_argument("--dataset",type=Path,default=Path("transformer-efficiency/results/phase3/dataset-v1"))
    parser.add_argument("--training",type=Path,default=Path("transformer-efficiency/results/phase3/trained-v1"))
    parser.add_argument("--evaluation",type=Path,default=Path("transformer-efficiency/results/phase3/evaluation-v1"))
    args=parser.parse_args(argv)
    config=json.loads(args.config.read_text(encoding="utf-8"))
    if args.command=="acquire":
        from .public_corpus import acquire
        result=acquire(config,args.corpus.parent)
        print(f"Public documents: {len(result['documents'])}")
    elif args.command=="train":
        from .training import fit
        result=fit(args.dataset,args.training,config)
        print(f"Trained models: {len(result['models'])}")
    else:
        from .checkpoint import load
        model_config=json.loads(args.model_config.read_text(encoding="utf-8"))
        model,tokenizer,adapter,_=load(model_config,args.model_dir)
        if args.command=="trace":
            from .counterfactual_data import collect
            result=collect(model,tokenizer,adapter,model_config,config,args.corpus,args.dataset)
            print(f"Feature rows: {result['feature_rows']}; offline actions: {result['action_rows']}")
        else:
            from .phase3_evaluation import evaluate
            result=evaluate(model,tokenizer,adapter,model_config,config,args.dataset,args.training,args.evaluation)
            print(f"Runtime checks: {len(result['checks'])}; passed={result['passed']}")
            return 0 if result['passed'] else 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
