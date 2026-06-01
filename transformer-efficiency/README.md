# Transformer Efficiency Research Prototype

This folder preserves the original technical research inspiration for SignalRank.

The prototype explores **HTIS**, or Hierarchical Token Importance Scoring, inside a transformer-style model. It demonstrates how token importance can be estimated and applied during a forward pass to suppress lower-importance token representations.

## Relationship To SignalRank

HTIS asks a model-level question:

> Which tokens deserve more attention during transformer processing?

SignalRank applies that same principle to human workflows:

> Which work items, risks, updates, notes, or requirements deserve more human attention?

The code in this folder is not the public product UI. It is a research artifact that explains where the importance-scoring concept came from.

## Current Files

- `src/filename.py`: PyTorch demo of a mini transformer block with token importance scoring.
- `Readme`, `notebook`, `results`: original placeholder artifacts retained for historical context.

## Suggested Future Cleanup

If this prototype is expanded, rename `src/filename.py` to `src/htis_transformer_demo.py` and replace the placeholder files with a real notebook and experiment results.
