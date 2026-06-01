# SignalRank Product Foundation

SignalRank is an explainable importance engine for project information. It scores messy work items, risks, notes, updates, and requirements based on what deserves human attention.

This folder contains the first product foundation. It is intentionally small: sample data, a scoring schema, a deterministic scoring engine, and example output.

## Folder Structure

```text
signalrank/
  data/
    sample-items.json
  schema/
    scoring-schema.json
  examples/
    example-output.json
  src/
    scoring-engine.js
    run-demo.js
```

## Run

```bash
npm run score
```

No dependencies are required.

## Current Scope

This is not the full UI. It is the scoring and product foundation for a future public demo.

The current engine:

- Reads sample project items
- Scores each item across six dimensions
- Calculates an overall importance score
- Assigns a priority label
- Generates an explanation
- Recommends a next action

## Scoring Dimensions

- Impact
- Urgency
- Risk
- Dependency
- Strategic alignment
- Confidence

The model is deterministic and transparent by design, so reviewers can inspect exactly how scores are produced.
