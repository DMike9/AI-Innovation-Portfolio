# SignalRank Product Foundation

SignalRank is an explainable importance engine for project information. It scores messy work items, risks, notes, updates, and requirements based on what deserves human attention.

The product is inspired by HTIS, or Hierarchical Token Importance Scoring. The original AI idea was that not all tokens deserve equal attention inside a transformer model. SignalRank translates that into a public-facing demo: not all project updates, risks, blockers, or work items deserve equal human attention.

This folder contains the first public demo foundation. It is intentionally small: sample data, a scoring schema, a deterministic scoring engine, example output, and a single-screen React demo.

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
    App.jsx
    main.jsx
    scoring-engine.js
    run-demo.js
    styles.css
```

## Run The UI

```bash
npm install
npm run dev
```

Then open the local Vite URL printed by the terminal.

## Run The Scoring CLI

```bash
npm run score
```

No dependencies are required.

## Current Scope

This is not a full enterprise app. It is a polished portfolio demo for the scoring concept.

The current engine:

- Reads sample project items
- Scores each item across six dimensions
- Calculates an overall importance score
- Assigns a priority label
- Generates an explanation
- Recommends a next action
- Splits pasted notes by line and scores them locally

## Scoring Dimensions

- Impact
- Urgency
- Risk
- Dependency
- Strategic alignment
- Confidence

The model is deterministic and transparent by design, so reviewers can inspect exactly how scores are produced. It acts like a simplified attention mechanism for decision support: detect importance signals, weight them, and rank what deserves focus.

## Intentionally Not Included Yet

- Authentication
- Database
- Backend API
- LLM scoring
- Jira, GitHub, or document integrations
- Team settings or saved workspaces
