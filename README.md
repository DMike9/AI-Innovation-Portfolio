# SignalRank

SignalRank is an explainable AI importance engine that turns messy project information into ranked attention priorities.

The core idea is simple: not all information deserves equal attention. SignalRank scores work items, risks, updates, notes, and requirements so teams can quickly see what matters, what can wait, and why.

This repository is being reframed from a technical AI portfolio into a focused public product concept. The original transformer efficiency work remains preserved as research inspiration.

## Why It Matters

Modern teams are overloaded with project notes, backlog items, operational updates, risks, meeting summaries, and unclear requirements. The hard part is often not collecting information. The hard part is deciding what deserves attention first.

SignalRank helps by making attention priority explicit and explainable. Instead of showing a flat list of work, it produces a ranked view with dimension scores, confidence, reasoning, and a recommended next action.

## Problem

Teams often struggle with:

- Important risks buried inside long updates
- Urgent blockers mixed with low-value enhancements
- Unclear requirements that slow delivery
- Dependencies that quietly delay multiple teams
- Too many project items competing for attention
- Prioritization decisions that are hard to explain

SignalRank is designed to convert this messy information into a clear attention map.

## Who It Is For

SignalRank is intended for:

- Product managers triaging backlog and roadmap items
- Engineering leads reviewing sprint risks and blocked work
- Program managers scanning operational updates
- Security or compliance teams surfacing high-risk issues
- Executives who need concise priority summaries
- Technical reviewers and employers evaluating applied AI product thinking

## HTIS Inspiration

The original research prototype in this repo explored **HTIS**, or Hierarchical Token Importance Scoring, inside transformer models. That prototype asked a model-level question: can a system score which tokens deserve more attention during processing?

SignalRank applies the same principle at a human workflow level:

> If transformer tokens do not all deserve equal compute, project information does not all deserve equal human attention.

The HTIS work is preserved in [`transformer-efficiency/`](transformer-efficiency/) as technical inspiration and research context.

## Scoring Model

SignalRank uses a transparent deterministic scoring model for the first MVP. No LLM is required yet.

Each item is scored across six dimensions:

- **Impact**: How much the item affects customers, revenue, delivery, safety, quality, or business goals.
- **Urgency**: How time-sensitive the item is.
- **Risk**: How severe the downside is if the item is ignored.
- **Dependency**: Whether the item blocks people, teams, systems, or decisions.
- **Strategic Alignment**: How closely the item supports stated goals, priorities, launches, compliance needs, or executive commitments.
- **Confidence**: How much evidence the system has to trust the score.

The current formula is:

```text
overall_importance_score =
  impact * 0.25 +
  urgency * 0.20 +
  risk * 0.20 +
  dependency * 0.15 +
  strategic_alignment * 0.15 +
  confidence * 0.05
```

Dimension scores are calculated on a 1-5 scale, then converted to a 0-100 overall score.

Priority labels:

- `Act Now`: 80-100
- `Prioritize`: 60-79
- `Watch`: 40-59
- `Can Wait`: 0-39

The scoring engine is deliberately inspectable. Reviewers can see which keyword and category signals affect each score.

## Current Product Foundation

The new product foundation lives in [`signalrank/`](signalrank/):

```text
signalrank/
  README.md
  package.json
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

The current foundation includes:

- Sample project items
- Scoring schema
- Deterministic local scoring engine
- Example scored output
- Local demo script

## Run Locally

SignalRank currently runs as a dependency-free Node demo.

```bash
cd signalrank
npm run score
```

You can also run it directly:

```bash
node src/run-demo.js
```

This reads `data/sample-items.json`, scores each item, and prints a ranked list with explanations and recommended actions.

## MVP

The first testable MVP should include:

- Paste or load messy project items
- Score each item using transparent dimensions
- Show ranked priorities
- Explain why each item scored the way it did
- Recommend the next action
- Include sample scenarios for reviewers
- Run without authentication, database, backend, or integrations

The first UI should stay simple: one input area, one ranked output table, and one detail panel explaining the selected item.

## Roadmap Ideas

Near-term:

- Add a lightweight web UI
- Add editable scoring weights
- Add sample scenarios for product, engineering, security, and operations
- Add export to Markdown or CSV

Later:

- Add optional LLM-assisted scoring explanations
- Compare deterministic and AI-generated scoring
- Add feedback controls so users can correct scores
- Add Jira, GitHub Issues, or document import
- Add team-specific scoring profiles
- Track priority drift over time

## Research Prototype

The original HTIS transformer prototype remains in [`transformer-efficiency/`](transformer-efficiency/). It is not the public product, but it documents the technical inspiration behind SignalRank's importance-scoring philosophy.
