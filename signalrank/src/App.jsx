import React, { useMemo, useState } from "react";
import sampleItems from "../data/sample-items.json";
import { parseNotesToItems, scoreItems, WEIGHTS } from "./scoring-engine.js";

const dimensionLabels = [
  ["impactScore", "Impact"],
  ["urgencyScore", "Urgency"],
  ["riskScore", "Risk"],
  ["dependencyScore", "Dependency"],
  ["strategicAlignmentScore", "Alignment"],
  ["confidenceScore", "Confidence"]
];

const sampleNotes = sampleItems
  .map((item) => `${item.title}: ${item.description}`)
  .join("\n");

const initialResults = scoreItems(sampleItems);

function formatCategory(category) {
  return category.replace(/_/g, " ");
}

function getPriorityClass(label) {
  return label.toLowerCase().replace(/\s+/g, "-");
}

function getShortReason(explanation) {
  return `${explanation.split(". Signals include")[0]}.`;
}

function DimensionBar({ label, value }) {
  return (
    <div className="dimension-row">
      <div className="dimension-label">
        <span>{label}</span>
        <strong>{value}/5</strong>
      </div>
      <div className="bar-track" aria-hidden="true">
        <div className="bar-fill" style={{ width: `${value * 20}%` }} />
      </div>
    </div>
  );
}

function ResultCard({ item, index, selected, onSelect }) {
  return (
    <button
      className={`result-card ${selected ? "selected" : ""}`}
      onClick={() => onSelect(item.id)}
      type="button"
    >
      <div className="result-rank mono">#{String(index + 1).padStart(2, "0")}</div>
      <div className="result-main">
        <div className="result-title-row">
          <div>
            <div className="result-meta">
              <span className={`priority ${getPriorityClass(item.priorityLabel)}`}>
                {item.priorityLabel}
              </span>
              <span>{formatCategory(item.category)}</span>
            </div>
            <h3>{item.title}</h3>
          </div>
          <div className="score-badge">
            <strong>{item.overallImportanceScore}</strong>
            <span>score</span>
          </div>
        </div>

        <p className="short-reason">{getShortReason(item.explanation)}</p>

        {selected && (
          <div className="expanded-detail">
            <p>{item.description}</p>
            <div className="dimension-grid">
              {dimensionLabels.map(([key, label]) => (
                <DimensionBar key={key} label={label} value={item[key]} />
              ))}
            </div>
            <div className="recommended-action">
              <span>Next move</span>
              {item.recommendedAction}
            </div>
          </div>
        )}
      </div>
    </button>
  );
}

function App() {
  const [notes, setNotes] = useState(sampleNotes);
  const [results, setResults] = useState(initialResults);
  const [selectedId, setSelectedId] = useState(initialResults[0]?.id);
  const [loading, setLoading] = useState(false);

  const selectedItem = useMemo(() => {
    return results.find((item) => item.id === selectedId) || results[0];
  }, [results, selectedId]);

  function findSignal() {
    if (loading) return;
    setLoading(true);
    setTimeout(() => {
      const usesOriginalSamples = notes.trim() === sampleNotes.trim();
      const inputItems = usesOriginalSamples ? sampleItems : parseNotesToItems(notes);
      const scored = scoreItems(inputItems);
      setResults(scored);
      setSelectedId(scored[0]?.id);
      setLoading(false);
    }, 700);
  }

  return (
    <main className="app-shell">
      <nav className="top-nav">
        <span className="nav-wordmark">SignalRank</span>
        <a href="#how-it-works" className="nav-link">How it works</a>
      </nav>

      <header className="hero">
        <div className="eyebrow">SignalRank</div>
        <h1>Find what deserves attention.</h1>
        <p>
          Paste messy project updates, risks, blockers, or notes. SignalRank
          scores the signal, explains why it matters, and helps you focus on the
          next best action. SignalRank is inspired by token importance scoring,
          where AI systems estimate which inputs matter most.
        </p>
        <a href="#demo" className="hero-anchor">Try the demo</a>
      </header>

      <section id="demo" className="demo-card" aria-label="SignalRank demo">
        <div className="input-block">
          <div className="section-label">Your messy information</div>
          <textarea
            aria-label="Project notes to score"
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
            placeholder={`Auth service throwing 503s on retry — backend says it's transient\nMobile checkout timeout still unresolved from sprint 4\nStakeholder wants a dashboard for Q3 — not on the roadmap\nCompliance audit scheduled for June, docs are incomplete`}
          />
          <button
            className={`primary-button${loading ? " loading" : ""}`}
            type="button"
            onClick={findSignal}
            disabled={loading}
          >
            {loading ? "Finding the signal..." : "Find the Signal"}
          </button>
        </div>

        <div className="results-block">
          <div className="results-heading">
            <div>
              <div className="section-label">Ranked attention list</div>
              <h2>{results.length} items ranked by attention</h2>
            </div>
          </div>

          <div className="results-list">
            {results.map((item, index) => (
              <ResultCard
                item={item}
                index={index}
                key={item.id}
                selected={selectedItem?.id === item.id}
                onSelect={setSelectedId}
              />
            ))}
          </div>
        </div>
      </section>

      <details id="how-it-works" className="concept-footer">
        <summary className="section-label">How scoring works</summary>
        <div className="concept-footer-body">
          <p>
            Each item gets local importance signals across impact, urgency, risk,
            dependency, alignment, and confidence — then those signals are weighted
            into one attention score. Inspired by token importance scoring in
            transformer models: the idea that not every input deserves equal weight.
          </p>
          <div className="weight-row">
            {Object.entries(WEIGHTS).map(([key, weight]) => (
              <span key={key}>
                {formatCategory(key.replace("Score", ""))} {Math.round(weight * 100)}%
              </span>
            ))}
          </div>
        </div>
      </details>
    </main>
  );
}

export default App;
