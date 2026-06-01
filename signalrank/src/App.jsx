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
      <div className="result-rank">#{index + 1}</div>
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

  const selectedItem = useMemo(() => {
    return results.find((item) => item.id === selectedId) || results[0];
  }, [results, selectedId]);

  function findSignal() {
    const usesOriginalSamples = notes.trim() === sampleNotes.trim();
    const inputItems = usesOriginalSamples ? sampleItems : parseNotesToItems(notes);
    const scored = scoreItems(inputItems);
    setResults(scored);
    setSelectedId(scored[0]?.id);
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div className="eyebrow">Signal vs noise</div>
        <h1>SignalRank</h1>
        <p>SignalRank scores messy information so you can see what deserves attention.</p>
        <span>
          Inspired by HTIS: not every token, task, risk, or update has equal value.
        </span>
      </header>

      <section className="demo-card" aria-label="SignalRank demo">
        <div className="input-block">
          <div className="section-label">Messy information</div>
          <textarea
            aria-label="Project notes to score"
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
            placeholder="Paste one project note, risk, update, or backlog item per line."
          />
          <button className="primary-button" type="button" onClick={findSignal}>
            Find the Signal
          </button>
        </div>

        <div className="results-block">
          <div className="results-heading">
            <div>
              <div className="section-label">Ranked attention list</div>
              <h2>{results.length} items scored</h2>
            </div>
            <div className="top-score">
              <strong>{results[0]?.overallImportanceScore}</strong>
              <span>top signal</span>
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

      <section className="how-it-works" aria-label="How scoring works">
        <div>
          <div className="section-label">How scoring works</div>
          <p>
            Each item is scored locally across impact, urgency, risk, dependency,
            alignment, and confidence, then weighted into one importance score.
          </p>
        </div>
        <div className="weight-row">
          {Object.entries(WEIGHTS).map(([key, weight]) => (
            <span key={key}>
              {formatCategory(key.replace("Score", ""))} {Math.round(weight * 100)}%
            </span>
          ))}
        </div>
      </section>
    </main>
  );
}

export default App;
