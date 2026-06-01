import React, { useMemo, useState } from "react";
import sampleItems from "../data/sample-items.json";
import scoringSchema from "../schema/scoring-schema.json";
import exampleOutput from "../examples/example-output.json";
import { parseNotesToItems, scoreItems, WEIGHTS } from "./scoring-engine.js";

const dimensionLabels = [
  ["impactScore", "Impact"],
  ["urgencyScore", "Urgency"],
  ["riskScore", "Risk"],
  ["dependencyScore", "Dependency"],
  ["strategicAlignmentScore", "Strategic alignment"],
  ["confidenceScore", "Confidence"]
];

const starterNotes = [
  "Security review found exposed admin endpoint: admin settings endpoint is reachable without expected network restriction.",
  "Analytics vendor API delay blocks launch reporting: attribution data is delayed and the launch report is due Friday.",
  "Add dark mode preference to account settings: nice-to-have enhancement that does not affect the current launch plan."
].join("\n");

function formatCategory(category) {
  return category.replace(/_/g, " ");
}

function getPriorityClass(label) {
  return label.toLowerCase().replace(/\s+/g, "-");
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
      <div className="result-body">
        <div className="result-topline">
          <h3>{item.title}</h3>
          <div className="score-badge">{item.overallImportanceScore}</div>
        </div>
        <div className="result-meta">
          <span className={`priority ${getPriorityClass(item.priorityLabel)}`}>
            {item.priorityLabel}
          </span>
          <span>{formatCategory(item.category)}</span>
        </div>
        <p>{item.explanation}</p>
        <div className="recommended-action">{item.recommendedAction}</div>
      </div>
    </button>
  );
}

function App() {
  const [notes, setNotes] = useState(starterNotes);
  const [mode, setMode] = useState("sample");
  const [results, setResults] = useState(() => scoreItems(sampleItems));
  const [selectedId, setSelectedId] = useState(() => exampleOutput[0]?.id || "SR-006");

  const selectedItem = useMemo(() => {
    return results.find((item) => item.id === selectedId) || results[0];
  }, [results, selectedId]);

  const schemaFields = scoringSchema.required.length;

  function runSignalRank(nextMode = mode) {
    const inputItems = nextMode === "sample" ? sampleItems : parseNotesToItems(notes);
    const scored = scoreItems(inputItems);
    setResults(scored);
    setSelectedId(scored[0]?.id);
  }

  function useSamples() {
    setMode("sample");
    const scored = scoreItems(sampleItems);
    setResults(scored);
    setSelectedId(scored[0]?.id);
  }

  function scorePastedNotes() {
    setMode("notes");
    runSignalRank("notes");
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <div className="eyebrow">Explainable attention priority</div>
          <h1>SignalRank</h1>
          <p>
            SignalRank is an explainable AI importance engine that scores messy
            project information based on what deserves human attention. It
            separates urgent signal from background noise and traces its roots
            to HTIS: the idea that not all information has equal value.
          </p>
        </div>
        <div className="hero-proof">
          <span>{sampleItems.length} sample items</span>
          <strong>{exampleOutput[0]?.overallImportanceScore}</strong>
          <span>top scored priority</span>
        </div>
      </header>

      <section className="workspace" aria-label="SignalRank demo workspace">
        <aside className="panel input-panel">
          <div className="panel-heading">
            <span>Input</span>
            <strong>{mode === "sample" ? "Sample data" : "Pasted notes"}</strong>
          </div>

          <button className="primary-button" type="button" onClick={() => runSignalRank()}>
            Run SignalRank
          </button>

          <div className="control-row">
            <button className="secondary-button" type="button" onClick={useSamples}>
              Load samples
            </button>
            <button className="secondary-button" type="button" onClick={scorePastedNotes}>
              Score pasted notes
            </button>
          </div>

          <label htmlFor="notes">Paste rough project notes or backlog items</label>
          <textarea
            id="notes"
            value={notes}
            onChange={(event) => {
              setNotes(event.target.value);
              setMode("notes");
            }}
            placeholder="One item per line. Example: Security issue: exposed endpoint needs review."
          />

          <div className="schema-note">
            <strong>{schemaFields}</strong>
            <span>schema fields support each scored result.</span>
          </div>
        </aside>

        <section className="panel results-panel">
          <div className="panel-heading">
            <span>Ranked priorities</span>
            <strong>{results.length} results</strong>
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
        </section>

        <aside className="panel detail-panel">
          {selectedItem && (
            <>
              <div className="panel-heading">
                <span>Why it scored this way</span>
                <strong>{selectedItem.overallImportanceScore}/100</strong>
              </div>

              <div className="detail-title">
                <span className={`priority ${getPriorityClass(selectedItem.priorityLabel)}`}>
                  {selectedItem.priorityLabel}
                </span>
                <h2>{selectedItem.title}</h2>
                <p>{selectedItem.description}</p>
              </div>

              <div className="dimension-stack">
                {dimensionLabels.map(([key, label]) => (
                  <DimensionBar key={key} label={label} value={selectedItem[key]} />
                ))}
              </div>

              <div className="detail-copy">
                <h3>Explanation</h3>
                <p>{selectedItem.explanation}</p>
                <h3>Recommended action</h3>
                <p>{selectedItem.recommendedAction}</p>
              </div>
            </>
          )}
        </aside>
      </section>

      <section className="how-it-works" aria-label="How scoring works">
        <div>
          <h2>How scoring works</h2>
          <p>
            SignalRank uses transparent local scoring. Each item receives a 1-5
            score for impact, urgency, risk, dependency, strategic alignment,
            and confidence. Those dimensions are weighted into a 0-100
            importance score, then mapped to a practical priority label.
          </p>
        </div>
        <div className="weight-grid">
          {Object.entries(WEIGHTS).map(([key, weight]) => (
            <div className="weight-item" key={key}>
              <span>{formatCategory(key.replace("Score", ""))}</span>
              <strong>{Math.round(weight * 100)}%</strong>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}

export default App;
