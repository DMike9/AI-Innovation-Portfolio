import test from 'node:test';
import assert from 'node:assert/strict';
import { inferCategory, parseNotesToItems, scoreItems, scoreItem, WEIGHTS } from '../src/scoring-engine.js';

const sample = {
  id: 'A', title: 'A confirmed production failure',
  description: 'Customer checkout failed today and blocks a revenue transaction.',
  category: 'production_issue'
};

test('dimension weights sum to one', () => {
  assert.ok(Math.abs(Object.values(WEIGHTS).reduce((a, b) => a + b, 0) - 1) < 1e-12);
});

test('results deterministic with bounded dimension values and useful evidence', () => {
  assert.deepEqual(scoreItem(sample), scoreItem(sample));
  const scored = scoreItem(sample);
  assert.ok(scored.overallImportanceScore >= 0 && scored.overallImportanceScore <= 100);
  for (const key of Object.keys(WEIGHTS)) {
    assert.ok(scored[key] >= 1 && scored[key] <= 5, `${key} out of bounds`);
  }
  assert.ok(scored.explanation.includes('Signals include'));
});

test('parse notes preserves descriptions with colons and assigns unique IDs', () => {
  const data = parseNotesToItems('Outage: customer production failure: today\nDocs: release notes');
  assert.equal(data.length, 2);
  assert.equal(data[0].description, 'customer production failure: today');
  assert.notEqual(data[0].id, data[1].id);
  assert.equal(inferCategory('Potential compliance audit'), 'compliance');
});

test('items sort by importance descending', () => {
  const scored = scoreItems([sample, {
    id: 'B', title: 'Minor optional enhancement',
    description: 'Nice-to-have preference only; does not affect delivery.',
    category: 'enhancement'
  }]);
  assert.equal(scored[0].id, 'A');
  assert.ok(scored[0].overallImportanceScore > scored[1].overallImportanceScore);
});
