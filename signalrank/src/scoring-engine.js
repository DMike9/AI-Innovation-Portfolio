const WEIGHTS = {
  impactScore: 0.25,
  urgencyScore: 0.2,
  riskScore: 0.2,
  dependencyScore: 0.15,
  strategicAlignmentScore: 0.15,
  confidenceScore: 0.05
};

const CATEGORY_BASELINES = {
  production_issue: {
    impactScore: 4,
    urgencyScore: 5,
    riskScore: 4,
    dependencyScore: 3,
    strategicAlignmentScore: 4,
    confidenceScore: 4
  },
  security_risk: {
    impactScore: 4,
    urgencyScore: 4,
    riskScore: 5,
    dependencyScore: 3,
    strategicAlignmentScore: 4,
    confidenceScore: 3
  },
  delayed_dependency: {
    impactScore: 3,
    urgencyScore: 4,
    riskScore: 3,
    dependencyScore: 5,
    strategicAlignmentScore: 4,
    confidenceScore: 4
  },
  unclear_requirement: {
    impactScore: 3,
    urgencyScore: 3,
    riskScore: 3,
    dependencyScore: 4,
    strategicAlignmentScore: 3,
    confidenceScore: 2
  },
  blocked_work: {
    impactScore: 3,
    urgencyScore: 4,
    riskScore: 3,
    dependencyScore: 5,
    strategicAlignmentScore: 3,
    confidenceScore: 4
  },
  executive_decision: {
    impactScore: 4,
    urgencyScore: 4,
    riskScore: 3,
    dependencyScore: 5,
    strategicAlignmentScore: 5,
    confidenceScore: 4
  },
  enhancement: {
    impactScore: 2,
    urgencyScore: 1,
    riskScore: 1,
    dependencyScore: 1,
    strategicAlignmentScore: 2,
    confidenceScore: 3
  },
  documentation: {
    impactScore: 2,
    urgencyScore: 2,
    riskScore: 2,
    dependencyScore: 2,
    strategicAlignmentScore: 3,
    confidenceScore: 4
  },
  compliance: {
    impactScore: 4,
    urgencyScore: 4,
    riskScore: 5,
    dependencyScore: 3,
    strategicAlignmentScore: 5,
    confidenceScore: 3
  },
  operational_risk: {
    impactScore: 4,
    urgencyScore: 4,
    riskScore: 4,
    dependencyScore: 3,
    strategicAlignmentScore: 4,
    confidenceScore: 3
  },
  go_to_market: {
    impactScore: 3,
    urgencyScore: 4,
    riskScore: 3,
    dependencyScore: 4,
    strategicAlignmentScore: 4,
    confidenceScore: 4
  },
  technical_debt: {
    impactScore: 2,
    urgencyScore: 1,
    riskScore: 2,
    dependencyScore: 1,
    strategicAlignmentScore: 2,
    confidenceScore: 4
  }
};

const DEFAULT_BASELINE = {
  impactScore: 3,
  urgencyScore: 3,
  riskScore: 3,
  dependencyScore: 3,
  strategicAlignmentScore: 3,
  confidenceScore: 3
};

const SIGNAL_RULES = [
  {
    score: "impactScore",
    keywords: ["customer", "customers", "revenue", "enterprise", "billing", "production", "support"],
    reason: "broad customer, revenue, enterprise, billing, or production impact"
  },
  {
    score: "urgencyScore",
    keywords: ["today", "now", "urgent", "last two hours", "friday", "next week", "scheduled", "launch"],
    reason: "time-sensitive language or near-term deadline"
  },
  {
    score: "riskScore",
    keywords: ["security", "exposed", "exploit", "compliance", "legal", "audit", "failed", "missing", "risk"],
    reason: "security, compliance, failure, or downside risk"
  },
  {
    score: "dependencyScore",
    keywords: ["blocked", "blocks", "depends", "dependency", "waiting", "decision", "pending", "cannot"],
    reason: "other work, teams, or decisions are blocked"
  },
  {
    score: "strategicAlignmentScore",
    keywords: ["executive", "enterprise", "roadmap", "release", "rollout", "launch", "compliance", "goals"],
    reason: "alignment with strategic, launch, enterprise, or compliance priorities"
  },
  {
    score: "confidenceScore",
    keywords: ["confirmed", "tickets", "reported", "found", "scheduled", "asked", "recommends"],
    reason: "specific evidence is present"
  }
];

const DAMPENING_RULES = [
  {
    scores: ["impactScore", "urgencyScore", "strategicAlignmentScore"],
    phrases: ["does not affect", "does not impact"],
    reason: "explicitly states limited effect"
  },
  {
    scores: ["urgencyScore", "riskScore"],
    phrases: ["nice-to-have", "low priority"],
    reason: "described as optional or low priority"
  },
  {
    scores: ["dependencyScore", "strategicAlignmentScore"],
    phrases: ["does not block", "does not block current roadmap"],
    reason: "explicitly states it does not block roadmap work"
  }
];

function clampScore(value) {
  return Math.max(1, Math.min(5, value));
}

function normalizeText(item) {
  return `${item.title} ${item.description} ${item.category}`.toLowerCase();
}

function applySignalRules(item, baseline) {
  const text = normalizeText(item);
  const scores = { ...baseline };
  const reasons = {
    impactScore: [],
    urgencyScore: [],
    riskScore: [],
    dependencyScore: [],
    strategicAlignmentScore: [],
    confidenceScore: []
  };

  for (const rule of SIGNAL_RULES) {
    const matched = rule.keywords.some((keyword) => text.includes(keyword));
    if (matched) {
      scores[rule.score] = clampScore(scores[rule.score] + 1);
      reasons[rule.score].push(rule.reason);
    }
  }

  for (const rule of DAMPENING_RULES) {
    const matched = rule.phrases.some((phrase) => text.includes(phrase));
    if (matched) {
      for (const score of rule.scores) {
        scores[score] = clampScore(scores[score] - 1);
        reasons[score].push(rule.reason);
      }
    }
  }

  if (item.description.length < 90) {
    scores.confidenceScore = clampScore(scores.confidenceScore - 1);
    reasons.confidenceScore.push("limited description detail lowers confidence");
  }

  if (text.includes("unclear") || text.includes("incomplete") || text.includes("unknown")) {
    scores.confidenceScore = clampScore(scores.confidenceScore - 1);
    reasons.confidenceScore.push("unclear or incomplete information lowers confidence");
  }

  return { scores, reasons };
}

function calculateOverallScore(scores) {
  const weightedTotal = Object.entries(WEIGHTS).reduce((total, [scoreName, weight]) => {
    return total + scores[scoreName] * weight;
  }, 0);

  return Math.round((weightedTotal / 5) * 100);
}

function getPriorityLabel(overallImportanceScore) {
  if (overallImportanceScore >= 80) return "Act Now";
  if (overallImportanceScore >= 60) return "Prioritize";
  if (overallImportanceScore >= 40) return "Watch";
  return "Can Wait";
}

function getTopDrivers(scores) {
  return Object.entries(scores)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([name]) => name);
}

function readableScoreName(scoreName) {
  return scoreName
    .replace("Score", "")
    .replace(/([A-Z])/g, " $1")
    .toLowerCase();
}

function selectEvidence(priorityLabel, reasons) {
  const allEvidence = Object.values(reasons).flat();
  const dampeningEvidence = allEvidence.filter((reason) => {
    return reason.includes("explicitly") || reason.includes("optional") || reason.includes("limited");
  });

  if (priorityLabel === "Can Wait" && dampeningEvidence.length > 0) {
    return dampeningEvidence;
  }

  return allEvidence;
}

function createExplanation(item, scores, reasons, priorityLabel) {
  const topDrivers = getTopDrivers(scores)
    .map(readableScoreName)
    .join(", ");
  const evidence = selectEvidence(priorityLabel, reasons);
  const uniqueEvidence = [...new Set(evidence)].slice(0, 3);

  const evidenceText = uniqueEvidence.length > 0
    ? ` Signals include ${uniqueEvidence.join("; ")}.`
    : "";

  return `${item.title} is labeled "${priorityLabel}" because its strongest drivers are ${topDrivers}.${evidenceText}`;
}

function createRecommendedAction(priorityLabel, scores) {
  if (priorityLabel === "Act Now") {
    if (scores.riskScore >= 5) return "Assign an owner immediately, confirm mitigation steps, and review status today.";
    if (scores.dependencyScore >= 5) return "Escalate the blocker, identify the decision owner, and unblock dependent work today.";
    return "Move to the top of the review queue and decide the next action today.";
  }

  if (priorityLabel === "Prioritize") {
    if (scores.confidenceScore <= 2) return "Clarify missing details, then schedule prioritization in the next planning review.";
    return "Plan action this week and track ownership, due date, and dependency status.";
  }

  if (priorityLabel === "Watch") {
    return "Monitor for changes and revisit if impact, urgency, or dependency signals increase.";
  }

  return "Defer for now unless new evidence raises impact, urgency, or strategic relevance.";
}

export function scoreItem(item) {
  const baseline = CATEGORY_BASELINES[item.category] || DEFAULT_BASELINE;
  const { scores, reasons } = applySignalRules(item, baseline);
  const overallImportanceScore = calculateOverallScore(scores);
  const priorityLabel = getPriorityLabel(overallImportanceScore);

  return {
    id: item.id,
    title: item.title,
    description: item.description,
    category: item.category,
    ...scores,
    overallImportanceScore,
    priorityLabel,
    explanation: createExplanation(item, scores, reasons, priorityLabel),
    recommendedAction: createRecommendedAction(priorityLabel, scores)
  };
}

export function scoreItems(items) {
  return items
    .map(scoreItem)
    .sort((a, b) => b.overallImportanceScore - a.overallImportanceScore);
}

export { WEIGHTS };
