import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { scoreItems } from "./scoring-engine.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const inputPath = path.join(__dirname, "..", "data", "sample-items.json");

const sampleItems = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const scoredItems = scoreItems(sampleItems);

console.log("SignalRank scored priorities");
console.log("===========================");

for (const item of scoredItems) {
  console.log(`${item.overallImportanceScore} | ${item.priorityLabel} | ${item.id} | ${item.title}`);
  console.log(`  ${item.explanation}`);
  console.log(`  Recommended action: ${item.recommendedAction}`);
  console.log("");
}
