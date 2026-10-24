import fs from "node:fs/promises";
import { VOTES_FILE } from "./lib";

// Serialise read-modify-write cycles so concurrent votes aren't lost.
let queue = Promise.resolve();

export async function readVotes() {
  try {
    return JSON.parse(await fs.readFile(VOTES_FILE, "utf8"));
  } catch {
    return {};
  }
}

export function changeVote(name, delta) {
  const job = queue.then(async () => {
    const votes = await readVotes();
    const count = Math.max(0, (votes[name] || 0) + delta);
    if (count) votes[name] = count;
    else delete votes[name];
    await fs.writeFile(VOTES_FILE, JSON.stringify(votes, null, 2));
    return count;
  });
  queue = job.catch(() => {});
  return job;
}
