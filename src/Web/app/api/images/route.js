import fs from "node:fs/promises";
import path from "node:path";
import { CAPTURES_DIR, IMAGE_TYPES } from "../../lib";
import { readVotes } from "../../votes";

export const dynamic = "force-dynamic";

export async function GET() {
  let entries = [];
  try {
    entries = await fs.readdir(CAPTURES_DIR);
  } catch {
    // captures folder doesn't exist yet
  }
  const votes = await readVotes();
  const images = [];
  for (const name of entries) {
    if (!(path.extname(name).toLowerCase() in IMAGE_TYPES)) continue;
    const stat = await fs.stat(path.join(CAPTURES_DIR, name));
    images.push({ name, mtime: stat.mtimeMs, votes: votes[name] || 0 });
  }
  images.sort((a, b) => b.mtime - a.mtime);
  return Response.json(images);
}
