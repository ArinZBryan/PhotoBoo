import fs from "node:fs/promises";
import path from "node:path";
import { CAPTURES_DIR, IMAGE_TYPES } from "../../lib";
import { changeVote } from "../../votes";

export const dynamic = "force-dynamic";

export async function POST(req) {
  const { name, action } = await req.json().catch(() => ({}));
  if (
    typeof name !== "string" ||
    path.basename(name) !== name ||
    !(path.extname(name).toLowerCase() in IMAGE_TYPES) ||
    !["add", "remove"].includes(action)
  ) {
    return Response.json({ error: "bad request" }, { status: 400 });
  }
  try {
    await fs.access(path.join(CAPTURES_DIR, name));
  } catch {
    return Response.json({ error: "not found" }, { status: 404 });
  }
  const votes = await changeVote(name, action === "add" ? 1 : -1);
  return Response.json({ name, votes });
}
