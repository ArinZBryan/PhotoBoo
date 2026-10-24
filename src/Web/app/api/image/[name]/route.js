import fs from "node:fs/promises";
import path from "node:path";
import { CAPTURES_DIR, IMAGE_TYPES } from "../../../lib";

export const dynamic = "force-dynamic";

export async function GET(_req, { params }) {
  const { name } = await params;
  const type = IMAGE_TYPES[path.extname(name).toLowerCase()];
  // basename check blocks path traversal
  if (!type || path.basename(name) !== name) {
    return new Response("Not found", { status: 404 });
  }
  try {
    const data = await fs.readFile(path.join(CAPTURES_DIR, name));
    return new Response(data, {
      headers: { "Content-Type": type, "Cache-Control": "no-cache" },
    });
  } catch {
    return new Response("Not found", { status: 404 });
  }
}
