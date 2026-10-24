import path from "node:path";

export const CAPTURES_DIR = path.resolve(
  process.env.CAPTURES_DIR || path.join(process.cwd(), "..", "..", "captures")
);

export const IMAGE_TYPES = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".bmp": "image/bmp",
};

export const VOTES_FILE = path.join(CAPTURES_DIR, "votes.json");
