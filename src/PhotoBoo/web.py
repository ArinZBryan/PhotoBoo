"""Launch the PhotoBoo gallery website (Next.js). Usage: uv run photoboo-web [--dev] [--port N]"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "src" / "Web"


def run():
    parser = argparse.ArgumentParser(description="PhotoBoo gallery website")
    parser.add_argument("--dev", action="store_true", help="run the Next.js dev server")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--captures", default=str(REPO_ROOT / "captures"),
                        help="folder containing the captured images")
    args = parser.parse_args()

    npm = shutil.which("npm")
    if npm is None:
        sys.exit("npm not found; install Node.js to run the gallery")

    env = {**os.environ, "CAPTURES_DIR": str(Path(args.captures).resolve())}

    def npm_run(*cmd):
        subprocess.run([npm, *cmd], cwd=WEB_DIR, env=env, check=True)

    try:
        if not (WEB_DIR / "node_modules").exists():
            npm_run("install")
        if args.dev:
            npm_run("run", "dev", "--", "-p", str(args.port))
        else:
            npm_run("run", "build")
            npm_run("run", "start", "--", "-p", str(args.port))
    except KeyboardInterrupt:
        pass
