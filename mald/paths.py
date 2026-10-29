"""file roots of the pipeline. each root can be set with an env var.

MALD_ROOT     work root, default <repo>/work. put it on a fast local disk.
  casimir/    casimir release files      (MALD_CASIMIR)
  build/      corpus tables, paper texts (MALD_BUILD)
  out/        system outputs, scores     (MALD_OUT)
  cache/      llm reply cache            (MALD_CACHE)
  judge/      llm-judge bundles, labels  (MALD_JUDGE)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("MALD_ROOT", REPO / "work"))


def env(name, sub):
    return Path(os.environ.get(name, ROOT / sub))


CASIMIR = env("MALD_CASIMIR", "casimir")
BUILD = env("MALD_BUILD", "build")
OUT = env("MALD_OUT", "out")
CACHE = env("MALD_CACHE", "cache")
JUDGE = env("MALD_JUDGE", "judge")
REPORT = REPO / "report"
