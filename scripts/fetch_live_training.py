"""Fetch unreported immutable training plots; --ack marks displayed snapshots."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "results/live"
CURSOR = LOCAL / "reported.json"
BASE = "https://huggingface.co/datasets/pymlex/reshuege-ru-2026/resolve/main/"


def fetch(path):
    request = urllib.request.Request(BASE + path + "?fresh=" + str(time.time_ns()),
                                     headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ack", nargs="+")
    args = parser.parse_args()
    LOCAL.mkdir(parents=True, exist_ok=True)
    seen = set(json.loads(CURSOR.read_text()) if CURSOR.exists() else [])
    if args.ack:
        seen.update(args.ack)
        CURSOR.write_text(json.dumps(sorted(seen)), encoding="utf-8")
        print(json.dumps({"acknowledged": args.ack})); return
    try:
        latest = json.loads(fetch("artifacts/dpo-qwen3.5-2b/training/latest.json"))
        manifest = json.loads(fetch(latest["manifest"]))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(json.dumps({"status": "waiting", "reports": []})); return
        raise
    reports = []
    for event in manifest["events"]:
        key = f"{manifest['run_id']}:{event['step']}"
        if key in seen:
            continue
        folder = LOCAL / manifest["run_id"] / f"step-{event['step']:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("loss.png", "eval.png", "snapshot.json"):
            path = folder / name
            if not path.exists():
                path.write_bytes(fetch(event["path"] + "/" + name))
        reports.append({**event, "ack": key, "loss_plot": str(folder / "loss.png"), "eval_plot": str(folder / "eval.png")})
    print(json.dumps({"status": manifest["status"], "model_url": manifest.get("model_url"), "reports": reports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
