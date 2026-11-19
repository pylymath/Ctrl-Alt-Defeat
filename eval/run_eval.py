"""Run the full pipeline over a practice-pack style folder and score it.

    python eval/run_eval.py                         # all 100 items
    python eval/run_eval.py --modality code image   # some input types
    python eval/run_eval.py --ids p_img_001 p_vid_009
    NEUROASSIST_COUNT_MODE=direct python eval/run_eval.py --modality image --tag direct

The app only ever sees the question and the file. answers.json is read here, for scoring only.
Writes eval/results/<timestamp>[_tag].csv and .md (git-ignored: they contain reference answers)."""
import argparse
import csv
import json
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroassist import config, llm                      # noqa: E402
from neuroassist.pipeline import Session, answer        # noqa: E402
import scorers                                           # noqa: E402

ORDER = ["image", "audio", "text", "video", "code"]


def needed_models(modalities):
    m = set()
    if {"image", "video", "text", "audio"} & set(modalities):
        m.add(config.VISION_MODEL if {"image", "video"} & set(modalities) else config.TEXT_MODEL)
        m.add(config.TEXT_MODEL)
    if "code" in modalities:
        m.add(config.CODE_MODEL)
    return m


def score(item, ans_ref, result, data_dir):
    mod, q = item["modality"], item["question"]
    if mod == "image":
        return scorers.score_image(q, result.answer, ans_ref)
    if mod == "audio":
        return scorers.score_audio(q, result.answer, ans_ref)
    if mod == "text":
        return scorers.score_text(q, result.answer, ans_ref, result.abstain)
    if mod == "video":
        return scorers.score_video(q, result.answer, ans_ref)
    return scorers.score_code(result.answer, data_dir / "code" / "tests" / f"{item['id']}_test.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=config.DATA_DIR)
    ap.add_argument("--modality", nargs="*", choices=ORDER)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    items = json.loads((args.data / "items.json").read_text(encoding="utf-8"))
    refs = {a["id"]: a for a in json.loads((args.data / "answers.json").read_text(encoding="utf-8"))}
    if args.modality:
        items = [i for i in items if i["modality"] in args.modality]
    if args.ids:
        items = [i for i in items if i["id"] in args.ids]
    items.sort(key=lambda i: ORDER.index(i["modality"]))        # fewest model swaps
    if args.limit:
        items = items[: args.limit]

    mods = sorted({i["modality"] for i in items}, key=ORDER.index)
    have = llm.installed_models()
    missing = [m for m in needed_models(mods) if m not in have]
    print(f"profile={config.PROFILE} vision={config.VISION_MODEL} text={config.TEXT_MODEL} "
          f"code={config.CODE_MODEL} whisper={config.WHISPER_MODEL}")
    if missing:
        sys.exit("Missing Ollama models: " + ", ".join(missing) + "\nPull with: "
                 + " ; ".join(f"ollama pull {m}" for m in missing))

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + (f"_{args.tag}" if args.tag else "")
    rows = []
    for n, item in enumerate(items, 1):
        files = [str(args.data / item["file"])] if item.get("file") else []
        ref = refs[item["id"]]
        t = time.perf_counter()
        try:
            res = answer(item["question"], files, session=Session(), extras=False)     # fresh session: no leakage
            ok, sc, how = score(item, ref, res, args.data)
            err, pred, task, abst = "", res.answer, res.route.task if res.route else "", res.abstain
        except Exception as e:
            ok, sc, how, pred, task, abst = False, 0.0, "error", "", "", False
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc(limit=2)
        secs = time.perf_counter() - t
        rows.append({"id": item["id"], "modality": item["modality"], "difficulty": ref.get("difficulty", ""),
                     "task": task, "pass": int(ok), "score": round(sc, 3), "metric": how,
                     "latency_s": round(secs, 2), "abstain": int(abst),
                     "question": item["question"][:200], "prediction": str(pred)[:500],
                     "reference": str(ref.get("reference_answer"))[:300], "error": err})
        print(f"[{n}/{len(items)}] {item['id']:<11} {'PASS' if ok else 'fail'}  {secs:5.1f}s  "
              f"{task:<14} {str(pred)[:60]!r}" + (f"  ERROR {err[:80]}" if err else ""))

    with open(out_dir / f"{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(list)
    for r in rows:
        by[r["modality"]].append(r)
    lines = [f"# Eval {stamp}", "",
             f"Profile `{config.PROFILE}` · vision `{config.VISION_MODEL}` · text `{config.TEXT_MODEL}` · "
             f"code `{config.CODE_MODEL}` · whisper `{config.WHISPER_MODEL}` · count `{config.COUNT_MODE}` · "
             f"video `{config.VIDEO_MODE}`", "",
             "| Input | Passed | Easy | Medium | Hard | Median latency | Errors |",
             "|---|---:|---:|---:|---:|---:|---:|"]

    def frac(rs):
        return f"{sum(r['pass'] for r in rs)}/{len(rs)}" if rs else "–"

    for m in mods:
        rs = by[m]
        d = {k: [r for r in rs if r["difficulty"] == k] for k in ("easy", "medium", "hard")}
        lines.append(f"| {m} | **{frac(rs)}** | {frac(d['easy'])} | {frac(d['medium'])} | {frac(d['hard'])} | "
                     f"{statistics.median(r['latency_s'] for r in rs):.1f}s | {sum(bool(r['error']) for r in rs)} |")
    lines.append(f"| **all** | **{frac(rows)}** | | | | "
                 f"{statistics.median(r['latency_s'] for r in rows):.1f}s | {sum(bool(r['error']) for r in rows)} |")
    (out_dir / f"{stamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[2:]))
    print(f"\nDetails: {out_dir / (stamp + '.csv')}")


if __name__ == "__main__":
    main()
