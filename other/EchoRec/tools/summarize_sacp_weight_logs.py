#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


FLOAT_RE = r"(?:np\.float64\()?(?P<value>-?\d+(?:\.\d+)?)(?:\))?"


def extract_metric(block: str, name: str):
    pattern = rf"'{re.escape(name)}': {FLOAT_RE}"
    match = re.search(pattern, block)
    return float(match.group("value")) if match else None


def parse_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")

    best_metric_line = None
    for line in text.splitlines():
        if "Best metric: epoch" in line:
            best_metric_line = line

    if best_metric_line is None:
        return None

    val_match = re.search(r"val=\{.*?\}", best_metric_line)
    test_match = re.search(r"test=\{.*?\}", best_metric_line)
    if not val_match or not test_match:
        return None

    val_block = val_match.group(0)
    test_block = test_match.group(0)
    return {
        "tag": path.stem,
        "val_ndcg10": extract_metric(val_block, "NDCG@10"),
        "val_hr10": extract_metric(val_block, "HR@10"),
        "test_ndcg10": extract_metric(test_block, "NDCG@10"),
        "test_hr10": extract_metric(test_block, "HR@10"),
        "test_ndcg20": extract_metric(test_block, "NDCG@20"),
        "test_hr20": extract_metric(test_block, "HR@20"),
        "log": str(path),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize SACP weight search logs.")
    parser.add_argument(
        "log_dir",
        nargs="?",
        default="./debug_logs/weight_search",
        help="Directory containing *.log files from movies_sacp_weight_search.sh",
    )
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    rows = []
    for path in sorted(log_dir.glob("*.log")):
        row = parse_log(path)
        if row is not None:
            rows.append(row)

    rows.sort(
        key=lambda x: (
            x["val_ndcg10"] if x["val_ndcg10"] is not None else float("-inf"),
            x["test_hr10"] if x["test_hr10"] is not None else float("-inf"),
            x["test_ndcg10"] if x["test_ndcg10"] is not None else float("-inf"),
        ),
        reverse=True,
    )

    if not rows:
        print("No completed logs found.")
        return

    header = (
        f"{'rank':>4}  {'valN10':>7}  {'valH10':>7}  {'testN10':>8}  "
        f"{'testH10':>8}  {'testN20':>8}  {'testH20':>8}  tag"
    )
    print(header)
    print("-" * len(header))

    for idx, row in enumerate(rows[: args.topk], start=1):
        print(
            f"{idx:>4}  "
            f"{row['val_ndcg10']:>7.4f}  "
            f"{row['val_hr10']:>7.4f}  "
            f"{row['test_ndcg10']:>8.4f}  "
            f"{row['test_hr10']:>8.4f}  "
            f"{row['test_ndcg20']:>8.4f}  "
            f"{row['test_hr20']:>8.4f}  "
            f"{row['tag']}"
        )


if __name__ == "__main__":
    main()
