import argparse
import csv
import json
from pathlib import Path


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}{key}.", child, out)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix[:-1]] = value


def read_records(metrics_path):
    records = []
    if not metrics_path.exists():
        return records
    with metrics_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_run(run_dir):
    row = {
        "run_dir": str(run_dir),
        "task": run_dir.parts[-4] if len(run_dir.parts) >= 4 else "",
        "variant": run_dir.parts[-3] if len(run_dir.parts) >= 3 else "",
        "seed": run_dir.parts[-2] if len(run_dir.parts) >= 2 else "",
        "phase": run_dir.parts[-1],
    }
    records = read_records(run_dir / "metrics.jsonl")
    for record in records:
        stage = record.get("stage", "unknown")
        flat = {}
        flatten("", record, flat)
        for key, value in flat.items():
            if key == "stage":
                continue
            row[f"last_{stage}.{key}"] = value
    return row


def write_markdown(rows, path):
    selected = [
        "task",
        "variant",
        "seed",
        "phase",
        "last_real_env_eval.metrics.success_rate",
        "last_real_env_eval.evaluation_time",
        "last_train_epoch.train/loss_epoch",
        "last_val_epoch.val/loss_epoch",
        "run_dir",
    ]
    present = [key for key in selected if any(key in row for row in rows)]
    with path.open("w") as f:
        f.write("# Flow 2x2 Summary\n\n")
        if not rows:
            f.write("No runs found.\n")
            return
        f.write("| " + " | ".join(present) + " |\n")
        f.write("| " + " | ".join(["---"] * len(present)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(key, "")) for key in present) + " |\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    run_dirs = sorted(path.parent for path in args.root.rglob("metrics.jsonl"))
    rows = [summarize_run(run_dir) for run_dir in run_dirs]
    keys = sorted({key for row in rows for key in row})

    csv_path = args.root / "summary.csv"
    md_path = args.root / "summary.md"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
