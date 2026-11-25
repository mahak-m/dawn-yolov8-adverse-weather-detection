import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# Trained model checkpoints
RUNS = {
    "baseline_s108": "/content/drive/MyDrive/csc490_runs/baseline_s108/weights/best.pt",
    "weighted_sqrt_s108": "/content/drive/MyDrive/csc490_runs/weighted_sqrt_s108/weights/best.pt",
    "weighted_inverse_s108": "/content/drive/MyDrive/csc490_runs/weighted_inverse_s108/weights/best.pt",
}

# Dataset configuration
DATA_YAML = "/content/data.yaml"

# Output CSV
OUTPUT_CSV = "/content/drive/MyDrive/csc490_runs/comparison_50ep_s108.csv"


def evaluate_models():

    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    rows = []

    for run_name, weights_path in RUNS.items():

        if not Path(weights_path).exists():
            print(f"Skipping {run_name}: checkpoint not found.")
            continue

        print(f"\nEvaluating {run_name}...")

        model = YOLO(weights_path)

        results = model.val(
            data=DATA_YAML,
            split="val",
            imgsz=640,
            verbose=False
        )

        # Overall metrics
        rows.append({
            "run": run_name,
            "class": "ALL",
            "precision": round(float(results.box.mp), 4),
            "recall": round(float(results.box.mr), 4),
            "map50": round(float(results.box.map50), 4),
            "map50_95": round(float(results.box.map), 4),
        })

        # Per-class metrics
        for idx, cls_name in results.names.items():

            p, r, ap50, ap = results.box.class_result(idx)

            rows.append({
                "run": run_name,
                "class": cls_name,
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "map50": round(float(ap50), 4),
                "map50_95": round(float(ap), 4),
            })

    if not rows:
        raise RuntimeError("No models were evaluated. Check checkpoint paths.")

    # Generate comparison table
    df_summary = pd.DataFrame(rows)

    # Save results
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved evaluation results to: {OUTPUT_CSV}")

    print(df_summary.to_string(index=False))

    return df_summary


if __name__ == "__main__":
    evaluate_models()
