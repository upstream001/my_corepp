#!/usr/bin/env python3
import argparse
import csv
import json
import os
from statistics import mean, median

import open3d as o3d

from metrics_3d.chamfer_distance import ChamferDistance
from metrics_3d.precision_recall import PrecisionRecall


def load_split_ids(split_json_path):
    with open(split_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["StrawberryDataset"]["Strawberry"]


def load_prediction_geometry(pred_path):
    mesh = o3d.io.read_triangle_mesh(pred_path)
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        return mesh
    return o3d.io.read_point_cloud(pred_path)


def evaluate_one(gt_path, pred_path, threshold):
    gt = o3d.io.read_point_cloud(gt_path)
    pred = load_prediction_geometry(pred_path)

    cd = ChamferDistance()
    cd.reset()
    cd.update(gt, pred)
    chamfer = cd.compute(print_output=False)

    pr = PrecisionRecall(0.001, 0.01, 10)
    pr.reset()
    pr.update(gt, pred)
    precision, recall, f1, used_t = pr.compute_at_threshold(threshold, print_output=False)

    return chamfer, precision, recall, f1, used_t


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate strawberry reconstructed surfaces against ground truth point clouds."
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="/home/tianqi/my_corepp/logs/strawberry/test_results",
        help="Directory containing predicted .ply files.",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/home/tianqi/my_corepp/data/strawberry/complete",
        help="Directory containing ground-truth .ply files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="/home/tianqi/my_corepp/deepsdf/experiments/splits/strawberry_test.json",
        help="Split json that defines instance ids to evaluate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.005,
        help="Distance threshold for precision/recall/f1.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="/home/tianqi/my_corepp/logs/strawberry/test_results/metrics.csv",
        help="Output CSV for per-sample metrics.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="/home/tianqi/my_corepp/logs/strawberry/test_results/summary.json",
        help="Output JSON for overall summary.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if one sample is missing.",
    )
    args = parser.parse_args()

    ids = load_split_ids(args.split)
    rows = []
    missing = []
    used_threshold = None

    for instance_id in ids:
        gt_path = os.path.join(args.gt_dir, f"{instance_id}.ply")
        pred_path = os.path.join(args.pred_dir, f"{instance_id}.ply")

        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            missing.append(
                {
                    "id": instance_id,
                    "gt_exists": os.path.exists(gt_path),
                    "pred_exists": os.path.exists(pred_path),
                }
            )
            if args.strict:
                raise FileNotFoundError(f"Missing files for {instance_id}: gt={gt_path}, pred={pred_path}")
            continue

        chamfer, precision, recall, f1, used_threshold = evaluate_one(
            gt_path, pred_path, args.threshold
        )
        rows.append(
            {
                "id": instance_id,
                "chamfer_distance": float(chamfer),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "chamfer_distance", "precision", "recall", "f1"]
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_expected": len(ids),
        "num_evaluated": len(rows),
        "num_missing": len(missing),
        "missing": missing,
        "threshold_used": used_threshold,
    }

    if rows:
        cd_vals = [r["chamfer_distance"] for r in rows]
        p_vals = [r["precision"] for r in rows]
        r_vals = [r["recall"] for r in rows]
        f_vals = [r["f1"] for r in rows]
        summary.update(
            {
                "chamfer_mean": mean(cd_vals),
                "chamfer_median": median(cd_vals),
                "precision_mean": mean(p_vals),
                "recall_mean": mean(r_vals),
                "f1_mean": mean(f_vals),
            }
        )

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Per-sample metrics saved: {args.out_csv}")
    print(f"Summary saved: {args.out_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
