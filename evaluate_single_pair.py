#!/usr/bin/env python3
import argparse
import os

from evaluate_strawberry_results import evaluate_one

def main():
    parser = argparse.ArgumentParser(description="Evaluate a single strawberry reconstructed surface against its ground truth.")
    parser.add_argument("pred_path", type=str, help="Path to the predicted .ply file.")
    parser.add_argument("gt_path", type=str, help="Path to the ground truth .ply file.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Distance threshold for precision/recall/f1.")
    args = parser.parse_args()

    if not os.path.exists(args.pred_path):
        print(f"Error: Predicted file not found at {args.pred_path}")
        return
    if not os.path.exists(args.gt_path):
        print(f"Error: Ground truth file not found at {args.gt_path}")
        return

    print(f"Evaluating:\n  Pred: {args.pred_path}\n  GT:   {args.gt_path}\n  Threshold: {args.threshold}\n")

    try:
        chamfer, precision, recall, f1, used_threshold = evaluate_one(args.gt_path, args.pred_path, args.threshold)
        
        print("-" * 30)
        print("Evaluation Results:")
        print("-" * 30)
        print(f"Chamfer Distance : {chamfer:.6f}")
        print(f"Precision        : {precision:.2f}%")
        print(f"Recall           : {recall:.2f}%")
        print(f"F1 Score         : {f1:.2f}%")
        print("-" * 30)
        
    except Exception as e:
        print(f"Evaluation failed with error: {e}")

if __name__ == "__main__":
    main()
