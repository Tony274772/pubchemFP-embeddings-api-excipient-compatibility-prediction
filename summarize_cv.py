"""Print a clean mean +/- std summary of 5-fold CV metrics from cv_metrics.json.

Usage:
    python summarize_cv.py [--metrics-path path/to/cv_metrics.json]
"""

import argparse
import os

from src.evaluate import print_cv_summary


def main():
    parser = argparse.ArgumentParser(description="Summarize cross-validation metrics.")
    parser.add_argument(
        "--metrics-path",
        default=os.path.join("metrics", "molformer", "cv_metrics.json"),
        help="Path to cv_metrics.json (default: metrics/molformer/cv_metrics.json)",
    )
    args = parser.parse_args()
    print_cv_summary(args.metrics_path)


if __name__ == "__main__":
    main()
