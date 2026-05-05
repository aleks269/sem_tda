"""
Command-line interface for the SEM/AFM data analysis pipeline.

Usage
-----
python -m afm_tda_tools \
    --data-path /path/to/raw_txt \
    --save-path /path/to/output \
    [--multiply-const 1] \
    [--max-edge-length 0.5] \
    [--crop-size 200]

Notes
-----
- For SEM normalized intensity data: --multiply-const 1, --max-edge-length 0.5
- For AFM height data (meters): --multiply-const 1e9, --max-edge-length 100
- --crop-size 200 is recommended for memory safety (GUDHI Rips complex)
"""

import argparse

from afm_tda_tools.pipeline import AnalysisPipeline


def main():
    parser = argparse.ArgumentParser(description="Run the full SEM/AFM data analysis pipeline.")
    parser.add_argument("--data-path", "-d", required=True,
                        help="Path to directory with raw .txt files.")
    parser.add_argument("--save-path", "-s", required=True,
                        help="Path to directory where results will be saved.")
    parser.add_argument("--width-line", "-w", type=float, default=0.0196,
                        help="Sampling interval for autocorrelation lag axis. Default: 0.0196.")
    parser.add_argument("--max-edge-length", "-e", type=float, default=0.5,
                        help="Max edge length for Rips complex. "
                             "Use <=1.0 for SEM [0,1] data, ~100 for AFM nm data. Default: 0.5.")
    parser.add_argument("--matrix-size", "-m", type=int, default=3,
                        help="Block size n for n×n min-max patch analysis. Default: 3.")
    parser.add_argument("--crop-size", "-z", type=int, default=None,
                        help="Crop input matrices to (crop_size × crop_size) before analysis. "
                             "Recommended: 200. Default: no crop.")
    parser.add_argument("--delta-bottleneck", "-b", type=float, default=0.01,
                        help="Tolerance for bottleneck distance. Default: 0.01.")
    parser.add_argument("--order-wasserstein", "-o", type=float, default=1.0,
                        help="Order for Wasserstein distance. Default: 1.0.")
    parser.add_argument("--multiply-const", "-c", type=float, default=1e9,
                        help="Scaling factor for raw data. 1e9 for AFM, 1 for SEM. Default: 1e9.")
    parser.add_argument("--exclude", "-x", nargs="*",
                        default=["(3x3).csv", "_auto.csv", "output.csv"],
                        help="Additional filename suffixes to exclude from analysis.")

    args = parser.parse_args()

    pipeline = AnalysisPipeline(
        data_path=args.data_path,
        save_path=args.save_path,
        exclude_patterns=args.exclude,
        width_line=args.width_line,
        max_edge_length=args.max_edge_length,
        matrix_size=args.matrix_size,
        delta=args.delta_bottleneck,
        order=args.order_wasserstein,
        multiply_const=args.multiply_const,
        crop_size=args.crop_size,
    )
    pipeline.run()


if __name__ == "__main__":
    main()