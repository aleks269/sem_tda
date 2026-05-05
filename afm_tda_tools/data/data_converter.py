"""
Module for preprocessing raw AFM/SEM `.txt` files into CSV format.
"""

from pathlib import Path

import pandas as pd
from rich.progress import track


def _detect_skiprows(txt_file: Path) -> int:
    """Count non-numeric header lines at top of file."""
    with open(txt_file, encoding="latin-1") as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                float(stripped.split()[0])
                return i
            except ValueError:
                continue
    return 0


def txt_to_csv_folder(raw_data_path, processed_path, multiply_const=1e9, crop_size=None):
    """
    Convert all `.txt` files under a directory into CSV files.

    Parameters
    ----------
    raw_data_path : str or Path
    processed_path : str or Path
    multiply_const : float, default 1e9
        1e9 for AFM (m → nm), 1 for SEM normalized intensity.
    crop_size : int or None
        If set, crop each matrix to (crop_size × crop_size) before saving.
        Reduces GUDHI memory usage. None = no crop.
    """
    raw = Path(raw_data_path)
    proc = Path(processed_path)
    proc.mkdir(parents=True, exist_ok=True)

    txt_files = list(raw.rglob("*.txt"))
    for txt_file in track(txt_files, description="[green]Preprocessing txt->csv..."):
        skiprows = _detect_skiprows(txt_file)

        df = pd.read_csv(
            txt_file,
            sep=r"\s+",
            skiprows=skiprows,
            header=None,
            engine="python",
            encoding="latin-1",
        )
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all") * multiply_const

        if df.empty or df.shape[1] == 0:
            print(f"\n[SKIP] {txt_file.name}: empty after parsing")
            continue

        # Optional crop to square sub-matrix
        if crop_size is not None:
            n = min(crop_size, len(df), df.shape[1])
            df = df.iloc[:n, :n].copy()

        dataline_col = pd.Series(range(len(df)), name="DataLine")
        pos_cols = [f"Pos = {i}" for i in range(df.shape[1])]
        df.columns = pos_cols
        df = pd.concat([dataline_col, df], axis=1)

        stem = txt_file.stem
        out_dir = proc / stem
        out_dir.mkdir(exist_ok=True)
        df.to_csv(out_dir / f"{stem}.csv", index=False)