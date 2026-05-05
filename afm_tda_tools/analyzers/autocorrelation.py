"""
Module for autocorrelation analysis.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.progress import track

from .base import Analyzer


def _compute_acf_unbiased(series: np.ndarray, nlags: int) -> np.ndarray:
    """
    Unbiased normalized ACF per ISO 25178-2 / Mathematica CorrelationFunction.

    Divides by (N - lag) at each lag, not N.
    C(0) = 1 exactly for any non-constant series.

    Parameters
    ----------
    series : np.ndarray
    nlags : int

    Returns
    -------
    np.ndarray of length (nlags + 1)
    """
    series = series - series.mean()
    var = np.var(series)
    if var == 0:
        return np.zeros(nlags + 1)
    n = len(series)
    result = np.empty(nlags + 1)
    for lag in range(nlags + 1):
        m = n - lag
        result[lag] = np.dot(series[:m], series[lag:]) / (m * var) if m > 0 else 0.0
    return result


class AutocorrelationAnalyzer(Analyzer):

    def __init__(self, data_container=None):
        super().__init__(data_container)

    def analyze(self, datasets, width_line):
        for file_path in track(datasets, description="[green]Processing autocorrelation..."):
            df = pd.read_csv(file_path)

            n_cols = df.shape[1] - 1          # data columns (exclude DataLine)
            n_rows = len(df)
            n = min(n_rows, n_cols)

            # Cap at n//4: unbiased ACF is unreliable at large lags (N-lag → 1)
            # and β* always appears well within the first quarter of the series.
            nlags = max(1, n // 4)

            acf_df = self._get_acf(
                df=df,
                nlags=nlags,
                series_no=n_cols // 2,
                constant=width_line,
                plot_acf=True,
            )
            self.data.add_acf_data(file_path, acf_df)
            self._save_acf_data(file_path, acf_df)

    def _plot_acf_graph(self, acf_df_x, acf_df_y, ax_x, ax_y):
        self.plt_config.apply()

        acf_df_x = acf_df_x.rename(columns={"ACF": "Along x-direction"})
        acf_df_y = acf_df_y.rename(columns={"ACF": "Along y-direction"})

        plt.figure(figsize=(7, 5))
        plt.plot("ix", "Along x-direction", data=acf_df_x, color="darkorange", linewidth=2.5)
        plt.plot("ix", "Along y-direction", data=acf_df_y, color="royalblue", linewidth=2.5)
        plt.axhline(y=0,   xmin=0, xmax=1, linestyle="--", color="black")
        plt.axhline(y=0.1, xmin=0, xmax=1, linestyle="--", color="brown")
        plt.title(f"Autocorrelation along {ax_x}- and {ax_y}-direction")
        plt.legend()
        plt.xlabel("Sampling length, \u03bcm")
        plt.ylabel("Autocorrelation function, C(\u03c4)")
        plt.ylim(-1.1, 1.1)  # enforce physical bounds

    def _get_acf(self, df, nlags, series_no, constant, plot_acf=False):
        val_x = df[f"Pos = {series_no}"].values
        ax_x = "x"

        val_y = df.set_index("DataLine").T.iloc[:, series_no].values
        ax_y = "y"

        auto_corr_x = _compute_acf_unbiased(val_x, nlags)
        auto_corr_y = _compute_acf_unbiased(val_y, nlags)

        acf_df_x = pd.DataFrame({
            "z":      val_x[:nlags + 1],
            "ACF":    auto_corr_x,
            "ix":     [i * constant for i in range(nlags + 1)],
            "Series": [series_no] * (nlags + 1),
            "Axis":   [ax_x] * (nlags + 1),
        })
        acf_df_y = pd.DataFrame({
            "z":      val_y[:nlags + 1],
            "ACF":    auto_corr_y,
            "ix":     [i * constant for i in range(nlags + 1)],
            "Series": [series_no] * (nlags + 1),
            "Axis":   [ax_y] * (nlags + 1),
        })

        acf_df = pd.concat([acf_df_x, acf_df_y])

        if plot_acf:
            self._plot_acf_graph(acf_df_x, acf_df_y, ax_x, ax_y)
        return acf_df

    def _save_acf_data(self, file_path, acf_df):
        base_path = os.path.dirname(file_path)
        acf_df.to_csv(os.path.join(base_path, "autocorr.csv"))
        plt.savefig(os.path.join(base_path, "autocorr_function.png"),
                    format="png", dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(base_path, "autocorr_function.svg"),
                    format="svg", bbox_inches="tight")
        plt.savefig(os.path.join(base_path, "autocorr_function.pdf"),
                    format="pdf", bbox_inches="tight")
        plt.close("all")