"""
Module defining the analysis pipeline for SEM/AFM data processing.
"""

import os

from afm_tda_tools.analyzers import (
    AutocorrelationAnalyzer,
    BottleneckAnalyzer,
    MinMaxAnalyzer,
    PersistenceAnalyzer,
)
from afm_tda_tools.data import AnalysisData, txt_to_csv_folder

# These CSVs are pipeline outputs and must never be fed back as inputs
_ALWAYS_EXCLUDE = {
    "autocorr.csv",
    "diag_df_output.csv",
    "distances.csv",
}


class AnalysisPipeline:
    """
    Orchestrates the full analysis workflow on SEM/AFM datasets.

    Parameters
    ----------
    data_path : str
    save_path : str
    exclude_patterns : list of str, optional
    width_line : float, default 0.0196
    max_edge_length : float, default 0.5
        For SEM normalized data (values in [0,1]) use <= 1.0.
        For AFM nm-scale data use larger values (e.g. 100).
    matrix_size : int, default 3
        Block size for min-max patch analysis.
    delta : float, default 0.01
    order : float, default 1.0
    multiply_const : float, default 1e9
    crop_size : int or None
        Crop each input matrix to (crop_size × crop_size) before analysis.
        Recommended: 200 for SEM data to keep GUDHI memory safe.
    """

    def __init__(
        self,
        data_path,
        save_path,
        exclude_patterns=None,
        width_line=0.0196,
        max_edge_length=0.5,
        matrix_size=3,
        delta=0.01,
        order=1.0,
        multiply_const=1e9,
        crop_size=None,
    ):
        self.data_path = data_path
        self.save_path = save_path
        self.multiply_const = multiply_const
        self.crop_size = crop_size

        # Always exclude pipeline outputs regardless of CLI args
        user_exclude = set(exclude_patterns or ["(3x3).csv", "_auto.csv", "output.csv"])
        self.exclude_patterns = list(user_exclude | _ALWAYS_EXCLUDE)

        self.data_container = AnalysisData()
        self.acf_analyzer = AutocorrelationAnalyzer(self.data_container)
        self.persistence_analyzer = PersistenceAnalyzer(self.data_container)
        self.minmax_analyzer = MinMaxAnalyzer(self.data_container)
        self.bottleneck_analyzer = BottleneckAnalyzer(self.data_container)

        self.width_line = width_line
        self.max_edge_length = max_edge_length
        self.matrix_size = matrix_size
        self.delta = delta
        self.order = order

    def run(self):
        txt_to_csv_folder(
            self.data_path,
            self.save_path,
            multiply_const=self.multiply_const,
            crop_size=self.crop_size,
        )

        files = self.acf_analyzer.get_files(
            self.save_path, exclude_patterns=self.exclude_patterns
        )

        self.acf_analyzer.analyze(files, width_line=self.width_line)
        self.persistence_analyzer.analyze(files, max_edge_length=self.max_edge_length)
        self.minmax_analyzer.analyze(files, matrix_size=self.matrix_size)
        # Step 5 + 6 temporarily disabled — large diagrams cause hanging
        # self.bottleneck_analyzer.analyze(
        #     files,
        #     persistence_analyzer=self.persistence_analyzer,
        #     delta=self.delta,
        #     order=self.order,
        # )
        # self.bottleneck_analyzer.save_results(os.path.join(self.save_path))
        print("Pipeline finished successfully.")