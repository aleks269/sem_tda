"""
Module for persistence homology analysis.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gudhi
import pandas as pd
from rich.progress import track

from .base import Analyzer


class PersistenceAnalyzer(Analyzer):

    def __init__(self, data_container=None):
        super().__init__(data_container)

    def analyze(self, datasets, max_edge_length):
        """
        Parameters
        ----------
        max_edge_length : float
            Use <= 1.0 for SEM normalized [0,1] data.
            Use ~100 for AFM nm-scale data.
        """
        for file_path in track(datasets, description="[green]Processing persistence..."):
            self._process_persistence(file_path, max_edge_length)

    def _process_persistence(self, file_path, max_edge_length):
        df = pd.read_csv(file_path)
        # Drop DataLine to get square (N, N) distance matrix
        X = df.drop(columns=["DataLine"]).to_numpy()

        gudhi.persistence_graphical_tools._gudhi_matplotlib_use_tex = False

        rips_complex = gudhi.RipsComplex(distance_matrix=X, max_edge_length=max_edge_length)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
        diag = simplex_tree.persistence(min_persistence=0)

        self.data.add_persistence_diagram(file_path, diag)

        base_path = os.path.dirname(file_path)

        records = self._extract_list_from_raw_data(diag)
        diag_df = pd.DataFrame(records, columns=["Start", "End", "Length", "Homology group"])
        diag_df.to_csv(os.path.join(base_path, "diag_df_output.csv"), index=False)

        self._save_barcode(base_path, diag, len(records))
        self._save_persistence_diagram(base_path, diag, len(records))

    def _extract_list_from_raw_data(self, diagrams):
        records = []
        for dim, (birth, death) in diagrams:
            records.append([birth, death, abs(birth - death), dim])
        return records

    def _save_barcode(self, base_path, diag, diag_length):
        self.plt_config.apply()
        gudhi.plot_persistence_barcode(
            diag, fontsize=18, legend=True, inf_delta=0.5, max_intervals=diag_length + 1
        )
        plt.xlabel("Filtration value (a.u.)", fontsize=16)
        plt.ylabel("Topological invariants", fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=0)

        plt.savefig(os.path.join(base_path, "barcode.png"),
                    format="png", dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(base_path, "barcode.svg"),
                    format="svg", bbox_inches="tight")
        plt.savefig(os.path.join(base_path, "barcode.pdf"),
                    format="pdf", bbox_inches="tight")
        plt.close("all")

    def _save_persistence_diagram(self, base_path, diag, diag_length):
        self.plt_config.apply()
        gudhi.plot_persistence_diagram(
            diag,
            fontsize=18,
            alpha=0.5,
            legend=True,
            inf_delta=0.2,
            greyblock=False,
            max_intervals=diag_length + 1,
        )
        plt.xlabel("Feature appearance (a.u.)", fontsize=18)
        plt.ylabel("Feature disappearance (a.u.)", fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.savefig(os.path.join(base_path, "persistence_diagram.png"),
                    format="png", dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(base_path, "persistence_diagram.svg"),
                    format="svg", bbox_inches="tight")
        plt.savefig(os.path.join(base_path, "persistence_diagram.pdf"),
                    format="pdf", bbox_inches="tight")
        plt.close("all")