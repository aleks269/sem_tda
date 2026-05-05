"""
Core unit tests for sem_tda / afm_tda_tools.

Groups:
  A — ACF estimator mathematical invariants
  B — Distance matrix shape (DataLine bug regression)
  C — Data converter
  D — Integration smoke test (GUDHI)

All tests are self-contained: synthetic data only, no real data files required.
Run: pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from afm_tda_tools.analyzers.autocorrelation import _compute_acf_unbiased
from afm_tda_tools.data.data_converter import txt_to_csv_folder


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_ar1(n: int, phi: float, seed: int) -> np.ndarray:
    """AR(1) process: approximates exponentially correlated surface."""
    rng = np.random.default_rng(seed)
    series = np.zeros(n)
    noise = rng.normal(scale=np.sqrt(1 - phi ** 2), size=n)
    for i in range(1, n):
        series[i] = phi * series[i - 1] + noise[i]
    return series


def _make_mock_csv(n: int, seed: int = 99) -> pd.DataFrame:
    """DataFrame with DataLine + n Pos columns, mimicking data_converter output."""
    rng = np.random.default_rng(seed)
    data = {"DataLine": list(range(n))}
    data.update({f"Pos = {i}": rng.uniform(0, 1, size=n).tolist() for i in range(n)})
    return pd.DataFrame(data)


def _biased_acf(series: np.ndarray, nlags: int) -> np.ndarray:
    """Reference biased ACF (divides by N, not N-lag) — no external deps."""
    series = series - series.mean()
    var = np.var(series)
    if var == 0:
        return np.zeros(nlags + 1)
    n = len(series)
    return np.array([
        np.dot(series[:n - k], series[k:]) / (n * var)
        for k in range(nlags + 1)
    ])


# ─── Group A: ACF Estimator Invariants ────────────────────────────────────────

class TestACFInvariants:

    def test_c0_equals_one_white_noise(self):
        """C(0) = 1 for white noise."""
        series = np.random.default_rng(42).normal(50, 10, 400)
        acf = _compute_acf_unbiased(series, nlags=len(series) - 1)
        assert abs(acf[0] - 1.0) < 1e-12, f"C(0) = {acf[0]}, expected 1.0"

    def test_c0_equals_one_correlated(self):
        """C(0) = 1 for an AR(1) process."""
        series = _make_ar1(500, phi=np.exp(-1 / 30), seed=0)
        acf = _compute_acf_unbiased(series, nlags=50)
        assert abs(acf[0] - 1.0) < 1e-12

    def test_c0_equals_one_uniform(self):
        """C(0) = 1 for SEM-like uniform [0,1] intensity data."""
        series = np.random.default_rng(1).uniform(0, 1, 512)
        acf = _compute_acf_unbiased(series, nlags=100)
        assert abs(acf[0] - 1.0) < 1e-12

    def test_monotone_decay_for_ar1(self):
        """Unbiased ACF of AR(1) must be non-increasing for small lags."""
        series = _make_ar1(1000, phi=np.exp(-1 / 30), seed=1)
        acf = _compute_acf_unbiased(series, nlags=10)
        for lag in range(1, len(acf)):
            assert acf[lag] <= acf[lag - 1] + 1e-6, (
                f"Not monotone at lag {lag}: {acf[lag - 1]:.4f} -> {acf[lag]:.4f}"
            )

    def test_zero_variance_returns_zeros(self):
        """Constant series (zero variance) must return all zeros without raising."""
        acf = _compute_acf_unbiased(np.ones(100) * 5.0, nlags=10)
        assert acf.shape == (11,)
        assert np.all(acf == 0.0)

    def test_output_length(self):
        """Output length must be nlags + 1."""
        series = np.random.default_rng(10).normal(size=200)
        for nlags in [5, 50, 199]:
            assert len(_compute_acf_unbiased(series, nlags=nlags)) == nlags + 1

    def test_nlags_quarter_stays_bounded(self):
        """
        With nlags capped at n//4, ACF must stay within [-1.5, 1.5].
        This is the blow-up regression test: the unbiased estimator diverges
        at large lags (N-lag -> 1); capping at n//4 prevents this.
        """
        series = np.random.default_rng(7).uniform(0, 1, 200)
        nlags = len(series) // 4
        acf = _compute_acf_unbiased(series, nlags=nlags)
        assert np.all(np.abs(acf) <= 1.5), (
            f"ACF out of bounds: min={acf.min():.3f}, max={acf.max():.3f}"
        )

    def test_unbiased_larger_than_biased_at_large_lag(self):
        """
        Bug 1 regression: unbiased ACF > biased at large lag for
        long-range correlated signal.
        Uses a manual biased implementation — no statsmodels dependency.
        """
        series = _make_ar1(400, phi=np.exp(-1 / 100), seed=7)
        lag = 100  # n//4, safe range for unbiased
        unbiased = _compute_acf_unbiased(series, nlags=lag)[lag]
        biased = _biased_acf(series, nlags=lag)[lag]
        assert abs(unbiased) >= abs(biased) - 1e-10, (
            f"|unbiased|={abs(unbiased):.4f} should be >= |biased|={abs(biased):.4f}"
        )


# ─── Group B: Distance Matrix Shape ───────────────────────────────────────────

class TestDistanceMatrixShape:

    def test_drop_dataline_produces_square(self):
        """Regression test for Bug 2: DataLine removed -> square (N, N) matrix."""
        df = _make_mock_csv(10)
        X = df.drop(columns=["DataLine"]).to_numpy()
        assert X.shape == (10, 10)

    def test_without_fix_matrix_is_non_square(self):
        """Documents pre-fix shape: DataLine included -> (N, N+1)."""
        df = _make_mock_csv(10)
        assert df.to_numpy().shape == (10, 11)

    def test_dataline_column_is_sequential(self):
        """DataLine column must be 0..N-1."""
        df = _make_mock_csv(20)
        assert list(df["DataLine"]) == list(range(20))

    def test_crop_size_gives_correct_shape(self):
        """After crop, matrix must be exactly crop_size x crop_size."""
        df = _make_mock_csv(50)
        crop = 30
        n = min(crop, len(df), df.shape[1] - 1)
        X = df.iloc[:n, 1:n + 1].to_numpy()
        assert X.shape == (crop, crop)

    def test_gudhi_accepts_square_matrix(self):
        """GUDHI RipsComplex must not raise on a valid (N, N) matrix."""
        import gudhi
        rng = np.random.default_rng(3)
        n = 15
        M = rng.uniform(0, 1, size=(n, n))
        M = (M + M.T) / 2
        np.fill_diagonal(M, 0)
        rc = gudhi.RipsComplex(distance_matrix=M.tolist(), max_edge_length=1.0)
        st = rc.create_simplex_tree(max_dimension=2)
        diag = st.persistence(min_persistence=0)
        assert len(diag) > 0


# ─── Group C: Data Converter ──────────────────────────────────────────────────

class TestDataConverter:

    def _write_txt(self, path, n_rows, n_cols, n_headers=4, seed=5):
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        with open(path, "w") as f:
            for _ in range(n_headers):
                f.write("# header\n")
            for _ in range(n_rows):
                row = rng.uniform(0, 1e-8, size=n_cols)
                f.write("\t".join(f"{v:.10e}" for v in row) + "\n")

    def test_output_shape(self, tmp_path):
        """CSV must have shape (n_rows, n_cols + 1): +1 for DataLine."""
        n_rows, n_cols = 30, 30
        txt = tmp_path / "raw" / "sample.txt"
        self._write_txt(txt, n_rows, n_cols)
        txt_to_csv_folder(str(txt.parent), str(tmp_path / "proc"))
        df = pd.read_csv(tmp_path / "proc" / "sample" / "sample.csv")
        assert df.shape == (n_rows, n_cols + 1)
        assert "DataLine" in df.columns

    def test_scale_applied(self, tmp_path):
        """multiply_const must scale all data values correctly."""
        txt = tmp_path / "raw" / "s.txt"
        txt.parent.mkdir(parents=True, exist_ok=True)
        with open(txt, "w") as f:
            f.write("1.0\t2.0\t3.0\n4.0\t5.0\t6.0\n7.0\t8.0\t9.0\n")
        txt_to_csv_folder(str(txt.parent), str(tmp_path / "proc"), multiply_const=10.0)
        df = pd.read_csv(tmp_path / "proc" / "s" / "s.csv")
        assert abs(df["Pos = 0"].iloc[0] - 10.0) < 1e-9

    def test_dataline_starts_at_zero(self, tmp_path):
        """DataLine must be 0..N-1."""
        txt = tmp_path / "raw" / "t.txt"
        self._write_txt(txt, 10, 10)
        txt_to_csv_folder(str(txt.parent), str(tmp_path / "proc"))
        df = pd.read_csv(tmp_path / "proc" / "t" / "t.csv")
        assert list(df["DataLine"]) == list(range(10))

    def test_zero_header_file(self, tmp_path):
        """Files with no header lines (SEM format) must parse correctly."""
        txt = tmp_path / "raw" / "noheader.txt"
        txt.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        with open(txt, "w") as f:
            for _ in range(5):
                row = rng.uniform(0, 1, size=5)
                f.write("\t".join(f"{v:.6f}" for v in row) + "\n")
        txt_to_csv_folder(str(txt.parent), str(tmp_path / "proc"))
        df = pd.read_csv(tmp_path / "proc" / "noheader" / "noheader.csv")
        assert df.shape == (5, 6)

    def test_crop_size_applied(self, tmp_path):
        """crop_size must produce square matrix smaller than original."""
        n_rows, n_cols, crop = 20, 20, 10
        txt = tmp_path / "raw" / "crop.txt"
        self._write_txt(txt, n_rows, n_cols, n_headers=0)
        txt_to_csv_folder(str(txt.parent), str(tmp_path / "proc"), crop_size=crop)
        df = pd.read_csv(tmp_path / "proc" / "crop" / "crop.csv")
        assert df.shape == (crop, crop + 1)