"""dF/F baseline: the vectorised implementation must match the original loop exactly."""
import numpy as np
import pytest

torch = pytest.importorskip('torch', reason='the cmn analysis module imports torch at module level')

from entarchy_vxpy_suite2p.analysis.cmn import functions


def reference_baseline(fluorescence, window_size, percentile):
    """The previous per-sample implementation, kept here as the reference."""
    if window_size % 2 == 0:
        window_size += 1
    half_window_size = int((window_size - 1) // 2)

    f_padded = np.pad(fluorescence, half_window_size, mode='empty')
    f_padded[:half_window_size] = np.median(fluorescence[:half_window_size])
    f_padded[-half_window_size:] = np.median(fluorescence[-half_window_size:])

    baseline = np.zeros(fluorescence.shape)
    for i in range(baseline.shape[0]):
        fsub = f_padded[i:i + window_size]
        baseline[i] = np.mean(fsub[fsub < np.percentile(fsub, percentile)])

    return baseline


class TestMatchesReference:

    @pytest.mark.parametrize('n_samples,window_size,percentile', [
        (200, 51, 10),
        (200, 50, 10),      # even window sizes are widened to odd
        (500, 201, 10),
        (500, 201, 25),
        (500, 201, 1),
        (137, 1201, 10),    # window longer than the signal
        (60, 11, 50),
    ])
    def test_identical_to_the_loop(self, n_samples, window_size, percentile):
        rng = np.random.default_rng(n_samples + window_size + percentile)
        signal = 500 + 50 * rng.random(n_samples) + 20 * np.sin(np.arange(n_samples) / 7)

        expected = reference_baseline(signal, window_size, percentile)
        actual = functions.rolling_baseline(signal, window_size, percentile)

        assert np.allclose(expected, actual, equal_nan=True, rtol=0, atol=1e-12)

    def test_chunking_does_not_change_the_result(self):
        rng = np.random.default_rng(4)
        signal = 500 + rng.normal(size=400)

        whole = functions.rolling_baseline(signal, 101, 10, chunk_size=10_000)
        chunked = functions.rolling_baseline(signal, 101, 10, chunk_size=7)

        assert np.allclose(whole, chunked, equal_nan=True)

    def test_monotonic_signal(self):
        signal = np.linspace(100.0, 200.0, 300)

        expected = reference_baseline(signal, 51, 10)
        actual = functions.rolling_baseline(signal, 51, 10)

        assert np.allclose(expected, actual, equal_nan=True)


class TestProperties:

    def test_baseline_tracks_the_lower_envelope(self):
        """The baseline sits below the signal for a trace with positive transients."""
        rng = np.random.default_rng(7)
        signal = 500 + rng.normal(scale=1.0, size=400)
        signal[150:170] += 100.0  # a transient

        baseline = functions.rolling_baseline(signal, 101, 10)

        assert np.all(baseline[np.isfinite(baseline)] < signal.max())
        assert baseline[160] < signal[160]

    def test_output_shape_and_dtype(self):
        signal = np.random.default_rng(1).random(250) + 10
        baseline = functions.rolling_baseline(signal, 51, 10)

        assert baseline.shape == signal.shape
        assert baseline.dtype == np.float64

    def test_constant_signal_yields_nan(self):
        """No sample falls below the percentile of a flat trace, so the baseline is
        undefined; preserved from the original implementation."""
        signal = np.full(200, 500.0)

        with np.errstate(invalid='ignore'):
            with pytest.warns(RuntimeWarning):
                baseline = functions.rolling_baseline(signal, 51, 10)

        assert np.isnan(baseline).all()

    def test_does_not_mutate_the_input(self):
        signal = 500 + np.random.default_rng(2).random(200)
        original = signal.copy()

        functions.rolling_baseline(signal, 51, 10)

        assert np.array_equal(signal, original)


class TestCalculateDff:

    def test_writes_dff_to_the_roi(self):
        class FakeImaging(dict):
            pass

        class FakeRoi(dict):
            imaging = FakeImaging({'rate': 10.0})

        rng = np.random.default_rng(11)
        roi = FakeRoi({'fluorescence': 500 + 20 * rng.random(300)})

        functions.calculate_dff(roi, window_size=5, percentile=10)

        assert 'dff' in roi
        assert roi['dff'].shape == roi['fluorescence'].shape
        assert np.all(np.isfinite(roi['dff']))

    def test_matches_the_full_original_calculation(self):
        class FakeImaging(dict):
            pass

        class FakeRoi(dict):
            imaging = FakeImaging({'rate': 10.0})

        rng = np.random.default_rng(12)
        fluorescence = 500 + 20 * rng.random(400)
        roi = FakeRoi({'fluorescence': fluorescence})

        functions.calculate_dff(roi, window_size=6, percentile=10)

        expected_baseline = reference_baseline(fluorescence, int(6 * 10.0), 10)
        expected = (fluorescence - expected_baseline) / expected_baseline

        assert np.allclose(roi['dff'], expected, equal_nan=True)
