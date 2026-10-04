"""Tests for har.data.normalize.ChannelStandardizer."""

from __future__ import annotations

import json

import numpy as np
import pytest

from har.data.normalize import STD_FLOOR, ChannelStandardizer


def _windows(n: int = 200, t: int = 128, seed: int = 0) -> np.ndarray:
    """(n, t, 3) float32 windows whose channels have very different means and scales."""
    rng = np.random.default_rng(seed)
    loc, scale = np.array([5.0, -2.0, 0.0]), np.array([0.1, 3.0, 50.0])
    return (loc + scale * rng.standard_normal((n, t, 3))).astype(np.float32)


def test_fitted_data_has_zero_mean_unit_std_per_channel() -> None:
    x = _windows()
    z = ChannelStandardizer().fit_transform(x)
    assert z.shape == x.shape and z.dtype == np.float32
    np.testing.assert_allclose(z.mean(axis=(0, 1)), 0.0, atol=1e-5)
    np.testing.assert_allclose(z.std(axis=(0, 1)), 1.0, atol=1e-5)


def test_statistics_are_per_channel_over_windows_and_time() -> None:
    x = _windows()
    scaler = ChannelStandardizer().fit(x)
    assert scaler.mean.shape == scaler.std.shape == (3,)
    np.testing.assert_allclose(scaler.mean, x.reshape(-1, 3).mean(axis=0, dtype=np.float64))
    np.testing.assert_allclose(scaler.std, x.reshape(-1, 3).std(axis=0, dtype=np.float64))


def test_transform_uses_the_fitted_statistics_only() -> None:
    train, other = _windows(seed=0), _windows(seed=1) + 1.0
    scaler = ChannelStandardizer().fit(train)
    z = scaler.transform(other)
    # other data is shifted by +1, so its standardised means are 1/std, not 0
    np.testing.assert_allclose(z.mean(axis=(0, 1)), 1.0 / scaler.std, rtol=0.05, atol=0.05)
    assert not np.allclose(z.mean(axis=(0, 1))[0], 0.0, atol=1.0)


def test_transform_does_not_modify_its_input() -> None:
    x = _windows()
    before = x.copy()
    ChannelStandardizer().fit(x).transform(x)
    np.testing.assert_array_equal(x, before)


def test_constant_channel_is_floored_not_nan() -> None:
    x = _windows()
    x[:, :, 1] = 4.0
    scaler = ChannelStandardizer().fit(x)
    assert scaler.std[1] == STD_FLOOR
    z = scaler.transform(x)
    assert np.isfinite(z).all()
    np.testing.assert_array_equal(z[:, :, 1], 0.0)


def test_dict_round_trip_is_json_serialisable_and_exact() -> None:
    x = _windows()
    scaler = ChannelStandardizer().fit(x)
    d = json.loads(json.dumps(scaler.to_dict()))
    restored = ChannelStandardizer.from_dict(d)
    np.testing.assert_array_equal(restored.mean, scaler.mean)
    np.testing.assert_array_equal(restored.std, scaler.std)
    np.testing.assert_array_equal(restored.transform(x), scaler.transform(x))


def test_misuse_raises() -> None:
    with pytest.raises(RuntimeError):
        ChannelStandardizer().transform(_windows())
    with pytest.raises(RuntimeError):
        ChannelStandardizer().to_dict()
    with pytest.raises(ValueError):
        ChannelStandardizer().fit(np.zeros((10, 3)))
    with pytest.raises(ValueError):
        ChannelStandardizer().fit(_windows()).transform(np.zeros((2, 128, 4)))
    with pytest.raises(ValueError):
        ChannelStandardizer.from_dict({"mean": [0.0, 1.0], "std": [1.0]})
