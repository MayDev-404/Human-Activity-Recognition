"""Tests for har.data.augment: jitter, per-channel scaling and the augmented training dataset."""

from __future__ import annotations

import pytest
import torch

from har.data.augment import AugmentedTensorDataset, RandomAugment, jitter, scale


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _window(seed: int = 0) -> torch.Tensor:
    return torch.randn(128, 9, generator=_gen(seed))


@pytest.mark.parametrize("shape", [(128, 9), (4, 128, 9)])
def test_transforms_keep_the_shape_and_dtype(shape: tuple[int, ...]) -> None:
    x = torch.randn(shape)
    for out in (jitter(x), scale(x)):
        assert out.shape == shape and out.dtype == torch.float32
    out = RandomAugment(p=1.0)(_window())
    assert out.shape == (128, 9) and out.dtype == torch.float32


def test_jitter_adds_noise_with_the_given_sigma() -> None:
    x = torch.zeros(2000, 128, 9)
    noise = jitter(x, sigma=0.05, generator=_gen(0))
    assert abs(noise.mean().item()) < 1e-3
    assert abs(noise.std().item() - 0.05) < 1e-3


def test_scale_uses_one_factor_per_channel_constant_over_time() -> None:
    x = torch.ones(500, 128, 9)
    out = scale(x, sigma=0.1, generator=_gen(0))
    factors = out[:, 0, :]                                    # (500, 9)
    torch.testing.assert_close(out, factors[:, None, :].expand_as(out))  # constant over time
    assert abs(factors.mean().item() - 1.0) < 0.01
    assert abs(factors.std().item() - 0.1) < 0.01
    assert not torch.allclose(factors[0], factors[1])         # each window gets its own factors


def test_same_generator_seed_gives_identical_output() -> None:
    x = _window()
    a, b = (RandomAugment(p=1.0, generator=_gen(7))(x) for _ in range(2))
    assert torch.equal(a, b)
    outs = [RandomAugment(p=1.0, generator=_gen(s))(x) for s in (1, 2)]
    assert not torch.equal(outs[0], outs[1])


def test_probability_zero_is_identity_and_one_always_changes() -> None:
    x = _window()
    torch.testing.assert_close(RandomAugment(p=0.0, generator=_gen(0))(x), x, rtol=0, atol=0)
    assert not torch.equal(RandomAugment(p=1.0, generator=_gen(0))(x), x)


def test_each_transform_fires_about_half_the_time() -> None:
    aug = RandomAugment(generator=_gen(0))
    x = torch.ones(128, 9)
    changed = sum(not torch.equal(aug(x), x) for _ in range(2000))
    # P(at least one of two independent p=0.5 transforms) = 0.75
    assert 0.70 < changed / 2000 < 0.80


def test_input_is_not_modified() -> None:
    x = _window()
    before = x.clone()
    RandomAugment(p=1.0, generator=_gen(0))(x)
    assert torch.equal(x, before)


def test_invalid_probability_is_rejected() -> None:
    with pytest.raises(ValueError):
        RandomAugment(p=1.5)


def test_augmented_dataset_transforms_x_only_and_keeps_originals() -> None:
    x, y = torch.randn(10, 128, 9), torch.arange(10)
    ds = AugmentedTensorDataset(x, y, RandomAugment(p=1.0, generator=_gen(0)))
    xi, yi = ds[3]
    assert xi.shape == (128, 9) and yi.item() == 3
    assert not torch.equal(xi, x[3])
    assert torch.equal(ds.tensors[0], x) and len(ds) == 10
    assert RandomAugment().to_dict() == {"jitter_sigma": 0.05, "scale_sigma": 0.1, "p": 0.5}
