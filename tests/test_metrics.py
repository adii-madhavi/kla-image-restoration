"""Metric sanity tests: identical-image case and perturbed-image case
(Architecture.md §12: "Metric sanity must include an identical-image
case and a perturbed-image case.")."""
from __future__ import annotations

import numpy as np

from src.metrics.psnr import psnr
from src.metrics.ssim import ssim


def test_psnr_identical_is_infinite():
    img = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    assert psnr(img, img) == float("inf")


def test_psnr_perturbed_is_finite_and_bounded():
    rng = np.random.default_rng(1)
    img = rng.random((32, 32)).astype(np.float32)
    noisy = np.clip(img + rng.normal(0, 0.05, img.shape), 0, 1).astype(np.float32)
    value = psnr(img, noisy)
    assert np.isfinite(value)
    assert value > 0


def test_psnr_decreases_with_more_noise():
    rng = np.random.default_rng(2)
    img = rng.random((32, 32)).astype(np.float32)
    mild = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1).astype(np.float32)
    severe = np.clip(img + rng.normal(0, 0.2, img.shape), 0, 1).astype(np.float32)
    assert psnr(img, mild) > psnr(img, severe)


def test_ssim_identical_is_one():
    img = np.random.default_rng(3).random((32, 32)).astype(np.float32)
    assert ssim(img, img) > 0.999


def test_ssim_perturbed_is_less_than_one():
    rng = np.random.default_rng(4)
    img = rng.random((32, 32)).astype(np.float32)
    noisy = np.clip(img + rng.normal(0, 0.1, img.shape), 0, 1).astype(np.float32)
    value = ssim(img, noisy)
    assert value < 1.0
    assert value > -1.0
