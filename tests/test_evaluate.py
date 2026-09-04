import numpy as np

from src.evaluate import psnr, ssim


def test_identical_images_are_perfect():
    img = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    assert psnr(img, img) == float("inf")
    assert abs(ssim(img, img) - 1.0) < 1e-6


def test_different_images_are_not_perfect():
    rng = np.random.default_rng(0)
    img1 = rng.random((32, 32)).astype(np.float32)
    img2 = rng.random((32, 32)).astype(np.float32)
    assert psnr(img1, img2) < 20  # two independent random images should score poorly
    assert ssim(img1, img2) < 0.5


def test_closer_image_scores_higher():
    rng = np.random.default_rng(0)
    gt = rng.random((32, 32)).astype(np.float32)
    close = np.clip(gt + rng.normal(0, 0.01, gt.shape), 0, 1).astype(np.float32)
    far = np.clip(gt + rng.normal(0, 0.5, gt.shape), 0, 1).astype(np.float32)
    assert psnr(close, gt) > psnr(far, gt)
    assert ssim(close, gt) > ssim(far, gt)
