import pytest
import hypothesis
import torch
import numpy as np
import cv2
import os
import sys

# Ensure demo module is importable from workspace root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo import load_img, load_gray_img, save_img, save_gray_img


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_color_image(path, h=32, w=48):
    """Write a random uint8 BGR image to *path* using OpenCV."""
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return h, w


def _write_gray_image(path, h=32, w=48):
    """Write a random uint8 grayscale image to *path* using OpenCV."""
    img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return h, w


# ---------------------------------------------------------------------------
# Unit tests — load_img
# ---------------------------------------------------------------------------

class TestLoadImg:
    def test_returns_ndarray(self, tmp_path):
        p = tmp_path / "color.png"
        _write_color_image(p)
        result = load_img(str(p))
        assert isinstance(result, np.ndarray)

    def test_shape_is_H_W_3(self, tmp_path):
        p = tmp_path / "color.png"
        h, w = _write_color_image(p)
        result = load_img(str(p))
        assert result.shape == (h, w, 3), f"Expected ({h}, {w}, 3), got {result.shape}"

    def test_dtype_is_uint8(self, tmp_path):
        p = tmp_path / "color.png"
        _write_color_image(p)
        result = load_img(str(p))
        assert result.dtype == np.uint8

    def test_rgb_channel_order(self, tmp_path):
        """load_img should convert BGR→RGB so a pure-red pixel reads (255,0,0)."""
        p = tmp_path / "red.png"
        # Pure red in BGR is (0, 0, 255)
        img_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        img_bgr[:, :, 2] = 255  # red channel in BGR
        cv2.imwrite(str(p), img_bgr)
        result = load_img(str(p))
        # After BGR→RGB the first channel should be 255
        assert result[0, 0, 0] == 255
        assert result[0, 0, 1] == 0
        assert result[0, 0, 2] == 0

    def test_non_square_image(self, tmp_path):
        p = tmp_path / "wide.png"
        h, w = _write_color_image(p, h=16, w=64)
        result = load_img(str(p))
        assert result.shape == (h, w, 3)


# ---------------------------------------------------------------------------
# Unit tests — load_gray_img
# ---------------------------------------------------------------------------

class TestLoadGrayImg:
    def test_returns_ndarray(self, tmp_path):
        p = tmp_path / "gray.png"
        _write_gray_image(p)
        result = load_gray_img(str(p))
        assert isinstance(result, np.ndarray)

    def test_shape_is_H_W_1(self, tmp_path):
        p = tmp_path / "gray.png"
        h, w = _write_gray_image(p)
        result = load_gray_img(str(p))
        assert result.shape == (h, w, 1), f"Expected ({h}, {w}, 1), got {result.shape}"

    def test_dtype_is_uint8(self, tmp_path):
        p = tmp_path / "gray.png"
        _write_gray_image(p)
        result = load_gray_img(str(p))
        assert result.dtype == np.uint8

    def test_has_exactly_one_channel(self, tmp_path):
        p = tmp_path / "gray.png"
        _write_gray_image(p)
        result = load_gray_img(str(p))
        assert result.ndim == 3
        assert result.shape[2] == 1

    def test_non_square_image(self, tmp_path):
        p = tmp_path / "tall.png"
        h, w = _write_gray_image(p, h=64, w=16)
        result = load_gray_img(str(p))
        assert result.shape == (h, w, 1)


# ---------------------------------------------------------------------------
# Unit tests — save_img / save_gray_img round-trip
# ---------------------------------------------------------------------------

class TestSaveImg:
    def test_save_and_reload_color(self, tmp_path):
        src = tmp_path / "src.png"
        dst = tmp_path / "dst.png"
        h, w = _write_color_image(src)
        img = load_img(str(src))
        save_img(str(dst), img)
        reloaded = load_img(str(dst))
        assert reloaded.shape == (h, w, 3)
        assert reloaded.dtype == np.uint8
        np.testing.assert_array_equal(img, reloaded)

    def test_save_creates_file(self, tmp_path):
        src = tmp_path / "src.png"
        dst = tmp_path / "out.png"
        _write_color_image(src)
        img = load_img(str(src))
        save_img(str(dst), img)
        assert dst.exists()


class TestSaveGrayImg:
    def test_save_and_reload_gray(self, tmp_path):
        src = tmp_path / "src.png"
        dst = tmp_path / "dst.png"
        h, w = _write_gray_image(src)
        img = load_gray_img(str(src))
        save_gray_img(str(dst), img)
        reloaded = load_gray_img(str(dst))
        assert reloaded.shape == (h, w, 1)
        assert reloaded.dtype == np.uint8
        np.testing.assert_array_equal(img, reloaded)

    def test_save_creates_file(self, tmp_path):
        src = tmp_path / "src.png"
        dst = tmp_path / "out.png"
        _write_gray_image(src)
        img = load_gray_img(str(src))
        save_gray_img(str(dst), img)
        assert dst.exists()
