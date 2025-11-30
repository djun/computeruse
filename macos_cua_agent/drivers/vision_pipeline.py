from __future__ import annotations

import base64
import io
import time
from typing import Optional

import numpy as np
from PIL import Image, ImageChops, ImageDraw

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None

from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger
from macos_cua_agent.utils.macos_integration import DisplayInfo, get_display_info


class VisionPipeline:
    """Captures the framebuffer, aligns to logical resolution, and encodes to base64."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.display: DisplayInfo = get_display_info()
        self.mss = self._build_mss()
        self._ssim_warned = False

    def _build_mss(self) -> Optional[object]:
        try:
            import mss  # type: ignore

            return mss.mss()
        except Exception:
            self.logger.warning("mss not available; using placeholder images.")
            return None

    def capture_base64(self) -> str:
        image = self._grab_frame()
        return self._encode_image(image)

    def capture_with_hash(self) -> tuple[str, str]:
        """Capture the screen and return (base64, perceptual hash)."""
        image = self._grab_frame()
        img_hash = self._average_hash(image)
        return self._encode_image(image), img_hash

    def _grab_frame(self) -> Image.Image:
        if self.mss:
            try:
                monitor = self.mss.monitors[1]  # Primary monitor only for V1
                raw = self.mss.grab(monitor)
                img = Image.frombytes("RGB", raw.size, raw.rgb)
                return self._to_logical_image(img)
            except Exception as exc:
                self.logger.warning("mss capture failed; falling back to placeholder: %s", exc)
        return self._placeholder_frame()

    def _placeholder_frame(self) -> Image.Image:
        width, height = self.display.logical_width, self.display.logical_height
        img = Image.new("RGB", (width, height), color=(32, 32, 32))
        draw = ImageDraw.Draw(img)
        text = f"Placeholder frame @ {time.strftime('%H:%M:%S')} ({width}x{height})"
        draw.text((20, 20), text, fill=(200, 200, 200))
        return img

    def _encode_image(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        fmt = "PNG" if self.settings.encode_format.upper() == "PNG" else "JPEG"
        image.save(buffer, format=fmt)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def hash_base64(self, image_b64: str) -> str:
        """Compute an average-hash from a base64-encoded image."""
        image = self._decode(image_b64)
        return self._average_hash(image)

    def hash_distance(self, hash_a: Optional[str], hash_b: Optional[str]) -> int:
        """Hamming distance between two hex hashes; returns max if invalid."""
        if not hash_a or not hash_b:
            return 64
        try:
            return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")
        except Exception:
            return 64

    def has_changed(self, previous_b64: str, current_b64: str, threshold: float = 0.01) -> bool:
        """Simple pixel delta check to decide if the UI changed."""
        try:
            prev_img = self._decode(previous_b64)
            curr_img = self._decode(current_b64)
        except Exception as exc:
            self.logger.warning("Failed to decode frames; assuming changed: %s", exc)
            return True

        diff = ImageChops.difference(prev_img, curr_img)
        histogram = diff.histogram()
        diff_score = sum(i * count for i, count in enumerate(histogram))
        max_score = 255 * sum(histogram)
        if max_score == 0:
            return False
        ratio = diff_score / max_score
        return ratio >= threshold

    def structural_similarity(self, previous_b64: str, current_b64: str) -> float | None:
        """
        True SSIM score in [0,1] (1 = identical). Falls back to None on failure.
        """
        if skimage_ssim is None:
            if not self._ssim_warned:
                self.logger.debug("skimage not available; SSIM will be skipped.")
                self._ssim_warned = True
            return None
        try:
            prev_img = self._decode(previous_b64).convert("L")
            curr_img = self._decode(current_b64).convert("L")
        except Exception as exc:
            self.logger.debug("SSIM decode failed: %s", exc)
            return None

        if prev_img.size != curr_img.size:
            try:
                curr_img = curr_img.resize(prev_img.size)
            except Exception:
                return None

        prev_arr = np.array(prev_img, dtype=np.float32)
        curr_arr = np.array(curr_img, dtype=np.float32)

        try:
            score, _ = skimage_ssim(prev_arr, curr_arr, full=True, data_range=255.0)
            return float(score)
        except Exception as exc:
            self.logger.debug("SSIM computation failed: %s", exc)
            return None

    def _decode(self, image_b64: str) -> Image.Image:
        raw = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def _average_hash(self, image: Image.Image, hash_size: int = 8) -> str:
        """Lightweight perceptual hash (aHash) to detect subtle stagnation/loops."""
        gray = image.convert("L")
        resample = getattr(Image, "Resampling", None)
        resample_filter = resample.LANCZOS if resample else Image.LANCZOS
        resized = gray.resize((hash_size, hash_size), resample_filter)
        pixels = list(resized.getdata())
        avg = sum(pixels) / len(pixels) if pixels else 0
        bits = "".join("1" if px > avg else "0" for px in pixels)
        return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"

    def _to_logical_image(self, image: Image.Image) -> Image.Image:
        """Downscale a physical capture to logical resolution to keep coordinates aligned."""
        target_w, target_h = self.display.logical_width, self.display.logical_height
        if image.width == target_w and image.height == target_h:
            return image
        resample = getattr(Image, "Resampling", None)
        resample_filter = resample.BICUBIC if resample else Image.BICUBIC
        try:
            return image.resize((target_w, target_h), resample_filter)
        except Exception as exc:
            self.logger.warning("Resize to logical failed (%s); returning original frame", exc)
            return image
