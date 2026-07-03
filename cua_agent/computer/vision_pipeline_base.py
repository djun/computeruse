"""Shared screen-capture and visual-grounding pipeline.

OS adapters subclass :class:`SharedVisionPipeline` and inject their own
``get_display_info`` callable. The only platform differences are the display
source and whether the OS cursor should be captured, so both are constructor
parameters; everything else (encoding, hashing, SSIM, OCR/blob/detector
fallback) is identical across platforms and lives here.
"""

from __future__ import annotations

import base64
import io
import time
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image, ImageChops, ImageDraw

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None

from cua_agent.computer.drivers import BaseVisionPipeline
from cua_agent.computer.types import DisplayInfo
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger
from cua_agent.utils.sensitive_redaction import redact_sensitive_regions


class SharedVisionPipeline(BaseVisionPipeline):
    """Captures the framebuffer, aligns to logical resolution, and encodes to base64."""

    def __init__(
        self,
        settings: Settings,
        *,
        get_display_info: Callable[[], DisplayInfo],
        with_cursor: bool = True,
    ) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.display: DisplayInfo = get_display_info()
        self._mss_with_cursor = with_cursor
        self.mss = self._build_mss()
        self._ssim_warned = False
        self._detector_backend_warned: set[str] = set()
        self._ultralytics_model: Any | None = None

    def _build_mss(self) -> Optional[object]:
        try:
            import mss  # type: ignore

            # Avoid cursor artifacts in change detection when the platform supports it.
            if self._mss_with_cursor:
                return mss.mss()
            return mss.mss(with_cursor=False)
        except Exception as exc:
            self.logger.warning("mss not available; using placeholder images: %s", exc)
            return None

    def capture_base64(self) -> str:
        image = self._grab_frame()
        image = self._redact_sensitive_content(image)
        return self._encode_image(image)

    def capture_with_hash(self) -> tuple[str, str]:
        """Capture the screen and return (base64, perceptual hash)."""
        image = self._grab_frame()
        image = self._redact_sensitive_content(image)
        img_hash = self._average_hash(image)
        return self._encode_image(image), img_hash

    def capture_zoom(self, box: Any, *, upscale: float = 2.0) -> str:
        """Capture the screen, crop to `box`, and return the region upscaled for detail."""
        image = self._grab_frame()
        image = self._redact_sensitive_content(image)
        return self._encode_image(self._crop_and_scale(image, box, upscale))

    def crop_region(self, image_b64: str, box: Any, *, upscale: float = 2.0) -> str:
        """Crop an existing base64 frame to `box` and return the region upscaled."""
        image = self._decode(image_b64)
        return self._encode_image(self._crop_and_scale(image, box, upscale))

    def _crop_and_scale(self, image: Image.Image, box: Any, upscale: float) -> Image.Image:
        left, top, right, bottom = self._normalize_box(box, image.width, image.height)
        cropped = image.crop((left, top, right, bottom))
        factor = max(1.0, float(upscale or 1.0))
        if factor > 1.0 and cropped.width > 0 and cropped.height > 0:
            resample = getattr(Image, "Resampling", None)
            resample_filter = resample.LANCZOS if resample else Image.LANCZOS
            cropped = cropped.resize(
                (max(1, int(cropped.width * factor)), max(1, int(cropped.height * factor))),
                resample_filter,
            )
        return cropped

    @staticmethod
    def _normalize_box(box: Any, width: int, height: int) -> tuple[int, int, int, int]:
        """Coerce a [x1,y1,x2,y2] or {x,y,w,h} box into clamped pixel bounds."""
        if isinstance(box, dict):
            x1 = float(box.get("x", 0))
            y1 = float(box.get("y", 0))
            x2 = x1 + float(box.get("w", 0))
            y2 = y1 + float(box.get("h", 0))
        else:
            seq = list(box or [])
            if len(seq) < 4:
                return 0, 0, width, height
            x1, y1, x2, y2 = (float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]))
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        left = max(0, min(int(round(left)), width))
        top = max(0, min(int(round(top)), height))
        right = max(0, min(int(round(right)), width))
        bottom = max(0, min(int(round(bottom)), height))
        if right - left < 1:
            right = min(width, left + 1)
        if bottom - top < 1:
            bottom = min(height, top + 1)
        return left, top, right, bottom

    def _redact_sensitive_content(self, image: Image.Image) -> Image.Image:
        if not self.settings.enable_sensitive_vision_redaction:
            return image
        try:
            redacted, count = redact_sensitive_regions(
                image,
                min_confidence=float(self.settings.vision_redaction_min_ocr_conf),
                blur_padding_px=int(self.settings.vision_redaction_blur_padding_px),
            )
            if count > 0:
                self.logger.debug("Applied sensitive redaction to %d OCR region(s)", count)
            return redacted
        except Exception as exc:
            self.logger.debug("Sensitive redaction failed: %s", exc)
            return image

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
        """Compute a lightweight visual hash (average hash) from a base64 image."""
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
            score = float(skimage_ssim(prev_arr, curr_arr, data_range=255.0))
            return score
        except Exception as exc:
            self.logger.debug("SSIM computation failed: %s", exc)
            return None

    def detect_ui_elements(self, image_b64: str) -> list[dict]:
        """
        Visual grounding fallback.
        Detection order: optional object detector -> OCR text boxes -> vision blobs.
        Returns AX-like nodes compatible with the core tree contract.
        """
        try:
            img = self._decode(image_b64)
        except Exception as exc:
            self.logger.warning("Vision blob/OCR decode failed: %s", exc)
            return []

        candidates: list[dict] = []
        candidates.extend(self._detect_with_optional_detector(img))
        candidates.extend(self._detect_ocr_elements(img))
        candidates.extend(self._detect_blob_elements(img))
        return self._normalize_elements(candidates, img.width, img.height)

    def _detect_with_optional_detector(self, image: Image.Image) -> list[dict]:
        if not self.settings.enable_visual_detector:
            return []

        backend = (self.settings.visual_detector_backend or "auto").strip().lower()
        if backend in {"none", "off", "disabled"}:
            return []

        if backend in {"auto", "ultralytics"}:
            return self._detect_with_ultralytics(image, required=(backend == "ultralytics"))
        if backend == "groundingdino":
            key = "groundingdino"
            if key not in self._detector_backend_warned:
                self._detector_backend_warned.add(key)
                self.logger.warning(
                    "VISUAL_DETECTOR_BACKEND=groundingdino is not wired in this build; "
                    "falling back to OCR/blob visual grounding."
                )
            return []

        key = f"unknown:{backend}"
        if key not in self._detector_backend_warned:
            self._detector_backend_warned.add(key)
            self.logger.warning("Unknown visual detector backend %r; using OCR/blob fallback only.", backend)
        return []

    def _detect_with_ultralytics(self, image: Image.Image, *, required: bool) -> list[dict]:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            if required:
                self.logger.warning("Ultralytics backend requested but unavailable: %s", exc)
            return []

        model_name = (self.settings.visual_detector_model or "").strip() or "yolov8n.pt"
        try:
            if self._ultralytics_model is None or getattr(self._ultralytics_model, "_model_name", None) != model_name:
                model = YOLO(model_name)
                setattr(model, "_model_name", model_name)
                self._ultralytics_model = model
        except Exception as exc:
            if required:
                self.logger.warning("Failed to load ultralytics model %r: %s", model_name, exc)
            else:
                self.logger.debug("Ultralytics model %r unavailable: %s", model_name, exc)
            return []

        try:
            np_img = np.array(image)
            predictions = self._ultralytics_model.predict(
                source=np_img,
                conf=float(self.settings.visual_detector_confidence),
                iou=float(self.settings.visual_detector_iou),
                max_det=int(self.settings.visual_detector_max_detections),
                verbose=False,
            )
        except Exception as exc:
            if required:
                self.logger.warning("Ultralytics inference failed: %s", exc)
            else:
                self.logger.debug("Ultralytics inference failed: %s", exc)
            return []

        out: list[dict] = []
        if not predictions:
            return out

        result = predictions[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return out

        names = getattr(result, "names", {}) or {}
        for idx in range(len(boxes)):
            try:
                xyxy = boxes.xyxy[idx].tolist()
                conf = float(boxes.conf[idx].item()) if hasattr(boxes, "conf") else None
                cls_id = int(boxes.cls[idx].item()) if hasattr(boxes, "cls") else -1
                x0, y0, x1, y1 = [float(v) for v in xyxy]
                w = x1 - x0
                h = y1 - y0
                if w <= 1 or h <= 1:
                    continue
                class_name = str(names.get(cls_id, "detected_object"))
                role = "AXUnknown"
                class_l = class_name.lower()
                if "button" in class_l:
                    role = "AXButton"
                elif "text" in class_l or "input" in class_l or "field" in class_l:
                    role = "AXTextField"
                elif "link" in class_l:
                    role = "AXLink"
                out.append(
                    {
                        "role": role,
                        "title": class_name,
                        "label": class_name,
                        "frame": {"x": x0, "y": y0, "w": w, "h": h},
                        "source": "detector_ultralytics",
                        "confidence": conf,
                        "path": f"vision.detector.ultralytics.{idx + 1}",
                    }
                )
            except Exception:
                continue
        return out

    def _detect_ocr_elements(self, image: Image.Image) -> list[dict]:
        out: list[dict] = []
        try:
            import pytesseract  # type: ignore

            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            n_boxes = len(ocr_data.get("text", []))
            for i in range(n_boxes):
                text = (ocr_data["text"][i] or "").strip()
                conf = float(ocr_data.get("conf", [0])[i] or 0)
                if not text or conf < 55:
                    continue
                x, y, w, h = (
                    int(ocr_data["left"][i]),
                    int(ocr_data["top"][i]),
                    int(ocr_data["width"][i]),
                    int(ocr_data["height"][i]),
                )
                if w <= 1 or h <= 1:
                    continue
                out.append(
                    {
                        "role": "AXStaticText",
                        "title": text,
                        "label": text,
                        "frame": {"x": x, "y": y, "w": w, "h": h},
                        "source": "ocr",
                        "confidence": conf / 100.0,
                    }
                )
        except ImportError:
            pass
        except Exception as exc:
            self.logger.debug("OCR detection failed: %s", exc)
        return out

    def _detect_blob_elements(self, image: Image.Image) -> list[dict]:
        out: list[dict] = []
        try:
            if skimage_ssim is None:
                return out
            from skimage import filters, measure, morphology

            gray = np.array(image.convert("L"))
            edges = filters.sobel(gray)
            mask = edges > 0.04
            closed = morphology.closing(mask, morphology.footprint_rectangle((3, 3)))
            labels = measure.label(closed)
            props = measure.regionprops(labels)

            min_area = 120
            max_area = (image.width * image.height) / 3

            for prop in props:
                if prop.area < min_area or prop.area > max_area:
                    continue
                minr, minc, maxr, maxc = prop.bbox
                out.append(
                    {
                        "role": "AXUnknown",
                        "title": "visual_element",
                        "label": "visual_element",
                        "frame": {"x": minc, "y": minr, "w": maxc - minc, "h": maxr - minr},
                        "source": "vision_blob",
                    }
                )
        except Exception as exc:
            self.logger.debug("Vision blob detection failed: %s", exc)
        return out

    def _normalize_elements(self, elements: list[dict], width: int, height: int) -> list[dict]:
        cleaned: list[dict] = []
        for idx, elem in enumerate(elements, start=1):
            frame = self._coerce_frame(elem.get("frame"))
            if not frame:
                continue
            frame = self._clip_frame(frame, width, height)
            if not frame:
                continue

            role = (elem.get("role") or "").strip() or "AXUnknown"
            title = str(elem.get("title") or "").strip()
            label = str(elem.get("label") or title or role).strip()
            source = str(elem.get("source") or "vision").strip() or "vision"
            path = str(elem.get("path") or f"vision.{source}.{idx}").strip()
            confidence = elem.get("confidence")
            try:
                confidence_val = float(confidence) if confidence is not None else None
            except Exception:
                confidence_val = None

            node = {
                "role": role,
                "title": title or label,
                "label": label,
                "frame": frame,
                "source": source,
                "path": path,
            }
            if confidence_val is not None:
                node["confidence"] = max(0.0, min(confidence_val, 1.0))
            cleaned.append(node)

        if not cleaned:
            return []

        def _priority(node: dict) -> float:
            frame = node.get("frame") or {}
            area = float(frame.get("w", 0)) * float(frame.get("h", 0))
            src = (node.get("source") or "").lower()
            conf = float(node.get("confidence", 0.0) or 0.0)
            if "detector" in src:
                src_bias = 3.0
            elif src == "ocr":
                src_bias = 2.0
            else:
                src_bias = 1.0
            return src_bias + conf + min(area / 15000.0, 2.0)

        cleaned.sort(key=_priority, reverse=True)

        deduped: list[dict] = []
        for node in cleaned:
            frame = node.get("frame") or {}
            duplicate = False
            for kept in deduped:
                if self._frame_iou(frame, kept.get("frame") or {}) >= 0.85:
                    duplicate = True
                    break
            if not duplicate:
                deduped.append(node)
            if len(deduped) >= 120:
                break

        return deduped

    def _coerce_frame(self, frame: Any) -> dict | None:
        if not isinstance(frame, dict):
            return None
        try:
            x = float(frame.get("x", 0))
            y = float(frame.get("y", 0))
            w = float(frame.get("w", 0))
            h = float(frame.get("h", 0))
        except Exception:
            return None
        return {"x": x, "y": y, "w": w, "h": h}

    def _clip_frame(self, frame: dict, width: int, height: int) -> dict | None:
        x0 = max(0.0, min(float(width), frame["x"]))
        y0 = max(0.0, min(float(height), frame["y"]))
        x1 = max(0.0, min(float(width), frame["x"] + frame["w"]))
        y1 = max(0.0, min(float(height), frame["y"] + frame["h"]))
        w = x1 - x0
        h = y1 - y0
        if w <= 1.0 or h <= 1.0:
            return None
        return {"x": x0, "y": y0, "w": w, "h": h}

    def _frame_iou(self, a: dict, b: dict) -> float:
        ax0, ay0 = float(a.get("x", 0)), float(a.get("y", 0))
        ax1, ay1 = ax0 + float(a.get("w", 0)), ay0 + float(a.get("h", 0))
        bx0, by0 = float(b.get("x", 0)), float(b.get("y", 0))
        bx1, by1 = bx0 + float(b.get("w", 0)), by0 + float(b.get("h", 0))

        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)
        inter_w = max(0.0, inter_x1 - inter_x0)
        inter_h = max(0.0, inter_y1 - inter_y0)
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0

        area_a = max(0.0, (ax1 - ax0) * (ay1 - ay0))
        area_b = max(0.0, (bx1 - bx0) * (by1 - by0))
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _decode(self, image_b64: str) -> Image.Image:
        raw = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def _average_hash(self, image: Image.Image, hash_size: int = 8) -> str:
        """Lightweight visual hash (aHash) to detect subtle stagnation/loops."""
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
