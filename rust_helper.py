"""
External Rust helper architecture example.

NOTE:
- This file is an architectural template intended for educational purposes.
- It does not include offsets, memory reads, or any game-internal access.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import List, Optional, Tuple

import numpy as np

# GUI / Overlay
import dearpygui.dearpygui as dpg

# Screen capture
import mss

# AI inference (CUDA via onnxruntime)
import onnxruntime as ort

# Windows input for smooth aiming
import ctypes

# --------------------------------------------------------------------------------------
# Data Models
# --------------------------------------------------------------------------------------


@dataclass
class Detection:
    """Represents a single detection from the model."""

    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls: int

    def center(self) -> Tuple[int, int]:
        """Return the center point of the bounding box."""
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2


@dataclass
class FrameResult:
    """Stores model detections and the capture timestamp."""

    timestamp: float
    detections: List[Detection] = field(default_factory=list)


@dataclass
class Settings:
    """Runtime settings controlled by the menu."""

    fov_radius: int = 250
    smoothness: float = 0.4
    enable_esp: bool = True


# --------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------


def _clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value into the provided range."""
    return max(min_value, min(value, max_value))


def _distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Compute Euclidean distance between 2D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# --------------------------------------------------------------------------------------
# Core Class
# --------------------------------------------------------------------------------------


class RustHelper:
    """Main class that coordinates capture, inference, and rendering."""

    def __init__(self) -> None:
        """Initialize capture, model session, state containers, and UI."""
        self.settings = Settings()
        self.running = threading.Event()
        self.running.set()

        # Shared data between threads
        self.frame_queue: Queue[FrameResult] = Queue(maxsize=2)
        self.latest_result: Optional[FrameResult] = None

        # Thread handles
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.gui_thread = threading.Thread(target=self._gui_loop, daemon=True)

        # Model/session configuration (CUDA for minimal latency)
        self.session = self._create_ort_session("models/yolov8n.onnx")
        self.input_name = self.session.get_inputs()[0].name
        self.model_input_size = (640, 640)

        # Screen capture config
        self.capture = mss.mss()
        self.monitor = self._get_center_monitor_box(self.settings.fov_radius)

        # Window center cache
        self.screen_center = (self.monitor["width"] // 2, self.monitor["height"] // 2)

        # GUI state
        self.menu_visible = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start capture and GUI threads."""
        self.capture_thread.start()
        self.gui_thread.start()

    def stop(self) -> None:
        """Stop threads gracefully."""
        self.running.clear()

    # ------------------------------------------------------------------
    # Model / Inference
    # ------------------------------------------------------------------

    def _create_ort_session(self, model_path: str) -> ort.InferenceSession:
        """Create ONNX Runtime session with CUDA provider for low latency."""
        providers = [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
            "CPUExecutionProvider",
        ]
        return ort.InferenceSession(model_path, providers=providers)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalize frame to the model's input format."""
        resized = self._resize_letterbox(frame, self.model_input_size)
        image = resized.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, axis=0)
        return image

    def _postprocess(self, outputs: List[np.ndarray], conf_threshold: float = 0.35) -> List[Detection]:
        """Convert model outputs into Detection objects."""
        detections: List[Detection] = []
        output = outputs[0]
        for row in output:
            conf = float(row[4])
            if conf < conf_threshold:
                continue
            cls = int(np.argmax(row[5:]))
            if cls != 0:
                continue  # class 0 == person in COCO
            x_center, y_center, w, h = row[0], row[1], row[2], row[3]
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)
            detections.append(Detection(x1, y1, x2, y2, conf, cls))
        return detections

    def _resize_letterbox(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize with letterboxing to preserve aspect ratio."""
        import cv2

        target_w, target_h = size
        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        pad_w, pad_h = (target_w - new_w) // 2, (target_h - new_h) // 2
        canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
        return canvas

    # ------------------------------------------------------------------
    # Capture / Processing Loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Capture frames, run inference, and publish results at high FPS."""
        target_frame_time = 1.0 / 60.0
        while self.running.is_set():
            start_time = time.perf_counter()
            frame = self._grab_frame()
            if frame is None:
                continue

            input_tensor = self._preprocess(frame)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            detections = self._postprocess(outputs)

            result = FrameResult(timestamp=time.time(), detections=detections)
            self._publish_result(result)

            elapsed = time.perf_counter() - start_time
            remaining = target_frame_time - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _grab_frame(self) -> Optional[np.ndarray]:
        """Capture the FOV region in the center of the screen."""
        monitor = self._get_center_monitor_box(self.settings.fov_radius)
        self.monitor = monitor
        self.screen_center = (monitor["width"] // 2, monitor["height"] // 2)
        frame = np.array(self.capture.grab(monitor))[:, :, :3]
        return frame

    def _publish_result(self, result: FrameResult) -> None:
        """Publish the latest inference result to the shared queue."""
        try:
            self.frame_queue.put_nowait(result)
        except Exception:
            try:
                _ = self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(result)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # GUI / Overlay Loop
    # ------------------------------------------------------------------

    def _gui_loop(self) -> None:
        """Create the menu and overlay, and draw ESP boxes."""
        dpg.create_context()
        dpg.create_viewport(title="Rust Helper", width=400, height=300)
        self._build_menu()
        dpg.setup_dearpygui()
        dpg.show_viewport()

        while dpg.is_dearpygui_running() and self.running.is_set():
            self._poll_results()
            self._render_overlay()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    def _build_menu(self) -> None:
        """Build the menu controls for FOV, smoothness, and ESP."""
        with dpg.window(label="Rust Helper", width=400, height=300):
            dpg.add_slider_int(
                label="FOV Size",
                default_value=self.settings.fov_radius,
                min_value=100,
                max_value=600,
                callback=self._on_fov_change,
            )
            dpg.add_slider_float(
                label="Smoothness",
                default_value=self.settings.smoothness,
                min_value=0.1,
                max_value=1.0,
                callback=self._on_smoothness_change,
            )
            dpg.add_checkbox(
                label="Enable ESP",
                default_value=self.settings.enable_esp,
                callback=self._on_esp_toggle,
            )

    def _poll_results(self) -> None:
        """Read inference results from the queue for rendering and aiming."""
        try:
            while True:
                self.latest_result = self.frame_queue.get_nowait()
        except Empty:
            pass

    def _render_overlay(self) -> None:
        """Render bounding boxes using an overlay drawlist."""
        if not self.settings.enable_esp or not self.latest_result:
            return

        draw_list = dpg.get_background_drawlist()
        dpg.delete_item(draw_list, children_only=True)

        for det in self.latest_result.detections:
            dpg.draw_rectangle(
                (det.x1, det.y1),
                (det.x2, det.y2),
                color=(0, 255, 0, 200),
                thickness=1,
                parent=draw_list,
            )

    # ------------------------------------------------------------------
    # Aim Logic
    # ------------------------------------------------------------------

    def aim_at_closest(self) -> None:
        """Compute the aim vector to the closest target and move the mouse smoothly."""
        if not self.latest_result:
            return

        center = self.screen_center
        candidates = [det for det in self.latest_result.detections]
        if not candidates:
            return

        closest = min(candidates, key=lambda det: _distance(center, det.center()))
        dx = closest.center()[0] - center[0]
        dy = closest.center()[1] - center[1]

        self._smooth_move(dx, dy, self.settings.smoothness)

    def _smooth_move(self, dx: int, dy: int, smoothing: float) -> None:
        """Move the mouse using a smoothing coefficient to reduce jitter."""
        smoothing = _clamp(smoothing, 0.1, 1.0)
        steps = max(1, int(1.0 / smoothing))
        step_x = dx / steps
        step_y = dy / steps

        for _ in range(steps):
            self._mouse_move(int(step_x), int(step_y))
            time.sleep(0.001)

    def _mouse_move(self, dx: int, dy: int) -> None:
        """Low-level mouse movement through Win32 API."""
        ctypes.windll.user32.mouse_event(0x0001, dx, dy, 0, 0)

    # ------------------------------------------------------------------
    # UI Callbacks
    # ------------------------------------------------------------------

    def _on_fov_change(self, sender: int, app_data: int) -> None:
        """Update FOV radius from slider."""
        self.settings.fov_radius = int(app_data)

    def _on_smoothness_change(self, sender: int, app_data: float) -> None:
        """Update smoothness from slider."""
        self.settings.smoothness = float(app_data)

    def _on_esp_toggle(self, sender: int, app_data: bool) -> None:
        """Enable or disable ESP drawing."""
        self.settings.enable_esp = bool(app_data)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_center_monitor_box(self, radius: int) -> dict:
        """Return a bounding box centered on screen with given radius."""
        primary = self.capture.monitors[1]
        center_x = primary["left"] + primary["width"] // 2
        center_y = primary["top"] + primary["height"] // 2
        return {
            "left": center_x - radius,
            "top": center_y - radius,
            "width": radius * 2,
            "height": radius * 2,
        }


def main() -> None:
    """Entry point to start the helper."""
    helper = RustHelper()
    helper.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        helper.stop()


if __name__ == "__main__":
    main()
