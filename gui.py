"""
PyQt6 GUI launcher for the Fitness Assistant pose analyser.

Reuses all detection / drawing / voice logic from fitness_assisst.py.
Replaces every OpenCV window with native Qt widgets.

Usage:
    python gui.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from PyQt6.QtCore import (
    QThread,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QImage, QPixmap, QAction, QShortcut, QKeySequence, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# ── Import the entire pose engine (functions, CFG, classes) ──────────────
import fitness_assisst as engine
from ai_summarizer import summarize_squat_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_mp_pose = mp.solutions.pose
_mp_drawing = mp.solutions.drawing_utils


def _cv_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
    """Convert BGR cv2 image → QPixmap."""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Processing thread  (replaces the old while-cap.isOpened loop)
# ---------------------------------------------------------------------------

class PoseWorker(QThread):
    """Runs pose detection in a background thread, emits processed frames."""

    frame_ready = pyqtSignal(np.ndarray, np.ndarray)  # (skeleton_frame, coach_panel)
    status_msg  = pyqtSignal(str)
    summary_ready = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._source: int | str = 0            # 0 = webcam, str = file path
        self._model_complexity: int = 1
        self._rep_enabled: bool = False
        self._max_reps: int = 10
        self._mirror: bool = True
        self._show_rep_overlay: bool = True
        self._ai_summary_enabled: bool = False
        self._summary_thread: threading.Thread | None = None

    # -- configuration (set before start or between restarts) ---------------
    def configure(
        self,
        source: int | str,
        model_complexity: int,
        rep_enabled: bool,
        max_reps: int,
        mirror: bool,
        show_rep_overlay: bool = True,
        ai_summary_enabled: bool = False,
    ) -> None:
        self._source = source
        self._model_complexity = model_complexity
        self._rep_enabled = rep_enabled
        self._max_reps = max_reps
        self._mirror = mirror
        self._show_rep_overlay = show_rep_overlay
        self._ai_summary_enabled = ai_summary_enabled

    def request_stop(self) -> None:
        self._running = False

    def set_show_rep_overlay(self, show: bool) -> None:
        self._show_rep_overlay = show

    def _run_ai_summary_async(self, payload: dict) -> None:
        """Run Gemini summary in a separate thread so pose loop never blocks."""
        def _job() -> None:
            summary = summarize_squat_metrics(payload)
            self.summary_ready.emit(summary)

        t = threading.Thread(target=_job, daemon=True)
        self._summary_thread = t
        t.start()

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:  # noqa: C901 (complex but mirrors original loop)
        self._running = True
        engine.MIRROR_VIEW = self._mirror
        engine.CFG.squat_max_reps = self._max_reps if self._rep_enabled else 0

        pose = _mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self._model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            self.status_msg.emit(f"Cannot open source: {self._source}")
            return

        # Recording setup
        script_dir = os.path.dirname(os.path.abspath(engine.__file__))
        rec_dir = os.path.join(script_dir, "recordings")
        os.makedirs(rec_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        skel_path = os.path.join(rec_dir, f"pose_session_{stamp}_skeleton.mp4")
        raw_path  = os.path.join(rec_dir, f"pose_session_{stamp}_raw.mp4")
        rec_skel: cv2.VideoWriter | None = None
        rec_raw:  cv2.VideoWriter | None = None
        rec_done = False

        coach = engine.TrainerCoach(cooldown_sec=3.0)
        voice = engine.VoiceCoach(cooldown_sec=5.0) if (engine._TTS_AVAILABLE and engine._WINMM_OK) else None
        tracker = engine.ViolationTracker()
        squat_tracker = engine.SquatRepTracker()
        _squat_too_deep_coach = engine.TrainerCoach(cooldown_sec=4.0)
        _squat_partial_coach  = engine.TrainerCoach(cooldown_sec=4.0)

        coach_panel = np.zeros((620, 980, 3), dtype=np.uint8)

        is_file = isinstance(self._source, str)
        prev_rep_count = 0
        rep_milestone_state: dict = {}
        summary_sent = False

        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    break
                time.sleep(0.03)
                continue

            if engine.MIRROR_VIEW:
                frame = cv2.flip(frame, 1)
            raw_frame = frame.copy()
            trainer_msg: str | None = None

            # Recording init (first frame)
            if not rec_done:
                rec_done = True
                fh, fw = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 1 or fps > 120:
                    fps = 30.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                ws = cv2.VideoWriter(skel_path, fourcc, fps, (fw, fh))
                wr = cv2.VideoWriter(raw_path, fourcc, fps, (fw, fh))
                if not ws.isOpened() or not wr.isOpened():
                    ws.release(); wr.release()
                    skel_path = skel_path.replace(".mp4", ".avi")
                    raw_path  = raw_path.replace(".mp4", ".avi")
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    ws = cv2.VideoWriter(skel_path, fourcc, fps, (fw, fh))
                    wr = cv2.VideoWriter(raw_path, fourcc, fps, (fw, fh))
                if ws.isOpened() and wr.isOpened():
                    rec_skel, rec_raw = ws, wr
                else:
                    ws.release(); wr.release()

            # ----- pose processing -----
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            status = "No person detected"
            color = (0, 0, 255)
            sf_label = fi_label = kn_label = hip_label = torso_label = shlvl_label = "--"
            sf_color = fi_color = kn_color = hip_color = torso_color = shlvl_color = (140, 140, 140)
            left_ratio = right_ratio = fi_left_ratio = fi_right_ratio = 0.0
            kn_left_ratio = kn_right_ratio = 0.0
            hip_dx_ratio = torso_dx_ratio = torso_v_gap_ratio = shlvl_dy_ratio = 0.0

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                sf_label, sf_color, left_ratio, right_ratio, sf_cues = engine.check_shoulder_foot_vertical(lm)
                fi_label, fi_color, fi_left_ratio, fi_right_ratio, fi_cues = engine.check_shoulder_foot_index_vertical(lm)
                kn_label, kn_color, kn_left_ratio, kn_right_ratio, kn_cues = engine.check_shoulder_knee_vertical(lm)
                hip_label, hip_color, hip_dx_ratio, hip_x, hip_y, knee_x, knee_y, shoulder_w_n, hip_cues = (
                    engine.check_hip_center_vertical(lm)
                )
                torso_label, torso_color, torso_dx_ratio, torso_v_gap_ratio, sh_x, sh_y, thip_x, thip_y, torso_sh_w, torso_cues = (
                    engine.check_torso_front_vertical(lm)
                )
                shlvl_label, shlvl_color, shlvl_dy_ratio, lsx, lsy, rsx, rsy, shlvl_w, shlvl_cues = engine.check_shoulder_level(lm)

                all_cues = sf_cues + fi_cues + kn_cues + hip_cues + torso_cues + shlvl_cues
                tracker.update(set(all_cues))
                engine._sustained_cues = tracker.get_sustained(engine.SUSTAIN_SEC)

                skel_ok = engine.skeleton_pose_all_pass(
                    left_ratio, right_ratio,
                    engine.CFG.shoulder_foot_align_ratio_max,
                    engine.CFG.sf_left_inner_offset_ratio, engine.CFG.sf_left_outer_offset_ratio,
                    engine.CFG.sf_right_inner_offset_ratio, engine.CFG.sf_right_outer_offset_ratio,
                ) and engine.skeleton_pose_all_pass(
                    fi_left_ratio, fi_right_ratio,
                    engine.CFG.foot_index_align_ratio_max,
                    engine.CFG.fi_left_inner_offset_ratio, engine.CFG.fi_left_outer_offset_ratio,
                    engine.CFG.fi_right_inner_offset_ratio, engine.CFG.fi_right_outer_offset_ratio,
                ) and engine.skeleton_pose_all_pass(
                    kn_left_ratio, kn_right_ratio,
                    engine.CFG.knee_align_ratio_max,
                    engine.CFG.kn_left_inner_offset_ratio, engine.CFG.kn_left_outer_offset_ratio,
                    engine.CFG.kn_right_inner_offset_ratio, engine.CFG.kn_right_outer_offset_ratio,
                ) and (hip_dx_ratio <= (engine.CFG.hip_align_ratio_max + engine.CFG.hip_offset_ratio)) and (
                    torso_dx_ratio <= (engine.CFG.torso_horizontal_align_ratio_max + engine.CFG.torso_horizontal_offset_ratio)
                ) and (
                    torso_v_gap_ratio >= max(0.01, min(0.995,
                        engine.CFG.torso_vertical_gap_min_ratio - engine.CFG.torso_vertical_gap_offset_ratio))
                ) and (
                    shlvl_dy_ratio <= (engine.CFG.shoulder_level_align_ratio_max + engine.CFG.shoulder_level_offset_ratio)
                )

                any_raw = not skel_ok
                any_sus = len(engine._sustained_cues) > 0
                if any_sus:
                    bone_bgr = (0, 140, 255)
                elif any_raw:
                    bone_bgr = (0, 200, 255)
                else:
                    bone_bgr = (0, 255, 0)

                skel_pt = _mp_drawing.DrawingSpec(color=bone_bgr, thickness=2, circle_radius=3)
                skel_ln = _mp_drawing.DrawingSpec(color=bone_bgr, thickness=2)
                _mp_drawing.draw_landmarks(frame, results.pose_landmarks, _mp_pose.POSE_CONNECTIONS,
                                           landmark_drawing_spec=skel_pt, connection_drawing_spec=skel_ln)

                if not any_sus and not any_raw:
                    status = "All checks OK"
                    color = (0, 255, 0)
                elif any_raw and not any_sus:
                    status = "Hold position..."
                    color = (0, 200, 255)
                else:
                    for lbl, lc in [(sf_label, sf_color), (fi_label, fi_color),
                                    (kn_label, kn_color), (hip_label, hip_color),
                                    (torso_label, torso_color), (shlvl_label, shlvl_color)]:
                        if lc == (0, 0, 255):
                            status = lbl; break
                    color = (0, 0, 255)

                engine.draw_shoulder_foot_guides(frame, lm)
                engine.draw_foot_index_guides(frame, lm)
                engine.draw_knee_guides(frame, lm)
                engine.draw_hip_center_guides(frame, hip_x, hip_y, knee_x, knee_y, shoulder_w_n, hip_dx_ratio)
                engine.draw_torso_front_guides(frame, sh_x, sh_y, thip_x, thip_y, torso_sh_w, torso_dx_ratio, torso_v_gap_ratio)
                engine.draw_shoulder_level_guides(frame, lsx, lsy, rsx, rsy, shlvl_w, shlvl_dy_ratio)

                if self._rep_enabled:
                    old_rep_count = squat_tracker.count
                    lh = lm[_mp_pose.PoseLandmark.LEFT_HIP.value]
                    rh = lm[_mp_pose.PoseLandmark.RIGHT_HIP.value]
                    lk = lm[_mp_pose.PoseLandmark.LEFT_KNEE.value]
                    rk = lm[_mp_pose.PoseLandmark.RIGHT_KNEE.value]
                    la = lm[_mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    ra = lm[_mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                    foot_y = (la.y + ra.y) * 0.5
                    squat_tracker.update(
                        hip_y,
                        knee_y,
                        foot_y=foot_y,
                        shoulder_w=shoulder_w_n,
                        left_gap=(lk.y - lh.y),
                        right_gap=(rk.y - rh.y),
                    )
                    squat_tracker.observe_rep_form_cues(all_cues)
                    engine.draw_squat_rep_overlay(
                        frame,
                        squat_tracker,
                        hip_x,
                        hip_y,
                        knee_x,
                        knee_y,
                        shoulder_w_n,
                        show_rep_overlay=self._show_rep_overlay,
                    )
                    if voice is not None:
                        voice.update_rep(squat_tracker.count, engine.CFG.squat_max_reps)
                    if squat_tracker.too_deep:
                        if _squat_too_deep_coach.get_cue("squat_too_deep") and voice is not None:
                            voice.speak("squat_too_deep")
                    if squat_tracker.partial_warn:
                        if _squat_partial_coach.get_cue("squat_go_deeper") and voice is not None:
                            voice.speak("squat_go_deeper")

                for cue_key in all_cues:
                    if cue_key in engine._sustained_cues:
                        trainer_msg = engine.draw_trainer_cue(frame, coach, voice, cue_key)
                        break

                if (
                    self._rep_enabled
                    and voice is not None
                    and engine.CFG.squat_max_reps > 0
                    and squat_tracker.count > old_rep_count
                ):
                    voice.speak_rep_progress(squat_tracker.count, engine.CFG.squat_max_reps)

                if self._rep_enabled and voice is not None and engine.CFG.squat_max_reps > 0:
                    engine.speak_squat_rep_milestones(
                        voice,
                        prev_rep_count,
                        squat_tracker.count,
                        engine.CFG.squat_max_reps,
                        rep_milestone_state,
                    )
                if self._rep_enabled:
                    prev_rep_count = squat_tracker.count
                    if (
                        self._ai_summary_enabled
                        and not summary_sent
                        and engine.CFG.squat_max_reps > 0
                        and squat_tracker.count >= engine.CFG.squat_max_reps
                    ):
                        summary_sent = True
                        payload = squat_tracker.llm_summary_payload()
                        self._run_ai_summary_async(payload)
            else:
                tracker.update(set())
                engine._sustained_cues = frozenset()

            # Recording
            if rec_raw is not None:
                rec_raw.write(raw_frame)
            if rec_skel is not None:
                rec_skel.write(frame)

            # Coach text panel
            engine.render_coach_text_panel(
                coach_panel, status, color, trainer_msg,
                engine.get_voice_status_text(voice), squat_tracker,
                sf_label, sf_color, left_ratio, right_ratio,
                fi_label, fi_color, fi_left_ratio, fi_right_ratio,
                kn_label, kn_color, kn_left_ratio, kn_right_ratio,
                hip_label, hip_color, hip_dx_ratio,
                torso_label, torso_color, torso_dx_ratio, torso_v_gap_ratio,
                shlvl_label, shlvl_color, shlvl_dy_ratio,
            )

            self.frame_ready.emit(frame, coach_panel.copy())

            # Throttle to ~30 fps if reading from a file too fast
            if is_file:
                time.sleep(0.025)

        # Cleanup
        if rec_skel is not None:
            rec_skel.release()
        if rec_raw is not None:
            rec_raw.release()
        cap.release()
        pose.close()
        if voice is not None:
            voice.shutdown()

        self.status_msg.emit("Stopped")


# ---------------------------------------------------------------------------
# Coach popup
# ---------------------------------------------------------------------------

class CoachPopup(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Comments")
        self.resize(900, 640)

        self._panel: np.ndarray | None = None
        self._summary_text: str | None = None
        self._label = QLabel("Comments will appear here during session")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._label.setStyleSheet("background: #111; color: #777; font-size: 16px;")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.addWidget(self._label)

    def set_panel_image(self, panel: np.ndarray) -> None:
        if self._summary_text:
            return
        self._panel = panel.copy()
        self._refresh_pixmap()

    def set_summary_text(self, summary: str) -> None:
        self._summary_text = summary
        self._label.setPixmap(QPixmap())
        self._label.setText(summary)
        self._label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._label.setWordWrap(True)
        self._label.setStyleSheet("background: #111; color: #ddd; font-size: 13px; padding: 8px;")

    def clear_summary_text(self) -> None:
        self._summary_text = None
        self._label.setText("Comments will appear here during session")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background: #111; color: #777; font-size: 16px;")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._panel is None:
            return
        lw = self._label.width()
        lh = self._label.height()
        if lw <= 10 or lh <= 10:
            return
        h, w = self._panel.shape[:2]
        scale = min(lw / w, lh / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        display = cv2.resize(self._panel, (nw, nh), interpolation=interp)
        self._label.setPixmap(_cv_to_qpixmap(display))


# ---------------------------------------------------------------------------
# Slider group builder (replaces every cv2 trackbar window)
# ---------------------------------------------------------------------------

def _make_slider(label: str, min_v: int, max_v: int, init: int, callback) -> QHBoxLayout:
    row = QHBoxLayout()
    lbl = QLabel(label)
    lbl.setFixedWidth(140)
    lbl.setStyleSheet("color: #ccc; font-size: 11px;")
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(min_v, max_v)
    slider.setValue(init)
    val_lbl = QLabel(str(init))
    val_lbl.setFixedWidth(40)
    val_lbl.setStyleSheet("color: #8cf; font-size: 11px;")
    slider.valueChanged.connect(lambda v: (callback(v), val_lbl.setText(str(v))))
    row.addWidget(lbl)
    row.addWidget(slider)
    row.addWidget(val_lbl)
    return row


def _build_ankle_group() -> QGroupBox:
    grp = QGroupBox("Ankle Tolerance - drag sliders")
    lay = QVBoxLayout()
    C = engine.CFG
    lay.addLayout(_make_slider("sf_align x1000", 20, 800, engine._clamp_int(C.shoulder_foot_align_ratio_max * 1000, 20, 800),
                               lambda v: setattr(C, "shoulder_foot_align_ratio_max", max(0.02, v / 1000.0))))
    lay.addLayout(_make_slider("sf_L_in x1000", 0, 400, engine._clamp_int(C.sf_left_inner_offset_ratio * 1000, 0, 400),
                               lambda v: setattr(C, "sf_left_inner_offset_ratio", v / 1000.0)))
    lay.addLayout(_make_slider("sf_L_out x1000", 0, 400, engine._clamp_int(C.sf_left_outer_offset_ratio * 1000, 0, 400),
                               lambda v: setattr(C, "sf_left_outer_offset_ratio", v / 1000.0)))
    lay.addLayout(_make_slider("sf_R_in x1000", 0, 400, engine._clamp_int(C.sf_right_inner_offset_ratio * 1000, 0, 400),
                               lambda v: setattr(C, "sf_right_inner_offset_ratio", v / 1000.0)))
    lay.addLayout(_make_slider("sf_R_out x1000", 0, 400, engine._clamp_int(C.sf_right_outer_offset_ratio * 1000, 0, 400),
                               lambda v: setattr(C, "sf_right_outer_offset_ratio", v / 1000.0)))
    lay.addLayout(_make_slider("sf_line_len x1000", 0, 120, engine._clamp_int(C.sf_line_half_len_ratio * 1000, 0, 120),
                               lambda v: setattr(C, "sf_line_half_len_ratio", v / 1000.0)))
    grp.setLayout(lay)
    return grp


def _build_foot_group() -> QGroupBox:
    grp = QGroupBox("Foot Tolerance - drag sliders")
    lay = QVBoxLayout()
    C = engine.CFG
    lay.addLayout(_make_slider("fi_align x1000", 20, 800, engine._clamp_int(C.foot_index_align_ratio_max * 1000, 20, 800),
                               lambda v: setattr(C, "foot_index_align_ratio_max", max(0.02, v / 1000.0))))
    lay.addLayout(_make_slider("fi_L_in x1000", 0, 400, engine._clamp_int(C.fi_left_inner_offset_ratio * 1000, 0, 400),
                               lambda v: setattr(C, "fi_left_inner_offset_ratio", v / 1000.0)))
    lay.addLayout(_make_slider("fi_L_out x1000", 0, 400, engine._clamp_int(C.fi_left_outer_offset_ratio * 1000, 0, 400),
                               lambda v: setattr(C, "fi_left_outer_offset_ratio", v / 1000.0)))
    lay.addLayout(_make_slider("fi_R_in x1000", 0, 400, engine._clamp_int(C.fi_right_inner_offset_ratio * 1000, 0, 400),
                               lambda v: setattr(C, "fi_right_inner_offset_ratio", v / 1000.0)))
    lay.addLayout(_make_slider("fi_R_out x1000", 0, 400, engine._clamp_int(C.fi_right_outer_offset_ratio * 1000, 0, 400),
                               lambda v: setattr(C, "fi_right_outer_offset_ratio", v / 1000.0)))
    lay.addLayout(_make_slider("fi_line_len x1000", 0, 120, engine._clamp_int(C.fi_line_half_len_ratio * 1000, 0, 120),
                               lambda v: setattr(C, "fi_line_half_len_ratio", v / 1000.0)))
    grp.setLayout(lay)
    return grp


def _build_knee_group() -> QGroupBox:
    grp = QGroupBox("Knee Tolerance - drag sliders")
    lay = QVBoxLayout()
    C = engine.CFG
    lay.addLayout(_make_slider("kn_align x1000", 20, 800, engine._clamp_int(C.knee_align_ratio_max * 1000, 20, 800),
                               lambda v: setattr(C, "knee_align_ratio_max", max(0.02, v / 1000.0))))
    lay.addLayout(_make_slider("kn_L_in x1000", 0, 400, engine._clamp_int(C.kn_left_inner_offset_ratio * 1000, 0, 400),
                               lambda v: setattr(C, "kn_left_inner_offset_ratio", v / 1000.0)))
    lay.addLayout(_make_slider("kn_L_out x1000", 0, 400, engine._clamp_int(C.kn_left_outer_offset_ratio * 1000, 0, 400),
                               lambda v: setattr(C, "kn_left_outer_offset_ratio", v / 1000.0)))
    lay.addLayout(_make_slider("kn_R_in x1000", 0, 400, engine._clamp_int(C.kn_right_inner_offset_ratio * 1000, 0, 400),
                               lambda v: setattr(C, "kn_right_inner_offset_ratio", v / 1000.0)))
    lay.addLayout(_make_slider("kn_R_out x1000", 0, 400, engine._clamp_int(C.kn_right_outer_offset_ratio * 1000, 0, 400),
                               lambda v: setattr(C, "kn_right_outer_offset_ratio", v / 1000.0)))
    lay.addLayout(_make_slider("kn_line_len x1000", 0, 120, engine._clamp_int(C.kn_line_half_len_ratio * 1000, 0, 120),
                               lambda v: setattr(C, "kn_line_half_len_ratio", v / 1000.0)))
    grp.setLayout(lay)
    return grp


def _build_hip_group() -> QGroupBox:
    grp = QGroupBox("Hip Tolerance - drag sliders")
    lay = QVBoxLayout()
    C = engine.CFG
    lay.addLayout(_make_slider("hip_align x1000", 10, 800, engine._clamp_int(C.hip_align_ratio_max * 1000, 10, 800),
                               lambda v: setattr(C, "hip_align_ratio_max", max(0.01, v / 1000.0))))
    lay.addLayout(_make_slider("hip_offset x1000", 0, 400, engine._clamp_int(C.hip_offset_ratio * 1000, 0, 400),
                               lambda v: setattr(C, "hip_offset_ratio", v / 1000.0)))
    grp.setLayout(lay)
    return grp


def _build_torso_group() -> QGroupBox:
    grp = QGroupBox("Torso Front Tolerance - drag sliders")
    lay = QVBoxLayout()
    C = engine.CFG
    lay.addLayout(_make_slider("torso_x_align x1000", 10, 800,
                               engine._clamp_int(C.torso_horizontal_align_ratio_max * 1000, 10, 800),
                               lambda v: setattr(C, "torso_horizontal_align_ratio_max", max(0.01, v / 1000.0))))
    lay.addLayout(_make_slider("torso_x_off x1000", 0, 400,
                               engine._clamp_int(C.torso_horizontal_offset_ratio * 1000, 0, 400),
                               lambda v: setattr(C, "torso_horizontal_offset_ratio", v / 1000.0)))
    lay.addLayout(_make_slider("torso_vmin x1000", 50, 995,
                               engine._clamp_int(C.torso_vertical_gap_min_ratio * 1000, 50, 995),
                               lambda v: setattr(C, "torso_vertical_gap_min_ratio", max(0.05, min(0.995, v / 1000.0)))))
    lay.addLayout(_make_slider("torso_voff x1000", 0, 500,
                               engine._clamp_int(C.torso_vertical_gap_offset_ratio * 1000, 0, 500),
                               lambda v: setattr(C, "torso_vertical_gap_offset_ratio", v / 1000.0)))
    lay.addLayout(_make_slider("sh_level x1000", 1, 600,
                               engine._clamp_int(C.shoulder_level_align_ratio_max * 1000, 1, 600),
                               lambda v: setattr(C, "shoulder_level_align_ratio_max", max(0.001, v / 1000.0))))
    lay.addLayout(_make_slider("sh_off x1000", 0, 300,
                               engine._clamp_int(C.shoulder_level_offset_ratio * 1000, 0, 300),
                               lambda v: setattr(C, "shoulder_level_offset_ratio", v / 1000.0)))
    grp.setLayout(lay)
    return grp


def _build_display_group() -> QGroupBox:
    grp = QGroupBox("Display Ratio - drag sliders")
    lay = QVBoxLayout()
    C = engine.CFG
    lay.addLayout(_make_slider("disp_w", 0, 360,
                               engine._clamp_int((C.display_max_width - 480) // 4, 0, 360),
                               lambda v: setattr(C, "display_max_width", 480 + v * 4)))
    lay.addLayout(_make_slider("disp_h", 0, 220,
                               engine._clamp_int((C.display_max_height - 240) // 4, 0, 220),
                               lambda v: setattr(C, "display_max_height", 240 + v * 4)))
    grp.setLayout(lay)
    return grp


def _build_rep_logic_group() -> QGroupBox:
    """Live tuning controls for squat counting hysteresis and anti-walk filters."""
    grp = QGroupBox("Rep Logic Thresholds")
    lay = QVBoxLayout()
    T = engine.SquatRepTracker

    lay.addLayout(_make_slider(
        "enter_frac x1000", 100, 980,
        engine._clamp_int(T.ENTER_FRAC * 1000, 100, 980),
        lambda v: setattr(T, "ENTER_FRAC", max(0.10, min(0.98, v / 1000.0))),
    ))
    lay.addLayout(_make_slider(
        "return_frac x1000", 700, 995,
        engine._clamp_int(T.RETURN_FRAC * 1000, 700, 995),
        lambda v: setattr(T, "RETURN_FRAC", max(T.ENTER_FRAC + 0.02, min(0.995, v / 1000.0))),
    ))
    lay.addLayout(_make_slider(
        "full_frac x1000", 50, 600,
        engine._clamp_int(T.FULL_FRAC * 1000, 50, 600),
        lambda v: setattr(T, "FULL_FRAC", max(0.05, min(0.60, v / 1000.0))),
    ))
    lay.addLayout(_make_slider(
        "min_down_ms", 50, 800,
        engine._clamp_int(T.MIN_DOWN_SEC * 1000, 50, 800),
        lambda v: setattr(T, "MIN_DOWN_SEC", max(0.05, min(0.80, v / 1000.0))),
    ))
    lay.addLayout(_make_slider(
        "walk_asym x1000", 100, 900,
        engine._clamp_int(T.MAX_LR_ASYM_FRAC * 1000, 100, 900),
        lambda v: setattr(T, "MAX_LR_ASYM_FRAC", max(0.10, min(0.90, v / 1000.0))),
    ))
    lay.addLayout(_make_slider(
        "too_deep_gap x1000", -200, 0,
        engine._clamp_int(T.TOO_DEEP_GAP * 1000, -200, 0),
        lambda v: setattr(T, "TOO_DEEP_GAP", max(-0.20, min(0.0, v / 1000.0))),
    ))

    grp.setLayout(lay)
    return grp


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

_DARK_STYLE = """
QMainWindow, QWidget { background: #1e1e1e; color: #ddd; }
QGroupBox { border: 1px solid #444; border-radius: 4px; margin-top: 8px;
            padding-top: 14px; font-weight: bold; color: #8cf; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLabel { color: #ccc; }
QPushButton { background: #333; border: 1px solid #555; border-radius: 4px;
              padding: 6px 14px; color: #eee; }
QPushButton:hover { background: #444; }
QPushButton:pressed { background: #555; }
QPushButton#startBtn { background: #1a6b3a; border-color: #2a9d4f; }
QPushButton#startBtn:hover { background: #228b47; }
QPushButton#stopBtn  { background: #8b2020; border-color: #c04040; }
QPushButton#stopBtn:hover  { background: #a03030; }
QComboBox, QSpinBox { background: #2a2a2a; border: 1px solid #555;
                      border-radius: 3px; padding: 3px 6px; color: #eee; }
QComboBox::drop-down { border: none; }
QSlider::groove:horizontal { height: 6px; background: #444; border-radius: 3px; }
QSlider::handle:horizontal { width: 14px; height: 14px; margin: -4px 0;
                             background: #6af; border-radius: 7px; }
QSlider::sub-page:horizontal { background: #4a90d9; border-radius: 3px; }
QCheckBox { color: #ccc; spacing: 6px; }
QCheckBox::indicator { width: 16px; height: 16px; }
QScrollArea { border: none; }
QSplitter::handle { background: #444; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fitness Assistant — Pose Analyser")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(_DARK_STYLE)

        self._worker: PoseWorker | None = None
        self._is_fullscreen = False
        self._show_rep_overlay = True
        self._latest_ai_summary: str = ""
        self._latest_ai_summary_path: str = ""
        self._voice_thread: threading.Thread | None = None
        self._ai_summary_check = QCheckBox("Enable AI summarizer (Gemini)")
        self._ai_summary_check.setChecked(False)
        self._coach_popup = CoachPopup(self)

        # ---- central splitter: [left panel | video] ----
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ===================== LEFT PANEL =====================
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(340)
        left_scroll.setMaximumWidth(440)
        left_widget = QWidget()
        left_lay = QVBoxLayout(left_widget)
        left_lay.setContentsMargins(8, 8, 8, 8)

        # ── Input source ──
        src_grp = QGroupBox("Input Source")
        src_lay = QVBoxLayout()
        self._src_combo = QComboBox()
        self._src_combo.addItem("Live Camera (0)", 0)
        self._src_combo.addItem("Live Camera (1)", 1)
        self._src_combo.addItem("Live Camera (2)", 2)
        self._src_combo.addItem("Browse Video File...", "file")
        # currentIndexChanged does not fire when the user picks "Browse..." again while
        # that row is already selected — use activated for the file dialog.
        self._src_combo.currentIndexChanged.connect(self._on_source_index_changed)
        self._src_combo.activated.connect(self._on_source_activated)
        src_lay.addWidget(self._src_combo)
        self._file_label = QLabel("No file selected")
        self._file_label.setWordWrap(True)
        self._file_label.setStyleSheet("color: #888; font-size: 11px;")
        src_lay.addWidget(self._file_label)
        self._video_path: str | None = None
        src_grp.setLayout(src_lay)
        left_lay.addWidget(src_grp)

        # ── Model complexity ──
        model_grp = QGroupBox("Model Complexity")
        model_lay = QVBoxLayout()
        self._model_combo = QComboBox()
        self._model_combo.addItem("0 — Lite  (fast, less stable)", 0)
        self._model_combo.addItem("1 — Full  (balanced)", 1)
        self._model_combo.addItem("2 — Heavy (smooth, most stable)", 2)
        self._model_combo.setCurrentIndex(1)
        model_lay.addWidget(self._model_combo)
        model_grp.setLayout(model_lay)
        left_lay.addWidget(model_grp)

        # ── Rep count ──
        rep_grp = QGroupBox("Rep Count")
        rep_lay = QVBoxLayout()
        self._rep_check = QCheckBox("Enable rep tracking")
        self._rep_check.setChecked(False)
        rep_lay.addWidget(self._rep_check)
        rep_row = QHBoxLayout()
        rep_row.addWidget(QLabel("Max reps:"))
        self._rep_spin = QSpinBox()
        self._rep_spin.setRange(1, 50)
        self._rep_spin.setValue(10)
        rep_row.addWidget(self._rep_spin)
        rep_lay.addLayout(rep_row)
        self._rep_check.toggled.connect(self._rep_spin.setEnabled)
        rep_grp.setLayout(rep_lay)
        left_lay.addWidget(rep_grp)

        rep_overlay_row = QHBoxLayout()
        self._rep_overlay_btn = QToolButton()
        self._rep_overlay_btn.clicked.connect(self._toggle_rep_overlay)
        rep_overlay_row.addWidget(self._rep_overlay_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        self._rep_overlay_lbl = QLabel()
        rep_overlay_row.addWidget(self._rep_overlay_lbl)
        rep_overlay_row.addStretch(1)
        left_lay.addLayout(rep_overlay_row)
        self._sync_rep_overlay_ui()
        left_lay.addWidget(self._ai_summary_check)

        ai_out_grp = QGroupBox("AI Summary Output")
        ai_out_lay = QVBoxLayout()
        ai_mode_row = QHBoxLayout()
        self._ai_show_check = QCheckBox("Show")
        self._ai_show_check.setChecked(True)
        self._ai_voice_check = QCheckBox("Voice")
        self._ai_voice_check.setChecked(False)
        ai_mode_row.addWidget(self._ai_show_check)
        ai_mode_row.addWidget(self._ai_voice_check)
        ai_mode_row.addStretch(1)
        ai_out_lay.addLayout(ai_mode_row)
        self._ai_output_btn = QPushButton("Output AI Summary")
        self._ai_output_btn.clicked.connect(self._apply_ai_summary_output)
        ai_out_lay.addWidget(self._ai_output_btn)
        self._ai_right_btn = QPushButton("AI Summary")
        self._ai_right_btn.clicked.connect(self._show_ai_summary_on_right)
        ai_out_lay.addWidget(self._ai_right_btn)
        ai_out_grp.setLayout(ai_out_lay)
        left_lay.addWidget(ai_out_grp)

        # ── Mirror ──
        self._mirror_check = QCheckBox("Mirror view (selfie mode)")
        self._mirror_check.setChecked(True)
        left_lay.addWidget(self._mirror_check)

        # ── Tolerance panels (collapsed by default via combo) ──
        tune_grp = QGroupBox("Tolerance Panels")
        tune_lay = QVBoxLayout()
        self._tune_combo = QComboBox()
        self._tune_combo.addItem("— Select panel to show —")
        self._tune_combo.addItem("Ankle Tolerance - drag sliders")
        self._tune_combo.addItem("Foot Tolerance - drag sliders")
        self._tune_combo.addItem("Knee Tolerance - drag sliders")
        self._tune_combo.addItem("Hip Tolerance - drag sliders")
        self._tune_combo.addItem("Torso Front Tolerance - drag sliders")
        self._tune_combo.addItem("Display Ratio - drag sliders")
        self._tune_combo.addItem("Rep Logic Thresholds")
        self._tune_combo.addItem("Coach Text")
        self._tune_combo.currentIndexChanged.connect(self._on_tune_selected)
        tune_lay.addWidget(self._tune_combo)

        self._tune_stack: dict[int, QGroupBox] = {}
        self._ankle_grp  = _build_ankle_group();  self._tune_stack[1] = self._ankle_grp
        self._foot_grp   = _build_foot_group();   self._tune_stack[2] = self._foot_grp
        self._knee_grp   = _build_knee_group();   self._tune_stack[3] = self._knee_grp
        self._hip_grp    = _build_hip_group();     self._tune_stack[4] = self._hip_grp
        self._torso_grp  = _build_torso_group();  self._tune_stack[5] = self._torso_grp
        self._display_grp = _build_display_group(); self._tune_stack[6] = self._display_grp
        self._rep_logic_grp = _build_rep_logic_group(); self._tune_stack[7] = self._rep_logic_grp

        # Coach Text is a display label, not sliders
        self._coach_label = QLabel("Coach text will appear here during session")
        self._coach_label.setWordWrap(True)
        self._coach_label.setStyleSheet("background: #111; padding: 6px; font-size: 11px; color: #ccc;")
        self._coach_label.setMinimumHeight(120)
        coach_box = QGroupBox("Coach Text")
        cl = QVBoxLayout()
        cl.addWidget(self._coach_label)
        coach_box.setLayout(cl)
        self._tune_stack[8] = coach_box

        for w in self._tune_stack.values():
            w.setVisible(False)
            tune_lay.addWidget(w)
        tune_grp.setLayout(tune_lay)
        left_lay.addWidget(tune_grp)

        coach_btn_row = QHBoxLayout()
        self._coach_popup_btn = QToolButton()
        self._coach_popup_btn.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MessageBoxInformation))
        self._coach_popup_btn.setToolTip("Open comments in a new window")
        self._coach_popup_btn.clicked.connect(self._open_coach_popup)
        coach_btn_row.addWidget(self._coach_popup_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        coach_btn_row.addWidget(QLabel("Open comments window"))
        coach_btn_row.addStretch(1)
        left_lay.addLayout(coach_btn_row)

        left_lay.addStretch(1)

        # ── Start / Stop / Fullscreen ──
        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start")
        self._start_btn.setObjectName("startBtn")
        self._start_btn.clicked.connect(self._start)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        self._fs_btn = QPushButton("Fullscreen (F11)")
        self._fs_btn.clicked.connect(self._toggle_fullscreen)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        btn_row.addWidget(self._fs_btn)
        left_lay.addLayout(btn_row)

        self._reset_btn = QPushButton("Reset All (R)")
        self._reset_btn.clicked.connect(self._reset_all)
        left_lay.addWidget(self._reset_btn)

        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # ===================== RIGHT: VIDEO =====================
        right_widget = QWidget()
        right_lay = QVBoxLayout(right_widget)
        right_lay.setContentsMargins(0, 0, 0, 0)

        self._right_stack = QStackedWidget()
        self._video_label = QLabel("Press Start to begin")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._video_label.setStyleSheet("background: #111; color: #666; font-size: 18px;")
        self._summary_right_label = QLabel("AI summary will appear here when ready.")
        self._summary_right_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._summary_right_label.setWordWrap(True)
        self._summary_right_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._summary_right_label.setStyleSheet("background: #111; color: #ddd; font-size: 14px; padding: 12px;")
        self._right_stack.addWidget(self._video_label)
        self._right_stack.addWidget(self._summary_right_label)
        self._right_stack.setCurrentIndex(0)
        right_lay.addWidget(self._right_stack, stretch=1)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        # ── Shortcuts ──
        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)
        QShortcut(QKeySequence("Escape"), self, activated=self._exit_fullscreen)
        QShortcut(QKeySequence("R"), self, activated=self._reset_all)

    # ---- slots -----------------------------------------------------------

    def _on_source_index_changed(self, idx: int) -> None:
        """Clear stored file path when switching to a live camera (not Browse)."""
        data = self._src_combo.itemData(idx)
        if data != "file":
            self._video_path = None
            self._file_label.setText("No file selected")

    def _on_source_activated(self, idx: int) -> None:
        """Open file picker whenever the user activates Browse (including re-pick)."""
        data = self._src_combo.itemData(idx)
        if data != "file":
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv *.flv);;All Files (*)",
        )
        if path:
            self._video_path = path
            self._file_label.setText(os.path.basename(path))
        else:
            self._src_combo.setCurrentIndex(0)

    def _on_tune_selected(self, idx: int) -> None:
        for i, w in self._tune_stack.items():
            w.setVisible(i == idx)

    def _get_source(self) -> int | str:
        data = self._src_combo.itemData(self._src_combo.currentIndex())
        if data == "file" and self._video_path:
            return self._video_path
        if isinstance(data, int):
            return data
        return 0

    def _start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._latest_ai_summary = ""
        self._latest_ai_summary_path = ""
        self._summary_right_label.setText("AI summary will appear here when ready.")
        self._right_stack.setCurrentIndex(0)
        self._coach_label.setText("Coach text will appear here during session")
        self._coach_popup.clear_summary_text()
        self._worker = PoseWorker(self)
        self._worker.configure(
            source=self._get_source(),
            model_complexity=self._model_combo.currentData(),
            rep_enabled=self._rep_check.isChecked(),
            max_reps=self._rep_spin.value(),
            mirror=self._mirror_check.isChecked(),
            show_rep_overlay=self._show_rep_overlay,
            ai_summary_enabled=self._ai_summary_check.isChecked(),
        )
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.status_msg.connect(self._on_status)
        self._worker.summary_ready.connect(self._on_ai_summary)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()

    def _open_coach_popup(self) -> None:
        self._coach_popup.show()
        self._coach_popup.raise_()
        self._coach_popup.activateWindow()

    def _sync_rep_overlay_ui(self) -> None:
        if self._show_rep_overlay:
            icon = self.style().standardIcon(self.style().StandardPixmap.SP_DialogYesButton)
            self._rep_overlay_lbl.setText("Rep count on video: ON")
            self._rep_overlay_btn.setToolTip("Hide rep count overlay")
        else:
            icon = self.style().standardIcon(self.style().StandardPixmap.SP_DialogNoButton)
            self._rep_overlay_lbl.setText("Rep count on video: OFF")
            self._rep_overlay_btn.setToolTip("Show rep count overlay")
        self._rep_overlay_btn.setIcon(icon)

    def _toggle_rep_overlay(self) -> None:
        self._show_rep_overlay = not self._show_rep_overlay
        self._sync_rep_overlay_ui()
        if self._worker is not None and self._worker.isRunning():
            self._worker.set_show_rep_overlay(self._show_rep_overlay)

    def _on_finished(self) -> None:
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._video_label.setText("Stopped — press Start to begin")

    @pyqtSlot(np.ndarray, np.ndarray)
    def _on_frame(self, skel_frame: np.ndarray, coach_panel: np.ndarray) -> None:
        # Scale video frame to fit the label
        lw, lh = self._video_label.width(), self._video_label.height()
        if lw > 10 and lh > 10:
            h, w = skel_frame.shape[:2]
            scale = min(lw / w, lh / h)
            nw, nh = int(w * scale), int(h * scale)
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            display = cv2.resize(skel_frame, (nw, nh), interpolation=interp)
        else:
            display = skel_frame
        self._video_label.setPixmap(_cv_to_qpixmap(display))
        self._coach_popup.set_panel_image(coach_panel)

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._video_label.setText(msg)

    @pyqtSlot(str)
    def _on_ai_summary(self, summary: str) -> None:
        self._latest_ai_summary = summary
        self._latest_ai_summary_path = self._save_ai_summary(summary)
        msg = "AI summary is ready. Click 'Output AI Summary' (Show/Voice) or 'AI Summary' button."
        if self._latest_ai_summary_path:
            msg += f"\nSaved: {self._latest_ai_summary_path}"
        self._coach_label.setText(msg)

    def _save_ai_summary(self, summary: str) -> str:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            out_dir = os.path.join(script_dir, "ai_summaries")
            os.makedirs(out_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"ai_summary_{stamp}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(summary.strip() + "\n")
            return out_path
        except Exception:
            return ""

    def _speak_text_async(self, text: str) -> None:
        def _job() -> None:
            ps_text = text.replace("'", "''")
            cmd = (
                "Add-Type -AssemblyName System.Speech; "
                "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$speak.Rate = 0; "
                f"$speak.Speak('{ps_text}');"
            )
            try:
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", cmd],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except Exception:
                pass

        t = threading.Thread(target=_job, daemon=True)
        self._voice_thread = t
        t.start()

    def _apply_ai_summary_output(self) -> None:
        if not self._latest_ai_summary.strip():
            self._coach_label.setText("No AI summary yet. Complete all reps first.")
            return
        do_show = self._ai_show_check.isChecked()
        do_voice = self._ai_voice_check.isChecked()
        if not do_show and not do_voice:
            self._coach_label.setText("Select Show and/or Voice.")
            return
        if do_show:
            self._coach_label.setText(self._latest_ai_summary)
            self._coach_popup.set_summary_text(self._latest_ai_summary)
            self._coach_popup.show()
            self._coach_popup.raise_()
        if do_voice:
            self._speak_text_async(self._latest_ai_summary)

    def _show_ai_summary_on_right(self) -> None:
        if not self._latest_ai_summary.strip():
            self._coach_label.setText("No AI summary yet. Complete all reps first.")
            return
        self._summary_right_label.setText(self._latest_ai_summary)
        self._right_stack.setCurrentIndex(1)

    def _toggle_fullscreen(self) -> None:
        if self._is_fullscreen:
            self.showNormal()
            self._is_fullscreen = False
        else:
            self.showFullScreen()
            self._is_fullscreen = True

    def _exit_fullscreen(self) -> None:
        if self._is_fullscreen:
            self.showNormal()
            self._is_fullscreen = False

    def _reset_all(self) -> None:
        fresh = engine.pose_filter_config()
        from dataclasses import fields as dc_fields
        for f in dc_fields(engine.PoseFilterConfig):
            setattr(engine.CFG, f.name, getattr(fresh, f.name))

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(3000)
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
