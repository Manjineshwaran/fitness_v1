"""
Front-view pose helper: read a video, run MediaPipe Pose, draw skeleton and overlays.

Developer flow: capture frame → pose landmarks → draw skeleton →
left/right shoulder-foot vertical coincidence checks (ratio based) → overlays → scaled preview (ESC).
"""
import asyncio
import ctypes
import os
import tempfile
import threading
from dataclasses import dataclass, fields
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

try:
    import edge_tts as _edge_tts
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False

# Windows MCI (winmm.dll) — plays MP3 without any third-party library.
try:
    _winmm = ctypes.windll.winmm  # type: ignore[attr-defined]
    _WINMM_OK = True
except Exception:
    _WINMM_OK = False

# MediaPipe pose model + drawing helpers (connections between joints).
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()


@dataclass
class PoseFilterConfig:
    """Filter thresholds for display scaling and shoulder-foot vertical checks."""

    # Max size for imshow so the window fits on screen (full frame is still processed above this scale).
    display_max_width: int = 1024
    display_max_height: int = 768

    # Ankle tolerance set (same-side |x_shoulder - x_ankle| / shoulder_width).
    shoulder_foot_align_ratio_max: float = 0.100
    # Four independent tolerance ratios for the four dynamic rails.
    # Left side: shoulder_x - left_inner <= ankle_x <= shoulder_x + left_outer.
    # Right side: shoulder_x - right_inner <= ankle_x <= shoulder_x + right_outer.
    sf_left_inner_offset_ratio: float = 0.04
    sf_left_outer_offset_ratio: float = 0.04
    sf_right_inner_offset_ratio: float = 0.04
    sf_right_outer_offset_ratio: float = 0.04
    # Half-length of tolerance rails around ankle height (normalized by frame height).
    sf_line_half_len_ratio: float = 0.02

    # Foot-index tolerance set (same-side |x_shoulder - x_foot_index| / shoulder_width).
    foot_index_align_ratio_max: float = 0.10
    fi_left_inner_offset_ratio: float = 0.0001
    fi_left_outer_offset_ratio: float = 0.04
    fi_right_inner_offset_ratio: float = 0.04
    fi_right_outer_offset_ratio: float = 0.0001
    # Half-length of tolerance rails around foot-index height.
    fi_line_half_len_ratio: float = 0.02

    # Knee tolerance set (same-side |x_shoulder - x_knee| / shoulder_width).
    knee_align_ratio_max: float = 0.12
    kn_left_inner_offset_ratio: float = 0.04
    kn_left_outer_offset_ratio: float = 0.04
    kn_right_inner_offset_ratio: float = 0.04
    kn_right_outer_offset_ratio: float = 0.04
    # Half-length of tolerance rails around knee height.
    kn_line_half_len_ratio: float = 0.02

    # Hip-center tolerance set: keep mid-knee near vertical line dropped from mid-hip.
    hip_align_ratio_max: float = 0.070
    hip_offset_ratio: float = 0.03

    # Torso front-bend tolerance (mid-shoulder vs mid-hip).
    torso_horizontal_align_ratio_max: float = 0.10
    torso_horizontal_offset_ratio: float = 0.03
    # Front-bend guard from shoulder-hip segment orientation.
    # vertical_ratio = |hip_y - shoulder_y| / torso_segment_length (0..1).
    torso_vertical_gap_min_ratio: float = 0.90
    torso_vertical_gap_offset_ratio: float = 0.05

    # Shoulder level tolerance (left/right shoulder height balance).
    shoulder_level_align_ratio_max: float = 0.04
    shoulder_level_offset_ratio: float = 0.02

    # Squat rep tracking (gap_ratio = (knee_y - hip_y) / shoulder_w, all normalised).
    # 0 = no target (count reps without a cap); GUI sets a positive goal when enabled.
    squat_max_reps: int = 0
    squat_standing_ratio: float = 0.70   # gap_ratio >= this → person is standing / rep complete
    squat_partial_ratio: float = 0.52    # gap_ratio <= this → squat has started
    squat_full_depth_ratio: float = 0.15 # gap_ratio <= this → hip near knee = full depth reached
    squat_too_deep_ratio: float = -0.05  # gap_ratio < this → hip below knee = too deep warning
    # Horizontal helper line placed between hip and knee, measured upward from knee.
    squat_between_line_from_knee_ratio: float = 0.50
    squat_between_line_width_ratio: float = 1.10
    # Fixed rep-timer line config (locked from standing baseline, not moving every frame).
    # Anchor can be "knee" or "foot".
    squat_time_line_anchor: str = "knee"
    squat_time_line_ratio: float = 0.45


@dataclass
class LLMRepExportConfig:
    """Toggle which per-rep variables are exported for LLM summarisation."""

    include_rep_index: bool = True
    include_active_time_sec: bool = True
    include_peak_depth_pct: bool = True
    include_full_depth: bool = True
    include_too_deep: bool = True
    include_min_gap: bool = True
    include_left_right_asym: bool = True


LLM_CFG = LLMRepExportConfig()


def pose_filter_config() -> PoseFilterConfig:
    """Default filter thresholds; call this if you want a fresh config object to customize."""
    return PoseFilterConfig()


CFG = pose_filter_config()

# Separate tune windows: ankle tolerance, foot-index tolerance, and display sizing.
ANKLE_TUNE_WINDOW = "Ankle Tolerance - drag sliders"
FOOT_TUNE_WINDOW = "Foot Tolerance - drag sliders"
KNEE_TUNE_WINDOW = "Knee Tolerance - drag sliders"
HIP_TUNE_WINDOW = "Hip Tolerance - drag sliders"
TORSO_TUNE_WINDOW = "Torso Front Tolerance - drag sliders"
DISPLAY_TUNE_WINDOW = "Display Ratio - drag sliders"
COACH_TEXT_WINDOW = "Coach Text"
MIRROR_VIEW = True  # Selfie-style view: your right side appears on the right.

# When mirror view is on, swap left/right wording and cue routing so prompts
# match what the user sees on screen.
_MIRROR_CUE_SWAP = {
    "ankle_left_inner": "ankle_right_inner",
    "ankle_left_outer": "ankle_right_outer",
    "ankle_right_inner": "ankle_left_inner",
    "ankle_right_outer": "ankle_left_outer",
    "toe_left_inner": "toe_right_inner",
    "toe_left_outer": "toe_right_outer",
    "toe_right_inner": "toe_left_inner",
    "toe_right_outer": "toe_left_outer",
    "knee_left_inner": "knee_right_inner",
    "knee_left_outer": "knee_right_outer",
    "knee_right_inner": "knee_left_inner",
    "knee_right_outer": "knee_left_outer",
    "hip_left": "hip_right",
    "hip_right": "hip_left",
    "torso_x_left": "torso_x_right",
    "torso_x_right": "torso_x_left",
    "shoulder_high_left": "shoulder_high_right",
    "shoulder_high_right": "shoulder_high_left",
}


def _view_side(side: str) -> str:
    if not MIRROR_VIEW:
        return side
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    return side


def _cue_for_view(key: str) -> str:
    return _MIRROR_CUE_SWAP.get(key, key) if MIRROR_VIEW else key


def _clamp_int(x, lo, hi):
    return max(lo, min(hi, int(round(x))))


# Keys that have been continuously violated for ≥ SUSTAIN_SEC seconds this frame.
# Updated once per frame in the main loop; read by draw functions for blink gating.
_sustained_cues: frozenset[str] = frozenset()

# How long (seconds) a violation must persist before blinking and voice fire.
SUSTAIN_SEC: float = 1.0


class ViolationTracker:
    """
    Per-key onset timer.
    • update(active_keys) — call every frame with the set of currently violated cue keys.
      Keys that have cleared are removed instantly; no grace period on recovery.
    • get_sustained(min_sec) — returns only keys that have been bad for ≥ min_sec
      without interruption.
    """

    def __init__(self) -> None:
        self._onset: dict[str, float] = {}

    def update(self, active_keys: set[str]) -> None:
        now = cv2.getTickCount() / cv2.getTickFrequency()
        for k in list(self._onset):
            if k not in active_keys:
                del self._onset[k]       # cleared immediately when pose corrects
        for k in active_keys:
            if k not in self._onset:
                self._onset[k] = now     # first frame of this violation

    def get_sustained(self, min_sec: float = SUSTAIN_SEC) -> frozenset[str]:
        now = cv2.getTickCount() / cv2.getTickFrequency()
        return frozenset(k for k, t in self._onset.items() if now - t >= min_sec)

    def reset(self) -> None:
        self._onset.clear()


def _blink_red_yellow(key: str = "") -> tuple[int, int, int]:
    """
    Blink red/yellow when the violation is sustained (key in _sustained_cues).
    If key is given but not yet sustained, return a dim static red — still
    an alert colour, but no blink, so momentary wobbles don't flash.
    Pass key="" (default) to always blink (used for legacy call sites that
    have no per-key context).
    """
    if key and key not in _sustained_cues:
        return (60, 60, 200)   # dim static red — violation exists but not yet confirmed
    t = cv2.getTickCount() / cv2.getTickFrequency()
    return (0, 0, 255) if int(t * 4) % 2 == 0 else (0, 255, 255)


# ---------------------------------------------------------------------------
# Trainer coaching engine
# ---------------------------------------------------------------------------
# On-screen banner text — very short, immediate.
_TRAINER_CUES: dict[str, list[str]] = {
    "ankle_left_inner":  ["Left foot → step it in",      "Left foot too wide — move inside"],
    "ankle_left_outer":  ["Left foot → open it out",     "Left foot too close — step wider"],
    "ankle_right_inner": ["Right foot → step it in",     "Right foot too wide — move inside"],
    "ankle_right_outer": ["Right foot → open it out",    "Right foot too close — step wider"],

    "toe_left_inner":  ["Left toes → turn inward",       "Left toes too wide — rotate in"],
    "toe_left_outer":  ["Left toes → spread outward",    "Left toes too close — rotate out"],
    "toe_right_inner": ["Right toes → turn inward",      "Right toes too wide — rotate in"],
    "toe_right_outer": ["Right toes → spread outward",   "Right toes too close — rotate out"],

    "knee_left_inner":  ["Left knee → push it out",      "Left knee caving — drive outward"],
    "knee_left_outer":  ["Left knee → bring it in",      "Left knee flaring — track your foot"],
    "knee_right_inner": ["Right knee → push it out",     "Right knee caving — drive outward"],
    "knee_right_outer": ["Right knee → bring it in",     "Right knee flaring — track your foot"],

    "hip_left":  ["Shift weight right — hips off centre", "Centre your squat"],
    "hip_right": ["Shift weight left — hips off centre",  "Centre your squat"],

    "torso_x_left":  ["Lean right — you're tilting left",  "Centre your chest"],
    "torso_x_right": ["Lean left — you're tilting right",  "Centre your chest"],
    "torso_bend":    ["Chest up! Back straight",            "Don't bend forward"],

    "shoulder_high_left":  ["Drop left shoulder",  "Level your shoulders"],
    "shoulder_high_right": ["Drop right shoulder", "Level your shoulders"],

    # Squat rep events
    "squat_too_deep":  ["Too deep! Rise up a little",     "Come up — you're below your knees"],
    "squat_go_deeper": ["Go deeper — not a full rep yet", "Squat lower to count this rep"],
}

# Spoken voice phrases — short, direct, gym-trainer style (pre-baked to MP3 at startup).
# Two alternating phrases per key to avoid repetition.
_VOICE_CUES: dict[str, list[str]] = {
    "ankle_left_inner":  ["Left foot, step it in.",          "Move your left foot inward."],
    "ankle_left_outer":  ["Left foot, open it out.",          "Step your left foot wider."],
    "ankle_right_inner": ["Right foot, step it in.",          "Move your right foot inward."],
    "ankle_right_outer": ["Right foot, open it out.",         "Step your right foot wider."],

    "toe_left_inner":  ["Left toes, turn them in.",           "Rotate your left foot inward."],
    "toe_left_outer":  ["Left toes, spread them out.",        "Rotate your left foot outward."],
    "toe_right_inner": ["Right toes, turn them in.",          "Rotate your right foot inward."],
    "toe_right_outer": ["Right toes, spread them out.",       "Rotate your right foot outward."],

    "knee_left_inner":  ["Left knee, push it out.",           "Drive your left knee outward."],
    "knee_left_outer":  ["Left knee, bring it in.",           "Track your left knee over your foot."],
    "knee_right_inner": ["Right knee, push it out.",          "Drive your right knee outward."],
    "knee_right_outer": ["Right knee, bring it in.",          "Track your right knee over your foot."],

    "hip_left":  ["Shift your weight to the right.",          "Centre your hips, you're leaning left."],
    "hip_right": ["Shift your weight to the left.",           "Centre your hips, you're leaning right."],

    "torso_x_left":  ["You're leaning left. Stand straight.", "Bring your chest back to centre."],
    "torso_x_right": ["You're leaning right. Stand straight.","Bring your chest back to centre."],
    "torso_bend":    ["Chest up! Back straight.",             "Don't bend forward. Keep your spine tall."],

    "shoulder_high_left":  ["Drop your left shoulder.",       "Level up — left shoulder is too high."],
    "shoulder_high_right": ["Drop your right shoulder.",      "Level up — right shoulder is too high."],

    "squat_too_deep":  ["Too deep! Rise up a little.",        "You are going too deep. Come up slightly."],
    "squat_go_deeper": ["Go a little deeper for a full rep.", "Squat lower — you need to reach your knees."],

    "squat_milestone_half": [
        "Nice work — you are halfway through your set.",
        "Halfway there. Keep your form steady.",
    ],
    "squat_milestone_three_quarter": [
        "Three quarters done. Push a little more — you have got this.",
        "Almost there. Dig in and finish strong.",
    ],
    "squat_goal_complete": [
        "Congratulations — you finished every rep.",
        "Great job. Set complete — take a breath.",
    ],
}


class TrainerCoach:
    """
    Throttled per-cue message display.
    Each cue key shows once per cooldown seconds, cycling through its message list.
    """
    def __init__(self, cooldown_sec: float = 3.0):
        self._cooldown = cooldown_sec
        self._last_shown: dict[str, float] = {}
        self._cycle_idx: dict[str, int] = {}

    def get_cue(self, key: str) -> str | None:
        """Return the next message for this key if cooldown has passed, else None."""
        now = cv2.getTickCount() / cv2.getTickFrequency()
        if now - self._last_shown.get(key, -999) >= self._cooldown:
            self._last_shown[key] = now
            msgs = _TRAINER_CUES.get(key, [])
            if not msgs:
                return None
            idx = self._cycle_idx.get(key, 0) % len(msgs)
            self._cycle_idx[key] = idx + 1
            return msgs[idx]
        return None

    def reset(self):
        self._last_shown.clear()
        self._cycle_idx.clear()


def draw_trainer_cue(
    frame,
    coach: TrainerCoach,
    voice: "VoiceCoach | None",
    key: str,
) -> str | None:
    """Return trainer cue text and speak it; text is rendered in separate panel."""
    msg = coach.get_cue(key)
    if msg is None:
        return None
    if voice is not None:
        voice.speak(key)
    return msg


def get_voice_status_text(voice: "VoiceCoach | None") -> str:
    """Return voice warmup text for separate coach panel."""
    if voice is None:
        return "Voice unavailable"
    return "Voice ready" if voice.is_ready() else "Voice warming up..."


def render_coach_text_panel(
    panel,
    status: str,
    status_color,
    trainer_msg: str | None,
    voice_status: str,
    squat: "SquatRepTracker",
    sf_label: str,
    sf_color,
    left_ratio: float,
    right_ratio: float,
    fi_label: str,
    fi_color,
    fi_left_ratio: float,
    fi_right_ratio: float,
    kn_label: str,
    kn_color,
    kn_left_ratio: float,
    kn_right_ratio: float,
    hip_label: str,
    hip_color,
    hip_dx_ratio: float,
    torso_label: str,
    torso_color,
    torso_dx_ratio: float,
    torso_v_gap_ratio: float,
    shlvl_label: str,
    shlvl_color,
    shlvl_dy_ratio: float,
) -> None:
    """Render all coaching/status text into a separate window panel."""
    panel[:] = 18
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 32

    def line(text, color=(220, 220, 220), scale=0.52, thick=1, dy=28):
        nonlocal y
        cv2.putText(panel, text, (14, y), font, scale, color, thick, cv2.LINE_AA)
        y += dy

    cv2.putText(panel, "Coach Text", (14, y), font, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    y += 34
    line(f"Status: {status}", status_color, 0.58, 2, 30)
    line(f"Voice: {voice_status}", (80, 180, 255))

    if trainer_msg:
        line(f"Trainer: {trainer_msg}", (0, 200, 255), 0.58, 2, 32)
    else:
        line("Trainer: --", (140, 140, 140))

    if not squat.is_calibrated:
        line(f"Squat: calibrating... stand still ({squat.calibration_pct}%)", (0, 200, 255), 0.56, 2, 28)
    else:
        depth_bar = f"{int(squat.depth_pct * 100):3d}%"
        state_str = "IN SQUAT" if squat._in_squat else "standing"
        line(f"Depth: {depth_bar}  [{state_str}]  gap={squat._last_raw_gap:.3f}", (200, 200, 200), 0.52)
        if squat.too_deep:
            line("Squat warning: Too deep, rise up a little", (0, 140, 255), 0.58, 2, 30)
        elif squat.partial_warn:
            line("Squat warning: Go deeper for a full rep", (0, 200, 255), 0.58, 2, 30)
        else:
            line("Squat warning: --", (140, 140, 140))

    y += 6
    line(sf_label, sf_color, 0.56, 2)
    line(f"Ankle  L={left_ratio:.3f} R={right_ratio:.3f} | align={CFG.shoulder_foot_align_ratio_max:.3f}")
    line(fi_label, fi_color, 0.56, 2)
    line(f"FootIdx L={fi_left_ratio:.3f} R={fi_right_ratio:.3f} | align={CFG.foot_index_align_ratio_max:.3f}")
    line(kn_label, kn_color, 0.56, 2)
    line(f"Knee   L={kn_left_ratio:.3f} R={kn_right_ratio:.3f} | align={CFG.knee_align_ratio_max:.3f}")
    line(hip_label, hip_color, 0.56, 2)
    line(f"Hip dx={hip_dx_ratio:.3f} | max={CFG.hip_align_ratio_max + CFG.hip_offset_ratio:.3f}")
    line(torso_label, torso_color, 0.56, 2)
    line(f"Torso dx={torso_dx_ratio:.3f} | v_ratio={torso_v_gap_ratio:.3f}")
    line(shlvl_label, shlvl_color, 0.56, 2)
    line(f"Shoulder dy={shlvl_dy_ratio:.3f} | max={CFG.shoulder_level_align_ratio_max + CFG.shoulder_level_offset_ratio:.3f}")


# ---------------------------------------------------------------------------
# Voice coaching engine  (edge-tts neural voice + Windows winmm MP3 playback)
# ---------------------------------------------------------------------------

_TTS_VOICE = "en-US-JennyNeural"   # Natural female trainer voice


_MCI_ALIAS = "vc_mp3"


def _mci(cmd: str) -> None:
    if _WINMM_OK:
        _winmm.mciSendStringW(cmd, None, 0, None)


def _mci_stop() -> None:
    """Immediately stop + close whatever is playing on our alias."""
    _mci(f"stop {_MCI_ALIAS}")
    _mci(f"close {_MCI_ALIAS}")


def _mci_play_blocking(path: str) -> None:
    """Open, play (blocking), close.  Caller should _mci_stop first if needed."""
    p = path.replace("/", "\\")
    _mci(f'open "{p}" type mpegvideo alias {_MCI_ALIAS}')
    _mci(f"play {_MCI_ALIAS} wait")
    _mci(f"close {_MCI_ALIAS}")


_REP_NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
}


def _rep_number_phrase(n: int) -> str:
    """Return a very short spoken word for *n* — keeps audio clip under ~0.5s."""
    if n in _REP_NUMBER_WORDS:
        return _REP_NUMBER_WORDS[n]
    if 21 <= n < 100:
        tens, ones = divmod(n, 10)
        base = _REP_NUMBER_WORDS[tens * 10] if tens * 10 in _REP_NUMBER_WORDS else str(tens * 10)
        if ones == 0:
            return base
        return f"{base} {_REP_NUMBER_WORDS.get(ones, str(ones))}"
    return str(n)


async def _tts_to_mp3(text: str, path: str) -> None:
    """Synthesise *text* with edge-tts and write MP3 bytes to *path*."""
    communicate = _edge_tts.Communicate(text, _TTS_VOICE)
    with open(path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])


class VoiceCoach:
    """
    Trainer-style voice engine — **interrupt instantly, no queue, no replay**.

    Rules:
    1. Only one clip plays at any moment.
    2. When a new cue arrives it REPLACES the current one: the ongoing clip
       is stopped immediately (MCI stop) so the new cue starts without lag.
    3. Nothing is queued or stored — if a newer cue replaces a pending one,
       the old cue is simply dropped.
    4. Correction keys have a short same-key cooldown so we don't nag.
       Rep-count and milestone cues have no cooldown.
    """

    def __init__(self, cooldown_sec: float = 3.0):
        self._cooldown = cooldown_sec
        self._last_spoken_key: str = ""
        self._last_spoken_t: float = -999.0
        self._cache: dict[str, list[str]] = {}
        self._cache_idx: dict[str, int] = {}
        self._ready = False
        self._tmp_dir = tempfile.mkdtemp(prefix="vc_cache_")
        self._rep_cache: dict[int, str] = {}

        # Single slot for the latest request. The worker picks whatever is in
        # the slot when it wakes up; any older value is silently discarded.
        self._pending_lock = threading.Lock()
        self._pending_path: str | None = None
        self._pending_key: str = ""
        self._wake = threading.Event()
        self._shutdown_flag = False

        self._play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self._play_thread.start()

        self._bake_thread = threading.Thread(target=self._bake_all, daemon=True)
        self._bake_thread.start()

    # ---- background TTS baking -------------------------------------------
    def _bake_all(self) -> None:
        if not _TTS_AVAILABLE:
            return

        async def _gen_all() -> None:
            jobs: list[tuple[str, str | int, int, str]] = []
            coros = []
            for key, phrases in _VOICE_CUES.items():
                for i, text in enumerate(phrases):
                    path = os.path.join(self._tmp_dir, f"{key}_{i}.mp3")
                    jobs.append(("cue", key, i, path))
                    coros.append(_tts_to_mp3(text, path))
            if CFG.squat_max_reps > 0:
                for rep_i in range(0, CFG.squat_max_reps + 1):
                    rep_text = _rep_number_phrase(rep_i)
                    rep_path = os.path.join(self._tmp_dir, f"rep_{rep_i}.mp3")
                    jobs.append(("rep", str(rep_i), rep_i, rep_path))
                    coros.append(_tts_to_mp3(rep_text, rep_path))
            results = await asyncio.gather(*coros, return_exceptions=True)
            for (kind, key, i, path), err in zip(jobs, results):
                if isinstance(err, Exception) or not os.path.exists(path) or os.path.getsize(path) == 0:
                    continue
                if kind == "cue":
                    bucket = self._cache.setdefault(key, [])
                    while len(bucket) <= i:
                        bucket.append("")
                    bucket[i] = path
                else:
                    self._rep_cache[i] = path
            self._ready = True
            print(f"[VoiceCoach] {sum(len(v) for v in self._cache.values())} audio cues ready.")

        try:
            asyncio.run(_gen_all())
        except Exception as e:
            print(f"[VoiceCoach] Pre-bake failed: {e}")

    # ---- play worker (single thread, latest-wins with instant interrupt) -
    def _play_worker(self) -> None:
        while True:
            self._wake.wait()
            if self._shutdown_flag:
                _mci_stop()
                break

            with self._pending_lock:
                path = self._pending_path
                self._pending_path = None
                self._pending_key = ""
                self._wake.clear()

            if not path or not _WINMM_OK:
                continue

            # Make sure nothing is still playing from a previous clip before
            # we start the new one — prevents any audible overlap.
            _mci_stop()
            try:
                _mci_play_blocking(path)
            except Exception:
                pass

    # ---- public API ------------------------------------------------------
    def _enqueue(self, path: str, key: str) -> None:
        """Replace the current request with *path* and interrupt playback."""
        with self._pending_lock:
            self._pending_path = path
            self._pending_key = key
        # Stop the worker's current blocking MCI call immediately so the
        # new clip starts with no lag. _mci_stop is thread-safe.
        _mci_stop()
        self._wake.set()

    def speak(self, key: str, rep_count: int = 0, rep_goal: int = 0) -> None:  # noqa: ARG002
        """Speak the cue identified by *key* (with same-key cooldown)."""
        if not self._ready:
            return
        now = cv2.getTickCount() / cv2.getTickFrequency()
        if key == self._last_spoken_key and now - self._last_spoken_t < self._cooldown:
            return

        paths = self._cache.get(key, [])
        if not paths:
            return
        idx = self._cache_idx.get(key, 0) % len(paths)
        path = paths[idx]
        if not path or not os.path.exists(path):
            return

        self._cache_idx[key] = idx + 1
        self._last_spoken_key = key
        self._last_spoken_t = now
        self._enqueue(path, key)

    def update_rep(self, rep_count: int, rep_goal: int) -> None:  # noqa: ARG002
        """Kept for API compatibility — no stored state needed now."""
        return

    def speak_rep_progress(self, rep_count: int, rep_goal: int) -> None:
        """Say "Rep X of Y" immediately when a rep is counted (no cooldown)."""
        if not self._ready or rep_goal <= 0:
            return
        rep_path = self._rep_cache.get(rep_count, "")
        if not rep_path or not os.path.exists(rep_path):
            return
        key = f"rep_{rep_count}"
        self._last_spoken_key = key
        self._last_spoken_t = cv2.getTickCount() / cv2.getTickFrequency()
        self._enqueue(rep_path, key)

    def is_ready(self) -> bool:
        return self._ready

    def reset(self) -> None:
        self._last_spoken_key = ""
        self._last_spoken_t = -999.0
        with self._pending_lock:
            self._pending_path = None
            self._pending_key = ""
        _mci_stop()

    def shutdown(self) -> None:
        self._shutdown_flag = True
        _mci_stop()
        self._wake.set()
        try:
            import shutil
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        except Exception:
            pass


def speak_squat_rep_milestones(
    voice: VoiceCoach | None,
    old_count: int,
    new_count: int,
    goal: int,
    state: dict,
) -> None:
    """
    Speak encouragement at half and three-quarter progress, and congratulations at goal.

    *state* should be a mutable dict with optional keys "half", "tq", "done" (booleans).
    Call after other voice cues in the same frame so milestones can take priority.
    """
    if voice is None or not voice.is_ready() or goal <= 0 or new_count <= old_count:
        return
    if state.get("done"):
        return
    if old_count < goal <= new_count:
        voice.speak("squat_goal_complete")
        state["done"] = True
        state["half"] = True
        state["tq"] = True
        return
    half = (goal + 1) // 2
    if half < goal and old_count < half <= new_count and not state.get("half"):
        voice.speak("squat_milestone_half")
        state["half"] = True
    tq = (3 * goal + 3) // 4
    if tq < goal and old_count < tq <= new_count and not state.get("tq"):
        voice.speak("squat_milestone_three_quarter")
        state["tq"] = True


def setup_pose_tune_trackbars():
    """Create trackbars that write into global CFG each frame (call once before the main loop)."""
    cv2.namedWindow(ANKLE_TUNE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ANKLE_TUNE_WINDOW, 330, 330)
    cv2.namedWindow(FOOT_TUNE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(FOOT_TUNE_WINDOW, 330, 330)
    cv2.namedWindow(KNEE_TUNE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(KNEE_TUNE_WINDOW, 330, 330)
    cv2.namedWindow(HIP_TUNE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(HIP_TUNE_WINDOW, 330, 200)
    cv2.namedWindow(TORSO_TUNE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TORSO_TUNE_WINDOW, 330, 260)
    cv2.namedWindow(DISPLAY_TUNE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DISPLAY_TUNE_WINDOW, 330, 200)

    cv2.createTrackbar(
        "sf_align x1000",
        ANKLE_TUNE_WINDOW,
        _clamp_int(CFG.shoulder_foot_align_ratio_max * 1000, 20, 800),
        800,
        lambda p: setattr(CFG, "shoulder_foot_align_ratio_max", max(0.02, p / 1000.0)),
    )
    cv2.createTrackbar(
        "sf_L_in x1000",
        ANKLE_TUNE_WINDOW,
        _clamp_int(CFG.sf_left_inner_offset_ratio * 1000, 0, 400),
        400,
        lambda p: setattr(CFG, "sf_left_inner_offset_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "sf_L_out x1000",
        ANKLE_TUNE_WINDOW,
        _clamp_int(CFG.sf_left_outer_offset_ratio * 1000, 0, 400),
        400,
        lambda p: setattr(CFG, "sf_left_outer_offset_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "sf_R_in x1000",
        ANKLE_TUNE_WINDOW,
        _clamp_int(CFG.sf_right_inner_offset_ratio * 1000, 0, 400),
        400,
        lambda p: setattr(CFG, "sf_right_inner_offset_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "sf_R_out x1000",
        ANKLE_TUNE_WINDOW,
        _clamp_int(CFG.sf_right_outer_offset_ratio * 1000, 0, 400),
        400,
        lambda p: setattr(CFG, "sf_right_outer_offset_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "sf_line_len x1000",
        ANKLE_TUNE_WINDOW,
        _clamp_int(CFG.sf_line_half_len_ratio * 1000, 0, 120),
        120,
        lambda p: setattr(CFG, "sf_line_half_len_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "fi_align x1000",
        FOOT_TUNE_WINDOW,
        _clamp_int(CFG.foot_index_align_ratio_max * 1000, 20, 800),
        800,
        lambda p: setattr(CFG, "foot_index_align_ratio_max", max(0.02, p / 1000.0)),
    )
    cv2.createTrackbar(
        "fi_L_in x1000",
        FOOT_TUNE_WINDOW,
        _clamp_int(CFG.fi_left_inner_offset_ratio * 1000, 0, 400),
        400,
        lambda p: setattr(CFG, "fi_left_inner_offset_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "fi_L_out x1000",
        FOOT_TUNE_WINDOW,
        _clamp_int(CFG.fi_left_outer_offset_ratio * 1000, 0, 400),
        400,
        lambda p: setattr(CFG, "fi_left_outer_offset_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "fi_R_in x1000",
        FOOT_TUNE_WINDOW,
        _clamp_int(CFG.fi_right_inner_offset_ratio * 1000, 0, 400),
        400,
        lambda p: setattr(CFG, "fi_right_inner_offset_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "fi_R_out x1000",
        FOOT_TUNE_WINDOW,
        _clamp_int(CFG.fi_right_outer_offset_ratio * 1000, 0, 400),
        400,
        lambda p: setattr(CFG, "fi_right_outer_offset_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "fi_line_len x1000",
        FOOT_TUNE_WINDOW,
        _clamp_int(CFG.fi_line_half_len_ratio * 1000, 0, 120),
        120,
        lambda p: setattr(CFG, "fi_line_half_len_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "kn_align x1000",
        KNEE_TUNE_WINDOW,
        _clamp_int(CFG.knee_align_ratio_max * 1000, 20, 800),
        800,
        lambda p: setattr(CFG, "knee_align_ratio_max", max(0.02, p / 1000.0)),
    )
    cv2.createTrackbar(
        "kn_L_in x1000",
        KNEE_TUNE_WINDOW,
        _clamp_int(CFG.kn_left_inner_offset_ratio * 1000, 0, 400),
        400,
        lambda p: setattr(CFG, "kn_left_inner_offset_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "kn_L_out x1000",
        KNEE_TUNE_WINDOW,
        _clamp_int(CFG.kn_left_outer_offset_ratio * 1000, 0, 400),
        400,
        lambda p: setattr(CFG, "kn_left_outer_offset_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "kn_R_in x1000",
        KNEE_TUNE_WINDOW,
        _clamp_int(CFG.kn_right_inner_offset_ratio * 1000, 0, 400),
        400,
        lambda p: setattr(CFG, "kn_right_inner_offset_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "kn_R_out x1000",
        KNEE_TUNE_WINDOW,
        _clamp_int(CFG.kn_right_outer_offset_ratio * 1000, 0, 400),
        400,
        lambda p: setattr(CFG, "kn_right_outer_offset_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "kn_line_len x1000",
        KNEE_TUNE_WINDOW,
        _clamp_int(CFG.kn_line_half_len_ratio * 1000, 0, 120),
        120,
        lambda p: setattr(CFG, "kn_line_half_len_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "hip_align x1000",
        HIP_TUNE_WINDOW,
        _clamp_int(CFG.hip_align_ratio_max * 1000, 10, 800),
        800,
        lambda p: setattr(CFG, "hip_align_ratio_max", max(0.01, p / 1000.0)),
    )
    cv2.createTrackbar(
        "hip_offset x1000",
        HIP_TUNE_WINDOW,
        _clamp_int(CFG.hip_offset_ratio * 1000, 0, 400),
        400,
        lambda p: setattr(CFG, "hip_offset_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "torso_x_align x1000",
        TORSO_TUNE_WINDOW,
        _clamp_int(CFG.torso_horizontal_align_ratio_max * 1000, 10, 800),
        800,
        lambda p: setattr(CFG, "torso_horizontal_align_ratio_max", max(0.01, p / 1000.0)),
    )
    cv2.createTrackbar(
        "torso_x_off x1000",
        TORSO_TUNE_WINDOW,
        _clamp_int(CFG.torso_horizontal_offset_ratio * 1000, 0, 400),
        400,
        lambda p: setattr(CFG, "torso_horizontal_offset_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "torso_vmin x1000",
        TORSO_TUNE_WINDOW,
        _clamp_int(CFG.torso_vertical_gap_min_ratio * 1000, 50, 995),
        995,
        lambda p: setattr(CFG, "torso_vertical_gap_min_ratio", max(0.05, min(0.995, p / 1000.0))),
    )
    cv2.createTrackbar(
        "torso_voff x1000",
        TORSO_TUNE_WINDOW,
        _clamp_int(CFG.torso_vertical_gap_offset_ratio * 1000, 0, 500),
        500,
        lambda p: setattr(CFG, "torso_vertical_gap_offset_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "sh_level x1000",
        TORSO_TUNE_WINDOW,
        _clamp_int(CFG.shoulder_level_align_ratio_max * 1000, 1, 600),
        600,
        lambda p: setattr(CFG, "shoulder_level_align_ratio_max", max(0.001, p / 1000.0)),
    )
    cv2.createTrackbar(
        "sh_off x1000",
        TORSO_TUNE_WINDOW,
        _clamp_int(CFG.shoulder_level_offset_ratio * 1000, 0, 300),
        300,
        lambda p: setattr(CFG, "shoulder_level_offset_ratio", p / 1000.0),
    )
    cv2.createTrackbar(
        "disp_w",
        DISPLAY_TUNE_WINDOW,
        _clamp_int((CFG.display_max_width - 480) // 4, 0, 360),
        360,
        lambda p: setattr(CFG, "display_max_width", 480 + p * 4),
    )
    cv2.createTrackbar(
        "disp_h",
        DISPLAY_TUNE_WINDOW,
        _clamp_int((CFG.display_max_height - 240) // 4, 0, 220),
        220,
        lambda p: setattr(CFG, "display_max_height", 240 + p * 4),
    )


def sync_pose_tune_trackbars():
    """Set trackbar positions from current CFG (e.g. after R reset)."""
    cv2.setTrackbarPos("sf_align x1000", ANKLE_TUNE_WINDOW, _clamp_int(CFG.shoulder_foot_align_ratio_max * 1000, 20, 800))
    cv2.setTrackbarPos("sf_L_in x1000", ANKLE_TUNE_WINDOW, _clamp_int(CFG.sf_left_inner_offset_ratio * 1000, 0, 400))
    cv2.setTrackbarPos("sf_L_out x1000", ANKLE_TUNE_WINDOW, _clamp_int(CFG.sf_left_outer_offset_ratio * 1000, 0, 400))
    cv2.setTrackbarPos("sf_R_in x1000", ANKLE_TUNE_WINDOW, _clamp_int(CFG.sf_right_inner_offset_ratio * 1000, 0, 400))
    cv2.setTrackbarPos("sf_R_out x1000", ANKLE_TUNE_WINDOW, _clamp_int(CFG.sf_right_outer_offset_ratio * 1000, 0, 400))
    cv2.setTrackbarPos("sf_line_len x1000", ANKLE_TUNE_WINDOW, _clamp_int(CFG.sf_line_half_len_ratio * 1000, 0, 120))
    cv2.setTrackbarPos("fi_align x1000", FOOT_TUNE_WINDOW, _clamp_int(CFG.foot_index_align_ratio_max * 1000, 20, 800))
    cv2.setTrackbarPos("fi_L_in x1000", FOOT_TUNE_WINDOW, _clamp_int(CFG.fi_left_inner_offset_ratio * 1000, 0, 400))
    cv2.setTrackbarPos("fi_L_out x1000", FOOT_TUNE_WINDOW, _clamp_int(CFG.fi_left_outer_offset_ratio * 1000, 0, 400))
    cv2.setTrackbarPos("fi_R_in x1000", FOOT_TUNE_WINDOW, _clamp_int(CFG.fi_right_inner_offset_ratio * 1000, 0, 400))
    cv2.setTrackbarPos("fi_R_out x1000", FOOT_TUNE_WINDOW, _clamp_int(CFG.fi_right_outer_offset_ratio * 1000, 0, 400))
    cv2.setTrackbarPos("fi_line_len x1000", FOOT_TUNE_WINDOW, _clamp_int(CFG.fi_line_half_len_ratio * 1000, 0, 120))
    cv2.setTrackbarPos("kn_align x1000", KNEE_TUNE_WINDOW, _clamp_int(CFG.knee_align_ratio_max * 1000, 20, 800))
    cv2.setTrackbarPos("kn_L_in x1000", KNEE_TUNE_WINDOW, _clamp_int(CFG.kn_left_inner_offset_ratio * 1000, 0, 400))
    cv2.setTrackbarPos("kn_L_out x1000", KNEE_TUNE_WINDOW, _clamp_int(CFG.kn_left_outer_offset_ratio * 1000, 0, 400))
    cv2.setTrackbarPos("kn_R_in x1000", KNEE_TUNE_WINDOW, _clamp_int(CFG.kn_right_inner_offset_ratio * 1000, 0, 400))
    cv2.setTrackbarPos("kn_R_out x1000", KNEE_TUNE_WINDOW, _clamp_int(CFG.kn_right_outer_offset_ratio * 1000, 0, 400))
    cv2.setTrackbarPos("kn_line_len x1000", KNEE_TUNE_WINDOW, _clamp_int(CFG.kn_line_half_len_ratio * 1000, 0, 120))
    cv2.setTrackbarPos("hip_align x1000", HIP_TUNE_WINDOW, _clamp_int(CFG.hip_align_ratio_max * 1000, 10, 800))
    cv2.setTrackbarPos("hip_offset x1000", HIP_TUNE_WINDOW, _clamp_int(CFG.hip_offset_ratio * 1000, 0, 400))
    cv2.setTrackbarPos("torso_x_align x1000", TORSO_TUNE_WINDOW, _clamp_int(CFG.torso_horizontal_align_ratio_max * 1000, 10, 800))
    cv2.setTrackbarPos("torso_x_off x1000", TORSO_TUNE_WINDOW, _clamp_int(CFG.torso_horizontal_offset_ratio * 1000, 0, 400))
    cv2.setTrackbarPos("torso_vmin x1000", TORSO_TUNE_WINDOW, _clamp_int(CFG.torso_vertical_gap_min_ratio * 1000, 50, 995))
    cv2.setTrackbarPos("torso_voff x1000", TORSO_TUNE_WINDOW, _clamp_int(CFG.torso_vertical_gap_offset_ratio * 1000, 0, 500))
    cv2.setTrackbarPos("sh_level x1000", TORSO_TUNE_WINDOW, _clamp_int(CFG.shoulder_level_align_ratio_max * 1000, 1, 600))
    cv2.setTrackbarPos("sh_off x1000", TORSO_TUNE_WINDOW, _clamp_int(CFG.shoulder_level_offset_ratio * 1000, 0, 300))
    cv2.setTrackbarPos("disp_w", DISPLAY_TUNE_WINDOW, _clamp_int((CFG.display_max_width - 480) // 4, 0, 360))
    cv2.setTrackbarPos("disp_h", DISPLAY_TUNE_WINDOW, _clamp_int((CFG.display_max_height - 240) // 4, 0, 220))


def reset_pose_filter_to_defaults():
    """Copy default dataclass values onto CFG and refresh trackbars."""
    fresh = pose_filter_config()
    for f in fields(PoseFilterConfig):
        setattr(CFG, f.name, getattr(fresh, f.name))
    sync_pose_tune_trackbars()


def check_shoulder_foot_vertical(landmarks):
    """
    Front view: each foot should be vertically under its same-side shoulder.
    Uses ratio for scale invariance:
    left_ratio = |x_left_shoulder - x_left_ankle| / shoulder_width
    right_ratio = |x_right_shoulder - x_right_ankle| / shoulder_width

    Returns:
        (label, BGR color, left_ratio, right_ratio, cue_keys)
    """
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    shoulder_w = max(abs(ls.x - rs.x), 1e-6)
    left_dx_ratio = (la.x - ls.x) / shoulder_w
    right_dx_ratio = (ra.x - rs.x) / shoulder_w
    l_lo = -(CFG.shoulder_foot_align_ratio_max + CFG.sf_left_inner_offset_ratio)
    l_hi = +(CFG.shoulder_foot_align_ratio_max + CFG.sf_left_outer_offset_ratio)
    r_lo = -(CFG.shoulder_foot_align_ratio_max + CFG.sf_right_inner_offset_ratio)
    r_hi = +(CFG.shoulder_foot_align_ratio_max + CFG.sf_right_outer_offset_ratio)

    issues = []
    cue_keys = []
    if not (l_lo <= left_dx_ratio <= l_hi):
        issues.append(f"{_view_side('left')} foot not under {_view_side('left')} shoulder")
        cue_keys.append(_cue_for_view("ankle_left_inner" if left_dx_ratio < l_lo else "ankle_left_outer"))
    if not (r_lo <= right_dx_ratio <= r_hi):
        issues.append(f"{_view_side('right')} foot not under {_view_side('right')} shoulder")
        # Right side is mirrored in x-sign logic vs left:
        # too low (more negative) means too inward -> cue should be "outer".
        cue_keys.append(_cue_for_view("ankle_right_outer" if right_dx_ratio < r_lo else "ankle_right_inner"))
    if issues:
        return "Shoulder-Foot: " + "; ".join(issues), (0, 0, 255), abs(left_dx_ratio), abs(right_dx_ratio), cue_keys
    return "Shoulder-Foot: OK", (0, 255, 0), abs(left_dx_ratio), abs(right_dx_ratio), []


def check_shoulder_foot_index_vertical(landmarks):
    """
    Same logic as ankle checker, but uses foot index keypoints:
    LEFT_FOOT_INDEX (31) and RIGHT_FOOT_INDEX (32).

    Returns:
        (label, BGR color, left_ratio, right_ratio, cue_keys)
    """
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lfi = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
    rfi = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

    shoulder_w = max(abs(ls.x - rs.x), 1e-6)
    left_dx_ratio = (lfi.x - ls.x) / shoulder_w
    right_dx_ratio = (rfi.x - rs.x) / shoulder_w
    l_lo = -(CFG.foot_index_align_ratio_max + CFG.fi_left_inner_offset_ratio)
    l_hi = +(CFG.foot_index_align_ratio_max + CFG.fi_left_outer_offset_ratio)
    r_lo = -(CFG.foot_index_align_ratio_max + CFG.fi_right_inner_offset_ratio)
    r_hi = +(CFG.foot_index_align_ratio_max + CFG.fi_right_outer_offset_ratio)

    issues = []
    cue_keys = []
    if not (l_lo <= left_dx_ratio <= l_hi):
        issues.append(f"{_view_side('left')} foot index not under {_view_side('left')} shoulder")
        cue_keys.append(_cue_for_view("toe_left_inner" if left_dx_ratio < l_lo else "toe_left_outer"))
    if not (r_lo <= right_dx_ratio <= r_hi):
        issues.append(f"{_view_side('right')} foot index not under {_view_side('right')} shoulder")
        cue_keys.append(_cue_for_view("toe_right_outer" if right_dx_ratio < r_lo else "toe_right_inner"))
    if issues:
        return "FootIdx: " + "; ".join(issues), (0, 0, 255), abs(left_dx_ratio), abs(right_dx_ratio), cue_keys
    return "FootIdx: OK", (0, 255, 0), abs(left_dx_ratio), abs(right_dx_ratio), []


def check_shoulder_knee_vertical(landmarks):
    """Same-side shoulder-knee tolerance check for squat knee tracking."""
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

    shoulder_w = max(abs(ls.x - rs.x), 1e-6)
    left_dx_ratio = (lk.x - ls.x) / shoulder_w
    right_dx_ratio = (rk.x - rs.x) / shoulder_w
    l_lo = -(CFG.knee_align_ratio_max + CFG.kn_left_inner_offset_ratio)
    l_hi = +(CFG.knee_align_ratio_max + CFG.kn_left_outer_offset_ratio)
    r_lo = -(CFG.knee_align_ratio_max + CFG.kn_right_inner_offset_ratio)
    r_hi = +(CFG.knee_align_ratio_max + CFG.kn_right_outer_offset_ratio)

    issues = []
    cue_keys = []
    if not (l_lo <= left_dx_ratio <= l_hi):
        issues.append(f"{_view_side('left')} knee not under {_view_side('left')} shoulder")
        cue_keys.append(_cue_for_view("knee_left_inner" if left_dx_ratio < l_lo else "knee_left_outer"))
    if not (r_lo <= right_dx_ratio <= r_hi):
        issues.append(f"{_view_side('right')} knee not under {_view_side('right')} shoulder")
        cue_keys.append(_cue_for_view("knee_right_outer" if right_dx_ratio < r_lo else "knee_right_inner"))
    if issues:
        return "Knee: " + "; ".join(issues), (0, 0, 255), abs(left_dx_ratio), abs(right_dx_ratio), cue_keys
    return "Knee: OK", (0, 255, 0), abs(left_dx_ratio), abs(right_dx_ratio), []


def check_hip_center_vertical(landmarks):
    """
    Squat cue: draw hip center, then vertical reference down to knee level.
    Alert when mid-knee center drifts too far from that hip vertical line.
    """
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

    shoulder_w = max(abs(ls.x - rs.x), 1e-6)
    hip_x = (lh.x + rh.x) * 0.5
    hip_y = (lh.y + rh.y) * 0.5
    knee_x = (lk.x + rk.x) * 0.5
    knee_y = (lk.y + rk.y) * 0.5
    dx_ratio = abs(knee_x - hip_x) / shoulder_w
    tol = CFG.hip_align_ratio_max + CFG.hip_offset_ratio
    ok = dx_ratio <= tol

    if ok:
        return "Hip: center over knee line OK", (0, 255, 0), dx_ratio, hip_x, hip_y, knee_x, knee_y, shoulder_w, []
    hip_cue = _cue_for_view("hip_left" if knee_x < hip_x else "hip_right")
    return "Hip ALERT: crossed tolerance", (0, 0, 255), dx_ratio, hip_x, hip_y, knee_x, knee_y, shoulder_w, [hip_cue]


def check_torso_front_vertical(landmarks):
    """Front torso guard with horizontal and vertical tolerances."""
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    shoulder_w = max(abs(ls.x - rs.x), 1e-6)
    sh_x = (ls.x + rs.x) * 0.5
    sh_y = (ls.y + rs.y) * 0.5
    hip_x = (lh.x + rh.x) * 0.5
    hip_y = (lh.y + rh.y) * 0.5

    # Horizontal torso drift (side bend proxy in front view).
    dx_ratio = abs(sh_x - hip_x) / shoulder_w
    x_tol = CFG.torso_horizontal_align_ratio_max + CFG.torso_horizontal_offset_ratio
    x_ok = dx_ratio <= x_tol

    # Vertical torso ratio (person-invariant): compare torso vertical component to torso segment length.
    torso_len = max(float(np.hypot(hip_x - sh_x, hip_y - sh_y)), 1e-6)
    v_gap_ratio = max(0.0, min(1.0, abs(hip_y - sh_y) / torso_len))
    v_min = max(0.01, min(0.995, CFG.torso_vertical_gap_min_ratio - CFG.torso_vertical_gap_offset_ratio))
    v_ok = v_gap_ratio >= v_min

    if x_ok and v_ok:
        return "Torso: front bend tolerance OK", (0, 255, 0), dx_ratio, v_gap_ratio, sh_x, sh_y, hip_x, hip_y, shoulder_w, []
    issues = []
    cue_keys = []
    if not x_ok:
        issues.append("horizontal drift")
        cue_keys.append(_cue_for_view("torso_x_left" if sh_x < hip_x else "torso_x_right"))
    if not v_ok:
        issues.append("too much front bend")
        cue_keys.append("torso_bend")
    return (
        "Torso ALERT: " + "; ".join(issues),
        (0, 0, 255),
        dx_ratio,
        v_gap_ratio,
        sh_x,
        sh_y,
        hip_x,
        hip_y,
        shoulder_w,
        cue_keys,
    )


def check_shoulder_level(landmarks):
    """Check if both shoulders stay at similar height (horizontal balance)."""
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_w = max(abs(ls.x - rs.x), 1e-6)
    dy_ratio = abs(ls.y - rs.y) / shoulder_w
    tol = CFG.shoulder_level_align_ratio_max + CFG.shoulder_level_offset_ratio
    ok = dy_ratio <= tol
    if ok:
        return "Shoulder level: OK", (0, 255, 0), dy_ratio, ls.x, ls.y, rs.x, rs.y, shoulder_w, []
    shlvl_cue = _cue_for_view("shoulder_high_left" if ls.y < rs.y else "shoulder_high_right")
    return "Shoulder ALERT: uneven height", (0, 0, 255), dy_ratio, ls.x, ls.y, rs.x, rs.y, shoulder_w, [shlvl_cue]


def skeleton_pose_all_pass(
    left_ratio: float,
    right_ratio: float,
    align_ratio_max: float,
    left_inner_offset_ratio: float,
    left_outer_offset_ratio: float,
    right_inner_offset_ratio: float,
    right_outer_offset_ratio: float,
) -> bool:
    """Generic pass gate for any same-side shoulder-point tolerance set."""
    l_max = align_ratio_max + max(left_inner_offset_ratio, left_outer_offset_ratio)
    r_max = align_ratio_max + max(right_inner_offset_ratio, right_outer_offset_ratio)
    return left_ratio <= l_max and right_ratio <= r_max


# ---------------------------------------------------------------------------
# Squat rep tracker
# ---------------------------------------------------------------------------

class SquatRepTracker:
    """
    Simple, forgiving squat rep counter — hip & knee only.

    Idea (trainer-style counting):
      raw_gap = knee_y - hip_y   (normalised screen coords)
      • When standing upright, hip is above knee  → raw_gap is a large positive number.
      • When squatting, hip drops toward the knee → raw_gap shrinks toward 0.
      • If hip actually TOUCHES or PASSES the knee → raw_gap ≈ 0 or negative (full squat).

    Standing baseline auto-calibrates from the maximum raw_gap ever seen.
    A rep is counted the moment the lifter finishes coming back up, provided
    they dropped at least a little on the way down (partial squats count).

    State machine:
      STANDING  → DOWN   when raw_gap ≤ enter_thresh   (≥ ~25 % descent)
      DOWN      → STANDING when raw_gap ≥ return_thresh (≥ 85 % of standing)
      On STANDING return, increment count and flag partial vs full (hip-near-knee).
    """

    ENTER_FRAC  = 0.40   # drop to 80 % of standing gap  → entered squat phase
    FULL_FRAC   = 0.25   # drop to 25 % of standing gap  → hip near / touching knee
    # Must be > ENTER_FRAC to create real down→up hysteresis; avoids walk/noise counts.
    RETURN_FRAC = 0.92   # rise back to 92 % of standing → rep finished
    MIN_STAND_GAP = 0.05 # normalised units — below this we cannot trust calibration
    MIN_DOWN_SEC = 0.18  # minimum time spent in "down" phase before counting
    MAX_LR_ASYM_FRAC = 0.45  # reject gait-like asymmetry (left/right legs out of sync)
    TOO_DEEP_GAP = -0.02  # raw_gap below this triggers "too deep" voice cue

    def __init__(self) -> None:
        self.count: int = 0
        self._in_squat: bool = False
        self._reached_full: bool = False
        self._too_deep: bool = False
        self._last_count_t: float = -999.0
        self._last_partial_t: float = -999.0

        self._stand_gap: float = 0.0
        self._last_raw_gap: float = 0.0
        self._depth_pct: float = 0.0
        self._down_start_t: float = -999.0
        self._rep_min_gap: float = 999.0
        self._rep_max_lr_asym: float = 0.0
        self.rep_metrics: list[dict] = []
        self.rep_mistake_notes: list[str] = []
        self.total_active_time_sec: float = 0.0
        self._goal_summary_printed: bool = False
        self._fixed_start_line_y: float | None = None
        self._time_start_t: float = -1.0
        self._rep_cue_counts: dict[str, int] = {}

    @property
    def is_calibrated(self) -> bool:
        return self._stand_gap >= self.MIN_STAND_GAP

    @property
    def calibration_pct(self) -> int:
        if self._stand_gap <= 0:
            return 0
        return min(100, int(round(self._stand_gap / self.MIN_STAND_GAP * 100)))

    @property
    def depth_pct(self) -> float:
        return self._depth_pct

    @property
    def too_deep(self) -> bool:
        return self._too_deep

    @property
    def partial_warn(self) -> bool:
        """True for 2 s after a rep ended without reaching full depth."""
        return (cv2.getTickCount() / cv2.getTickFrequency()) - self._last_partial_t < 2.0

    def just_counted(self) -> bool:
        """True for 1 s after a rep was registered (used for flash effect)."""
        return (cv2.getTickCount() / cv2.getTickFrequency()) - self._last_count_t < 1.0

    def all_done(self) -> bool:
        g = CFG.squat_max_reps
        if g <= 0:
            return False
        return self.count >= g

    # kept for overlay compatibility — drawing code still references _full_depth
    @property
    def _full_depth(self) -> bool:
        return self._reached_full

    def _legs_balanced(self, left_gap: float | None, right_gap: float | None) -> bool:
        """
        Squat should be mostly symmetric in left/right hip-knee gap.
        Walking is alternating and usually highly asymmetric.
        """
        if left_gap is None or right_gap is None:
            return True
        denom = max(self._stand_gap, self.MIN_STAND_GAP)
        return (abs(left_gap - right_gap) / denom) <= self.MAX_LR_ASYM_FRAC

    def update(
        self,
        hip_y: float,
        knee_y: float,
        foot_y: float | None = None,
        shoulder_w: float | None = None,
        left_gap: float | None = None,
        right_gap: float | None = None,
    ) -> None:  # noqa: ARG002
        raw_gap = knee_y - hip_y
        self._last_raw_gap = raw_gap
        self._too_deep = raw_gap < self.TOO_DEEP_GAP

        if raw_gap > self._stand_gap:
            self._stand_gap = raw_gap
            # Lock a fixed timing line from standing baseline once calibration improves.
            if self._fixed_start_line_y is None:
                ratio = max(0.05, min(0.95, CFG.squat_time_line_ratio))
                anchor = (CFG.squat_time_line_anchor or "knee").strip().lower()
                if anchor == "foot" and foot_y is not None:
                    base = max(1e-6, foot_y - knee_y)  # knee-to-foot vertical span
                    self._fixed_start_line_y = max(0.0, min(1.0, foot_y - base * ratio))
                else:
                    base = max(1e-6, knee_y - hip_y)   # hip-to-knee vertical span
                    self._fixed_start_line_y = max(0.0, min(1.0, knee_y - base * ratio))

        if not self.is_calibrated:
            self._depth_pct = 0.0
            return

        sg = self._stand_gap
        enter_thresh  = sg * self.ENTER_FRAC
        full_thresh   = sg * self.FULL_FRAC
        return_thresh = sg * self.RETURN_FRAC

        self._depth_pct = max(0.0, min(1.0, 1.0 - raw_gap / sg))
        if left_gap is not None and right_gap is not None:
            self._rep_max_lr_asym = max(
                self._rep_max_lr_asym,
                abs(left_gap - right_gap) / max(sg, self.MIN_STAND_GAP),
            )

        if not self._in_squat:
            if raw_gap <= enter_thresh and self._legs_balanced(left_gap, right_gap):
                self._in_squat = True
                self._reached_full = False
                self._down_start_t = cv2.getTickCount() / cv2.getTickFrequency()
                self._time_start_t = -1.0
                self._rep_min_gap = raw_gap
                self._rep_max_lr_asym = 0.0
                self._rep_cue_counts = {}
        else:
            # Start rep timer when hip crosses the fixed line downward.
            if self._time_start_t < 0 and self._fixed_start_line_y is not None and hip_y >= self._fixed_start_line_y:
                self._time_start_t = cv2.getTickCount() / cv2.getTickFrequency()
            self._rep_min_gap = min(self._rep_min_gap, raw_gap)
            if raw_gap <= full_thresh or raw_gap <= 0.0:
                self._reached_full = True

            if raw_gap >= return_thresh:
                now = cv2.getTickCount() / cv2.getTickFrequency()
                if (now - self._down_start_t) < self.MIN_DOWN_SEC or not self._legs_balanced(left_gap, right_gap):
                    self._in_squat = False
                    self._reached_full = False
                    self._rep_min_gap = 999.0
                    self._rep_max_lr_asym = 0.0
                    self._time_start_t = -1.0
                    return
                g = CFG.squat_max_reps
                if g <= 0 or self.count < g:
                    self.count += 1
                    self._last_count_t = now
                    t0 = self._time_start_t if self._time_start_t >= 0 else self._down_start_t
                    rep_active_sec = max(0.0, now - t0)
                    self.total_active_time_sec += rep_active_sec
                    if not self._reached_full:
                        self._last_partial_t = now
                    rep_min_gap = self._rep_min_gap if self._rep_min_gap < 900 else raw_gap
                    rep_depth_peak = max(0.0, min(1.0, 1.0 - rep_min_gap / max(sg, 1e-6)))
                    rep_data = {
                        "rep_index": self.count,
                        "active_time_sec": round(rep_active_sec, 3),
                        "peak_depth_pct": round(rep_depth_peak, 4),
                        "full_depth": bool(self._reached_full),
                        "too_deep": bool(rep_min_gap < self.TOO_DEEP_GAP),
                        "min_gap": round(rep_min_gap, 5),
                        "left_right_asym": round(self._rep_max_lr_asym, 5),
                    }
                    rep_data.update(self._rep_form_dict(rep_data["active_time_sec"]))
                    self.rep_metrics.append(rep_data)
                    self.rep_mistake_notes.append(self._build_rep_mistake_note(rep_data))
                    self._print_goal_summary_once()
                self._in_squat = False
                self._reached_full = False
                self._rep_min_gap = 999.0
                self._rep_max_lr_asym = 0.0
                self._time_start_t = -1.0
                self._rep_cue_counts = {}

    def reset(self) -> None:
        self.__init__()

    @property
    def fixed_start_line_y(self) -> float | None:
        return self._fixed_start_line_y

    def observe_rep_form_cues(self, cue_keys: list[str]) -> None:
        """
        Accumulate cue frequencies while in squat phase.
        Call once per frame after update().
        """
        if not self._in_squat:
            return
        for key in cue_keys:
            self._rep_cue_counts[key] = self._rep_cue_counts.get(key, 0) + 1

    def _dominant_status(self, prefix: str, pos_key: str, neg_key: str, ok_text: str = "ok") -> str:
        """
        Pick direction label from cue counts for one domain.
        Returns `ok_text` when no cue from this prefix appeared in the rep.
        """
        scoped = {k: v for k, v in self._rep_cue_counts.items() if k.startswith(prefix)}
        if not scoped:
            return ok_text
        best = max(scoped.items(), key=lambda kv: kv[1])[0]
        if best.endswith(pos_key):
            return pos_key
        if best.endswith(neg_key):
            return neg_key
        return "needs_correction"

    def _rep_form_dict(self, active_time_sec: float) -> dict:
        torso_state = "ok"
        if self._rep_cue_counts.get("torso_bend", 0) > 0:
            torso_state = "forward_lean"
        elif self._rep_cue_counts.get("torso_x_left", 0) > 0 or self._rep_cue_counts.get("torso_x_right", 0) > 0:
            torso_state = "side_lean"

        shoulder_state = "ok"
        if self._rep_cue_counts.get("shoulder_high_left", 0) > 0 or self._rep_cue_counts.get("shoulder_high_right", 0) > 0:
            shoulder_state = "lean"

        hip_state = "ok"
        if self._rep_cue_counts.get("hip_left", 0) > 0:
            hip_state = "shift_left"
        elif self._rep_cue_counts.get("hip_right", 0) > 0:
            hip_state = "shift_right"

        # Direction words match your request format.
        knee_state = self._dominant_status("knee_", "outer", "inner", ok_text="ok")
        foot_state = self._dominant_status("toe_", "outer", "inner", ok_text="ok")
        ankle_state = self._dominant_status("ankle_", "outer", "inner", ok_text="ok")

        return {
            "timing_sec": round(active_time_sec, 4),
            "shoulder_lean": shoulder_state,
            "torso": torso_state,
            "knee": knee_state,
            "foot": foot_state,
            "ankle": ankle_state,
            "hip": hip_state,
        }

    def _build_rep_mistake_note(self, rep_data: dict) -> str:
        mistakes: list[str] = []
        if rep_data.get("too_deep", False):
            mistakes.append("too deep")
        if not rep_data.get("full_depth", False):
            mistakes.append("not full depth")
        # "Sliding" proxy: strong side-to-side asymmetry during the rep.
        if rep_data.get("left_right_asym", 0.0) > (self.MAX_LR_ASYM_FRAC * 0.80):
            mistakes.append("sliding/imbalance")
        if not mistakes:
            return "no mistakes"
        return ", ".join(mistakes)

    def _print_goal_summary_once(self) -> None:
        goal = CFG.squat_max_reps
        if goal <= 0 or self.count < goal or self._goal_summary_printed:
            return
        self._goal_summary_printed = True
        print("\n=== Squat Rep Summary ===")
        for rep in self.rep_metrics:
            rep_i = rep.get("rep_index", 0)
            rep_dict = {
                "mistakes": self.rep_mistake_notes[rep_i - 1] if 0 < rep_i <= len(self.rep_mistake_notes) else "n/a",
                "shoulder_lean": rep.get("shoulder_lean", "ok"),
                "torso": rep.get("torso", "ok"),
                "timing_sec": rep.get("timing_sec", 0.0),
                "knee": rep.get("knee", "ok"),
                "foot": rep.get("foot", "ok"),
                "ankle": rep.get("ankle", "ok"),
                "hip": rep.get("hip", "ok"),
                "too_deep": rep.get("too_deep", False),
                "full_depth": rep.get("full_depth", False),
            }
            print(f"Rep {rep_i}: {rep_dict}")
        print("=========================\n")

    def llm_summary_payload(self) -> dict:
        """Compact payload that can be sent to an LLM summariser."""
        selected = []
        for rep in self.rep_metrics:
            item = {}
            if LLM_CFG.include_rep_index:
                item["rep_index"] = rep["rep_index"]
            if LLM_CFG.include_active_time_sec:
                item["active_time_sec"] = rep["active_time_sec"]
            if LLM_CFG.include_peak_depth_pct:
                item["peak_depth_pct"] = rep["peak_depth_pct"]
            if LLM_CFG.include_full_depth:
                item["full_depth"] = rep["full_depth"]
            if LLM_CFG.include_too_deep:
                item["too_deep"] = rep["too_deep"]
            if LLM_CFG.include_min_gap:
                item["min_gap"] = rep["min_gap"]
            if LLM_CFG.include_left_right_asym:
                item["left_right_asym"] = rep["left_right_asym"]
            selected.append(item)
        return {
            "total_reps": self.count,
            "total_active_time_sec": round(self.total_active_time_sec, 3),
            "rep_metrics": selected,
            "export_config": {
                "include_rep_index": LLM_CFG.include_rep_index,
                "include_active_time_sec": LLM_CFG.include_active_time_sec,
                "include_peak_depth_pct": LLM_CFG.include_peak_depth_pct,
                "include_full_depth": LLM_CFG.include_full_depth,
                "include_too_deep": LLM_CFG.include_too_deep,
                "include_min_gap": LLM_CFG.include_min_gap,
                "include_left_right_asym": LLM_CFG.include_left_right_asym,
            },
        }


def draw_shoulder_foot_guides(frame, landmarks) -> None:
    """
    Draw compact tolerance rails for ankles at shoulder x.
    """
    h, w = frame.shape[:2]
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    if min(ls.visibility, rs.visibility, la.visibility, ra.visibility) < 0.2:
        return

    shoulder_w = max(abs(ls.x - rs.x), 1e-6)
    left_dx_ratio = (la.x - ls.x) / shoulder_w
    right_dx_ratio = (ra.x - rs.x) / shoulder_w
    l_in = CFG.shoulder_foot_align_ratio_max + CFG.sf_left_inner_offset_ratio
    l_out = CFG.shoulder_foot_align_ratio_max + CFG.sf_left_outer_offset_ratio
    r_in = CFG.shoulder_foot_align_ratio_max + CFG.sf_right_inner_offset_ratio
    r_out = CFG.shoulder_foot_align_ratio_max + CFG.sf_right_outer_offset_ratio
    left_ok = -l_in <= left_dx_ratio <= l_out
    right_ok = -r_in <= right_dx_ratio <= r_out

    def draw_side(s, a, in_ratio, out_ratio, dx_signed_ratio, ok, inner_key, outer_key):
        sx = int(round(s.x * w))
        ax = int(round(a.x * w))
        ay = int(round(a.y * h))
        sx = max(0, min(w - 1, sx))
        ax = max(0, min(w - 1, ax))
        ay = max(0, min(h - 1, ay))
        half_len = max(4, int(round(CFG.sf_line_half_len_ratio * h)))
        y0 = max(0, ay - half_len)
        y1 = min(h - 1, ay + half_len)
        col = (0, 255, 0) if ok else (0, 0, 255)
        x_lo = max(0, min(w - 1, int(round(sx - in_ratio * shoulder_w * w))))
        x_hi = max(0, min(w - 1, int(round(sx + out_ratio * shoulder_w * w))))
        lo_cross = dx_signed_ratio < -in_ratio
        hi_cross = dx_signed_ratio > out_ratio
        lo_col = _blink_red_yellow(inner_key) if lo_cross else (0, 200, 255)
        hi_col = _blink_red_yellow(outer_key) if hi_cross else (0, 200, 255)
        cv2.line(frame, (x_lo, y0), (x_lo, y1), lo_col, 2, cv2.LINE_AA)
        cv2.line(frame, (x_hi, y0), (x_hi, y1), hi_col, 2, cv2.LINE_AA)
        cv2.line(frame, (sx, y0), (sx, y1), col, 1, cv2.LINE_AA)
        cv2.circle(frame, (sx, ay), 4, col, -1, cv2.LINE_AA)
        cv2.line(frame, (sx, ay), (ax, ay), (120, 180, 230), 2, cv2.LINE_AA)

    draw_side(
        ls, la, l_in, l_out, left_dx_ratio, left_ok,
        _cue_for_view("ankle_left_inner"), _cue_for_view("ankle_left_outer")
    )
    draw_side(
        rs, ra, r_in, r_out, right_dx_ratio, right_ok,
        _cue_for_view("ankle_right_outer"), _cue_for_view("ankle_right_inner")
    )


def draw_foot_index_guides(frame, landmarks) -> None:
    """Draw compact tolerance rails for foot-index points at shoulder x."""
    h, w = frame.shape[:2]
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lfi = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
    rfi = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
    if min(ls.visibility, rs.visibility, lfi.visibility, rfi.visibility) < 0.2:
        return

    shoulder_w = max(abs(ls.x - rs.x), 1e-6)
    left_dx_ratio = (lfi.x - ls.x) / shoulder_w
    right_dx_ratio = (rfi.x - rs.x) / shoulder_w
    l_in = CFG.foot_index_align_ratio_max + CFG.fi_left_inner_offset_ratio
    l_out = CFG.foot_index_align_ratio_max + CFG.fi_left_outer_offset_ratio
    r_in = CFG.foot_index_align_ratio_max + CFG.fi_right_inner_offset_ratio
    r_out = CFG.foot_index_align_ratio_max + CFG.fi_right_outer_offset_ratio
    left_ok = -l_in <= left_dx_ratio <= l_out
    right_ok = -r_in <= right_dx_ratio <= r_out

    def draw_side(s, p, in_ratio, out_ratio, dx_signed_ratio, ok, inner_key, outer_key):
        sx = int(round(s.x * w))
        px = int(round(p.x * w))
        py = int(round(p.y * h))
        sx = max(0, min(w - 1, sx))
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        half_len = max(4, int(round(CFG.fi_line_half_len_ratio * h)))
        y0 = max(0, py - half_len)
        y1 = min(h - 1, py + half_len)
        col = (80, 255, 80) if ok else (0, 80, 255)
        x_lo = max(0, min(w - 1, int(round(sx - in_ratio * shoulder_w * w))))
        x_hi = max(0, min(w - 1, int(round(sx + out_ratio * shoulder_w * w))))
        lo_cross = dx_signed_ratio < -in_ratio
        hi_cross = dx_signed_ratio > out_ratio
        lo_col = _blink_red_yellow(inner_key) if lo_cross else (255, 180, 80)
        hi_col = _blink_red_yellow(outer_key) if hi_cross else (255, 180, 80)
        cv2.line(frame, (x_lo, y0), (x_lo, y1), lo_col, 2, cv2.LINE_AA)
        cv2.line(frame, (x_hi, y0), (x_hi, y1), hi_col, 2, cv2.LINE_AA)
        cv2.line(frame, (sx, y0), (sx, y1), col, 1, cv2.LINE_AA)
        cv2.circle(frame, (sx, py), 4, col, -1, cv2.LINE_AA)
        cv2.line(frame, (sx, py), (px, py), (200, 170, 120), 2, cv2.LINE_AA)

    draw_side(
        ls, lfi, l_in, l_out, left_dx_ratio, left_ok,
        _cue_for_view("toe_left_inner"), _cue_for_view("toe_left_outer")
    )
    draw_side(
        rs, rfi, r_in, r_out, right_dx_ratio, right_ok,
        _cue_for_view("toe_right_outer"), _cue_for_view("toe_right_inner")
    )


def draw_knee_guides(frame, landmarks) -> None:
    """Draw compact tolerance rails for knees at shoulder x."""
    h, w = frame.shape[:2]
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    if min(ls.visibility, rs.visibility, lk.visibility, rk.visibility) < 0.2:
        return

    shoulder_w = max(abs(ls.x - rs.x), 1e-6)
    left_dx_ratio = (lk.x - ls.x) / shoulder_w
    right_dx_ratio = (rk.x - rs.x) / shoulder_w
    l_in = CFG.knee_align_ratio_max + CFG.kn_left_inner_offset_ratio
    l_out = CFG.knee_align_ratio_max + CFG.kn_left_outer_offset_ratio
    r_in = CFG.knee_align_ratio_max + CFG.kn_right_inner_offset_ratio
    r_out = CFG.knee_align_ratio_max + CFG.kn_right_outer_offset_ratio
    left_ok = -l_in <= left_dx_ratio <= l_out
    right_ok = -r_in <= right_dx_ratio <= r_out

    def draw_side(s, p, in_ratio, out_ratio, dx_signed_ratio, ok, inner_key, outer_key):
        sx = int(round(s.x * w))
        px = int(round(p.x * w))
        py = int(round(p.y * h))
        sx = max(0, min(w - 1, sx))
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        half_len = max(4, int(round(CFG.kn_line_half_len_ratio * h)))
        y0 = max(0, py - half_len)
        y1 = min(h - 1, py + half_len)
        col = (100, 255, 200) if ok else (0, 120, 255)
        x_lo = max(0, min(w - 1, int(round(sx - in_ratio * shoulder_w * w))))
        x_hi = max(0, min(w - 1, int(round(sx + out_ratio * shoulder_w * w))))
        lo_cross = dx_signed_ratio < -in_ratio
        hi_cross = dx_signed_ratio > out_ratio
        lo_col = _blink_red_yellow(inner_key) if lo_cross else (255, 120, 120)
        hi_col = _blink_red_yellow(outer_key) if hi_cross else (255, 120, 120)
        cv2.line(frame, (x_lo, y0), (x_lo, y1), lo_col, 2, cv2.LINE_AA)
        cv2.line(frame, (x_hi, y0), (x_hi, y1), hi_col, 2, cv2.LINE_AA)
        cv2.line(frame, (sx, y0), (sx, y1), col, 1, cv2.LINE_AA)
        cv2.circle(frame, (sx, py), 4, col, -1, cv2.LINE_AA)
        cv2.line(frame, (sx, py), (px, py), (180, 170, 220), 2, cv2.LINE_AA)

    draw_side(
        ls, lk, l_in, l_out, left_dx_ratio, left_ok,
        _cue_for_view("knee_left_inner"), _cue_for_view("knee_left_outer")
    )
    draw_side(
        rs, rk, r_in, r_out, right_dx_ratio, right_ok,
        _cue_for_view("knee_right_outer"), _cue_for_view("knee_right_inner")
    )


def draw_hip_center_guides(frame, hip_x, hip_y, knee_x, knee_y, shoulder_w_n, dx_ratio) -> None:
    """Draw hip-center point, knee reference point on hip-vertical, and tolerance rails."""
    h, w = frame.shape[:2]
    xh = int(round(hip_x * w))
    xk = int(round(knee_x * w))
    yh = int(round(hip_y * h))
    yk = int(round(knee_y * h))
    xh = max(0, min(w - 1, xh))
    xk = max(0, min(w - 1, xk))
    yh = max(0, min(h - 1, yh))
    yk = max(0, min(h - 1, yk))

    y0, y1 = (yh, yk) if yh <= yk else (yk, yh)
    tol_px = max(2, int(round((CFG.hip_align_ratio_max + CFG.hip_offset_ratio) * shoulder_w_n * w)))

    x_lo = max(0, min(w - 1, xh - tol_px))
    x_hi = max(0, min(w - 1, xh + tol_px))
    ok = dx_ratio <= (CFG.hip_align_ratio_max + CFG.hip_offset_ratio)
    main_col = (0, 255, 0) if ok else (0, 0, 255)

    dx_signed_ratio = (knee_x - hip_x) / max(shoulder_w_n, 1e-6)
    tol = CFG.hip_align_ratio_max + CFG.hip_offset_ratio
    lo_cross = dx_signed_ratio < -tol
    hi_cross = dx_signed_ratio > tol
    lo_col = _blink_red_yellow(_cue_for_view("hip_left")) if lo_cross else (255, 120, 255)
    hi_col = _blink_red_yellow(_cue_for_view("hip_right")) if hi_cross else (255, 120, 255)

    cv2.line(frame, (xh, y0), (xh, y1), main_col, 2, cv2.LINE_AA)
    cv2.line(frame, (x_lo, y0), (x_lo, y1), lo_col, 2, cv2.LINE_AA)
    cv2.line(frame, (x_hi, y0), (x_hi, y1), hi_col, 2, cv2.LINE_AA)
    cv2.circle(frame, (xh, yh), 6, (255, 255, 255), -1, cv2.LINE_AA)  # hip center point
    cv2.circle(frame, (xh, yk), 5, (255, 200, 0), -1, cv2.LINE_AA)     # knee reference point on vertical
    cv2.circle(frame, (xk, yk), 5, main_col, -1, cv2.LINE_AA)          # actual knee center point
    cv2.line(frame, (xh, yk), (xk, yk), (200, 200, 80), 2, cv2.LINE_AA)

    # Text alert is rendered in the separate Coach Text window.


def draw_torso_front_guides(frame, sh_x, sh_y, hip_x, hip_y, shoulder_w_n, dx_ratio, v_gap_ratio) -> None:
    """Draw torso front reference with horizontal and vertical tolerance cues."""
    h, w = frame.shape[:2]
    sx = int(round(sh_x * w))
    sy = int(round(sh_y * h))
    hx = int(round(hip_x * w))
    hy = int(round(hip_y * h))
    sx = max(0, min(w - 1, sx))
    sy = max(0, min(h - 1, sy))
    hx = max(0, min(w - 1, hx))
    hy = max(0, min(h - 1, hy))

    x_tol = CFG.torso_horizontal_align_ratio_max + CFG.torso_horizontal_offset_ratio
    x_tol_px = max(2, int(round(x_tol * shoulder_w_n * w)))
    v_min = max(0.01, min(0.995, CFG.torso_vertical_gap_min_ratio - CFG.torso_vertical_gap_offset_ratio))
    torso_h_px = max(1, abs(hy - sy))
    # Always keep target between shoulder and hip regardless of person/body size.
    y_target = int(round(hy - v_min * torso_h_px))
    y_target = max(min(sy, hy), min(max(sy, hy), y_target))

    x_ok = dx_ratio <= x_tol
    v_ok = v_gap_ratio >= v_min
    main_col = (0, 255, 0) if (x_ok and v_ok) else (0, 0, 255)

    dx_signed_ratio = (sh_x - hip_x) / max(shoulder_w_n, 1e-6)
    lo_cross = dx_signed_ratio < -x_tol
    hi_cross = dx_signed_ratio > x_tol
    lo_col = _blink_red_yellow(_cue_for_view("torso_x_left")) if lo_cross else (255, 180, 0)
    hi_col = _blink_red_yellow(_cue_for_view("torso_x_right")) if hi_cross else (255, 180, 0)

    # Hip vertical with side tolerance rails.
    cv2.line(frame, (hx, sy), (hx, hy), main_col, 2, cv2.LINE_AA)
    cv2.line(frame, (max(0, hx - x_tol_px), sy), (max(0, hx - x_tol_px), hy), lo_col, 2, cv2.LINE_AA)
    cv2.line(frame, (min(w - 1, hx + x_tol_px), sy), (min(w - 1, hx + x_tol_px), hy), hi_col, 2, cv2.LINE_AA)

    # Vertical-gap minimum target line (shoulder should stay above this).
    v_col = _blink_red_yellow("torso_bend") if not v_ok else (180, 120, 255)
    cv2.line(frame, (max(0, hx - 35), y_target), (min(w - 1, hx + 35), y_target), v_col, 2, cv2.LINE_AA)

    cv2.circle(frame, (hx, hy), 5, (255, 255, 255), -1, cv2.LINE_AA)  # hip center
    cv2.circle(frame, (sx, sy), 5, main_col, -1, cv2.LINE_AA)          # shoulder center
    cv2.line(frame, (hx, hy), (sx, sy), (150, 200, 240), 2, cv2.LINE_AA)


def draw_shoulder_level_guides(frame, ls_x, ls_y, rs_x, rs_y, shoulder_w_n, dy_ratio) -> None:
    """Draw shoulder-level reference with horizontal offset band and blink on touch/cross."""
    h, w = frame.shape[:2]
    lx = int(round(ls_x * w))
    ly = int(round(ls_y * h))
    rx = int(round(rs_x * w))
    ry = int(round(rs_y * h))
    lx = max(0, min(w - 1, lx))
    rx = max(0, min(w - 1, rx))
    ly = max(0, min(h - 1, ly))
    ry = max(0, min(h - 1, ry))

    y_ref = int(round((ly + ry) * 0.5))
    y_ref = max(0, min(h - 1, y_ref))
    tol = CFG.shoulder_level_align_ratio_max + CFG.shoulder_level_offset_ratio
    tol_px = max(1, int(round(tol * shoulder_w_n * h)))
    # Shoulder "line" touch/cross check against offset band.
    y_top = max(0, y_ref - tol_px)
    y_bot = min(h - 1, y_ref + tol_px)
    touch_top = ly <= y_top or ry <= y_top
    touch_bot = ly >= y_bot or ry >= y_bot
    cross = touch_top or touch_bot
    # Which shoulder is higher (lower y value = higher on screen)?
    shlvl_key = _cue_for_view("shoulder_high_left" if ly < ry else "shoulder_high_right")
    ref_col = _blink_red_yellow(shlvl_key) if cross else (120, 200, 255)

    cv2.line(frame, (lx, y_ref), (rx, y_ref), ref_col, 2, cv2.LINE_AA)
    top_col = _blink_red_yellow(shlvl_key) if touch_top else (255, 180, 80)
    bot_col = _blink_red_yellow(shlvl_key) if touch_bot else (255, 180, 80)
    # Horizontal offset lines above/below shoulder reference line.
    cv2.line(frame, (lx, y_top), (rx, y_top), top_col, 2, cv2.LINE_AA)
    cv2.line(frame, (lx, y_bot), (rx, y_bot), bot_col, 2, cv2.LINE_AA)
    pt_col = (0, 255, 0) if not cross else _blink_red_yellow(shlvl_key)
    cv2.circle(frame, (lx, ly), 4, pt_col, -1, cv2.LINE_AA)
    cv2.circle(frame, (rx, ry), 4, pt_col, -1, cv2.LINE_AA)
    cv2.line(frame, (lx, ly), (lx, y_ref), (180, 180, 180), 1, cv2.LINE_AA)
    cv2.line(frame, (rx, ry), (rx, y_ref), (180, 180, 180), 1, cv2.LINE_AA)


def draw_squat_rep_overlay(
    frame,
    squat: SquatRepTracker,
    hip_x: float, hip_y: float,
    knee_x: float, knee_y: float,
    shoulder_w_n: float,
    show_rep_overlay: bool = True,
) -> None:
    """
    Draw:
    • Depth zone lines: partial / full-depth target / too-deep
    • Rep counter + progress dots (top-right)
    • "Too deep" and "Go deeper" warning banners
    • "All done!" banner when target reps reached
    """
    h, w = frame.shape[:2]
    t = cv2.getTickCount() / cv2.getTickFrequency()

    knee_y_px = max(0, min(h - 1, int(round(knee_y * h))))
    hip_y_px  = max(0, min(h - 1, int(round(hip_y * h))))
    cx        = max(0, min(w - 1, int(round(((hip_x + knee_x) * 0.5) * w))))

    # Horizontal span for guide lines (centred on hip-knee mid x)
    lw = max(70, int(round(shoulder_w_n * w * 1.3)))
    x0 = max(0, cx - lw)
    x1 = min(w - 1, cx + lw)

    # Y positions for depth guide lines.
    # ENTER/FULL are relative to calibrated standing gap.
    # TOO_DEEP is an absolute normalised gap threshold (TOO_DEEP_GAP).
    too_deep_y = max(0, min(h - 1, knee_y_px - int(round(squat.TOO_DEEP_GAP * h))))
    if squat.is_calibrated:
        sg_px = int(round(squat._stand_gap * h))  # standing gap in pixels
        partial_y  = max(0, min(h - 1, knee_y_px - int(sg_px * squat.ENTER_FRAC)))
        full_y     = max(0, min(h - 1, knee_y_px - int(sg_px * squat.FULL_FRAC)))
    else:
        # Pre-calibration: just draw a faint placeholder at knee level
        partial_y = full_y = knee_y_px

    # ---- depth guide lines ----
    # Fixed timing line: locked from standing baseline, so it does not move each frame.
    fixed_line_n = squat.fixed_start_line_y
    if fixed_line_n is not None:
        line_y = max(0, min(h - 1, int(round(fixed_line_n * h))))
        line_w = max(60, int(round(shoulder_w_n * w * CFG.squat_between_line_width_ratio)))
        lx0 = max(0, cx - line_w)
        lx1 = min(w - 1, cx + line_w)
        cv2.line(frame, (lx0, line_y), (lx1, line_y), (180, 180, 255), 2, cv2.LINE_AA)
        # Show fixed pixel distance from knee (your requested "note dimensions from knee").
        d_knee_px = max(0, knee_y_px - line_y)
        cv2.putText(
            frame,
            f"start line (fixed) d_knee={d_knee_px}px",
            (max(0, lx0), max(16, line_y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 255),
            1,
            cv2.LINE_AA,
        )

    # Partial line (amber thin — "squat has started")
    cv2.line(frame, (x0, partial_y), (x1, partial_y), (0, 200, 255), 1, cv2.LINE_AA)

    # Full-depth target (cyan → turns green once reached this rep)
    depth_reached_col = (0, 255, 80) if squat._full_depth else (255, 220, 0)
    cv2.line(frame, (x0, full_y), (x1, full_y), depth_reached_col, 2, cv2.LINE_AA)

    # Too-deep warning line (dim blue → blinks red/yellow when crossed)
    too_deep_col = _blink_red_yellow("squat_too_deep") if squat.too_deep else (0, 60, 200)
    cv2.line(frame, (x0, too_deep_y), (x1, too_deep_y), too_deep_col, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        "Too deep",
        (max(0, x1 - 120), max(14, too_deep_y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        too_deep_col,
        1,
        cv2.LINE_AA,
    )

    # Hip-position dot on the guide axis (moves with hip as you squat)
    hip_dot_col = (0, 0, 220) if squat.too_deep else ((0, 255, 80) if squat._full_depth else (0, 220, 255))
    cv2.circle(frame, (cx, hip_y_px), 7, hip_dot_col, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, hip_y_px), 7, (255, 255, 255), 1, cv2.LINE_AA)
    # Vertical connector line
    cv2.line(frame, (cx, min(hip_y_px, knee_y_px)), (cx, max(hip_y_px, knee_y_px)),
             (80, 80, 80), 1, cv2.LINE_AA)

    if not show_rep_overlay:
        return

    # ---- rep counter (top-right) ----
    max_reps = CFG.squat_max_reps
    font = cv2.FONT_HERSHEY_SIMPLEX

    if squat.just_counted():
        rep_col = (0, 255, 100) if int(t * 6) % 2 == 0 else (0, 200, 255)
    elif squat.all_done():
        rep_col = (0, 255, 80)
    else:
        rep_col = (255, 255, 255)

    _, _ = max_reps, rep_col

    if max_reps > 0:
        rep_text = f"Reps: {squat.count} / {max_reps}"
    else:
        rep_text = f"Reps: {squat.count}"
    cv2.putText(frame, rep_text, (14, 34), font, 0.9, rep_col, 2, cv2.LINE_AA)

    # Progress dots
    if max_reps > 0:
        dot_r, dot_sp = 11, 28
        dots_x0 = w - max_reps * dot_sp - 12
        dots_y  = 42
        for i in range(max_reps):
            dx = dots_x0 + i * dot_sp + dot_r
            if i < squat.count:
                cv2.circle(frame, (dx, dots_y), dot_r, (0, 210, 80), -1, cv2.LINE_AA)
                cv2.circle(frame, (dx, dots_y), dot_r, (0, 255, 120), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (dx, dots_y), dot_r, (50, 50, 50), -1, cv2.LINE_AA)
                cv2.circle(frame, (dx, dots_y), dot_r, (110, 110, 110), 1, cv2.LINE_AA)

    # Text for rep count / warnings is rendered in the separate Coach Text window.


def resize_for_display(img):
    """
    Scale image for cv2.imshow to fit inside display_max_width × display_max_height.

    Single scale for both axes (aspect ratio preserved). May downscale or upscale
    from the camera frame depending on CFG and slider limits.
    """
    h, w = img.shape[:2]
    scale = min(CFG.display_max_width / w, CFG.display_max_height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w == w and new_h == h:
        return img
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


if __name__ == "__main__":
    # Input source: file path for cv2.VideoCapture (webcam would be 0).
    input_video = r"D:\AIDS\cv\fitness_assist\inputs\WhatsApp Video 2026-04-12 at 08.59.02.mp4"
    # input_video = r"D:\AIDS\cv\fitness_assist\inputs\WhatsApp Video 2026-04-11 at 12.36.40.mp4"
    cap = cv2.VideoCapture(0)

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _recordings_dir = os.path.join(_script_dir, "recordings")
    os.makedirs(_recordings_dir, exist_ok=True)
    _record_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _record_path = os.path.join(_recordings_dir, f"pose_session_{_record_stamp}_skeleton.mp4")
    _record_raw_path = os.path.join(_recordings_dir, f"pose_session_{_record_stamp}_raw.mp4")
    _record_writer = None
    _record_raw_writer = None
    _record_setup_done = False

    setup_pose_tune_trackbars()
    _ankle_panel = np.zeros((120, 420, 3), dtype=np.uint8)
    _foot_panel = np.zeros((120, 420, 3), dtype=np.uint8)
    _knee_panel = np.zeros((120, 420, 3), dtype=np.uint8)
    _hip_panel = np.zeros((120, 420, 3), dtype=np.uint8)
    _torso_panel = np.zeros((120, 420, 3), dtype=np.uint8)
    _display_panel = np.zeros((120, 420, 3), dtype=np.uint8)
    _coach_text_panel = np.zeros((620, 980, 3), dtype=np.uint8)
    cv2.putText(
        _ankle_panel,
        "ESC quit | R reset all sliders",
        (8, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        _foot_panel,
        "Foot-index tolerance sliders",
        (8, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        _display_panel,
        "Display ratio sliders",
        (8, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        _knee_panel,
        "Knee tolerance sliders",
        (8, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        _hip_panel,
        "Hip center tolerance sliders",
        (8, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        _torso_panel,
        "Torso front tolerance sliders",
        (8, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.namedWindow(COACH_TEXT_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(COACH_TEXT_WINDOW, 980, 620)

    coach = TrainerCoach(cooldown_sec=3.0)
    voice = VoiceCoach(cooldown_sec=5.0) if (_TTS_AVAILABLE and _WINMM_OK) else None
    if voice is None:
        print("Voice coaching unavailable: install edge-tts (pip install edge-tts) and run on Windows.")
    tracker = ViolationTracker()
    squat_tracker = SquatRepTracker()
    _squat_too_deep_coach = TrainerCoach(cooldown_sec=4.0)   # throttled voice for too-deep
    _squat_partial_coach  = TrainerCoach(cooldown_sec=4.0)   # throttled voice for partial rep
    _prev_rep_count = 0
    _rep_milestone_state: dict = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if MIRROR_VIEW:
            frame = cv2.flip(frame, 1)
        raw_frame = frame.copy()
        trainer_msg = None

        if not _record_setup_done:
            _record_setup_done = True
            fh, fw = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 1 or fps > 120:
                fps = 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w = cv2.VideoWriter(_record_path, fourcc, fps, (fw, fh))
            w_raw = cv2.VideoWriter(_record_raw_path, fourcc, fps, (fw, fh))
            if (not w.isOpened()) or (not w_raw.isOpened()):
                w.release()
                w_raw.release()
                _record_path = os.path.join(_recordings_dir, f"pose_session_{_record_stamp}_skeleton.avi")
                _record_raw_path = os.path.join(_recordings_dir, f"pose_session_{_record_stamp}_raw.avi")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                w = cv2.VideoWriter(_record_path, fourcc, fps, (fw, fh))
                w_raw = cv2.VideoWriter(_record_raw_path, fourcc, fps, (fw, fh))
            if w.isOpened() and w_raw.isOpened():
                _record_writer = w
                _record_raw_writer = w_raw
                print(f"Recording skeleton video to: {_record_path}")
                print(f"Recording raw video to: {_record_raw_path}")
            else:
                w.release()
                w_raw.release()
                print("Warning: could not start video recording (codec/path).")

        # Pose expects RGB; OpenCV gives BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        status = "No person detected"

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            sf_label, sf_color, left_ratio, right_ratio, sf_cues = check_shoulder_foot_vertical(landmarks)
            fi_label, fi_color, fi_left_ratio, fi_right_ratio, fi_cues = check_shoulder_foot_index_vertical(landmarks)
            kn_label, kn_color, kn_left_ratio, kn_right_ratio, kn_cues = check_shoulder_knee_vertical(landmarks)
            hip_label, hip_color, hip_dx_ratio, hip_x, hip_y, knee_x, knee_y, shoulder_w_n, hip_cues = (
                check_hip_center_vertical(landmarks)
            )
            torso_label, torso_color, torso_dx_ratio, torso_v_gap_ratio, sh_x, sh_y, thip_x, thip_y, torso_sh_w, torso_cues = (
                check_torso_front_vertical(landmarks)
            )
            shlvl_label, shlvl_color, shlvl_dy_ratio, lsx, lsy, rsx, rsy, shlvl_w, shlvl_cues = check_shoulder_level(landmarks)

            # Update sustain tracker immediately after all checks so _sustained_cues
            # is current before skeleton colour, blinks, status, and voice all read it.
            all_cues = sf_cues + fi_cues + kn_cues + hip_cues + torso_cues + shlvl_cues
            tracker.update(set(all_cues))
            _sustained_cues = tracker.get_sustained(SUSTAIN_SEC)

            skel_ok = skeleton_pose_all_pass(
                left_ratio,
                right_ratio,
                CFG.shoulder_foot_align_ratio_max,
                CFG.sf_left_inner_offset_ratio,
                CFG.sf_left_outer_offset_ratio,
                CFG.sf_right_inner_offset_ratio,
                CFG.sf_right_outer_offset_ratio,
            ) and skeleton_pose_all_pass(
                fi_left_ratio,
                fi_right_ratio,
                CFG.foot_index_align_ratio_max,
                CFG.fi_left_inner_offset_ratio,
                CFG.fi_left_outer_offset_ratio,
                CFG.fi_right_inner_offset_ratio,
                CFG.fi_right_outer_offset_ratio,
            ) and skeleton_pose_all_pass(
                kn_left_ratio,
                kn_right_ratio,
                CFG.knee_align_ratio_max,
                CFG.kn_left_inner_offset_ratio,
                CFG.kn_left_outer_offset_ratio,
                CFG.kn_right_inner_offset_ratio,
                CFG.kn_right_outer_offset_ratio,
            ) and (hip_dx_ratio <= (CFG.hip_align_ratio_max + CFG.hip_offset_ratio)) and (
                torso_dx_ratio <= (CFG.torso_horizontal_align_ratio_max + CFG.torso_horizontal_offset_ratio)
            ) and (
                torso_v_gap_ratio
                >= max(0.01, min(0.995, CFG.torso_vertical_gap_min_ratio - CFG.torso_vertical_gap_offset_ratio))
            ) and (
                shlvl_dy_ratio <= (CFG.shoulder_level_align_ratio_max + CFG.shoulder_level_offset_ratio)
            )
            # Skeleton and status colour follow the sustain gate, not raw per-frame geometry.
            # Green  = all OK (or wobble < 1 s — too short to report).
            # Yellow = violation exists but not yet sustained (dim warning).
            # Red    = confirmed sustained violation (≥ 1 s continuously bad).
            any_raw_violation = not skel_ok
            any_sustained = len(_sustained_cues) > 0
            if any_sustained:
                bone_bgr = (0, 140, 255)     # confirmed bad — orange
            elif any_raw_violation:
                bone_bgr = (0, 200, 255)     # brief wobble — amber/yellow, no alarm yet
            else:
                bone_bgr = (0, 255, 0)       # all good — green
            skel_pt = mp_drawing.DrawingSpec(color=bone_bgr, thickness=2, circle_radius=3)
            skel_ln = mp_drawing.DrawingSpec(color=bone_bgr, thickness=2)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=skel_pt,
                connection_drawing_spec=skel_ln,
            )

            if not any_sustained and not any_raw_violation:
                status = "Squat line: ankle + foot-index + knee + hip + torso OK"
                color = (0, 255, 0)
            elif any_raw_violation and not any_sustained:
                status = "Hold position..." 
                color = (0, 200, 255)        # amber — measuring, not alarming yet
            else:
                if sf_color == (0, 0, 255):
                    status = sf_label
                elif fi_color == (0, 0, 255):
                    status = fi_label
                elif kn_color == (0, 0, 255):
                    status = kn_label
                elif hip_color == (0, 0, 255):
                    status = hip_label
                elif torso_color == (0, 0, 255):
                    status = torso_label
                else:
                    status = shlvl_label
                color = (0, 0, 255)
            draw_shoulder_foot_guides(frame, landmarks)
            draw_foot_index_guides(frame, landmarks)
            draw_knee_guides(frame, landmarks)
            draw_hip_center_guides(frame, hip_x, hip_y, knee_x, knee_y, shoulder_w_n, hip_dx_ratio)
            draw_torso_front_guides(frame, sh_x, sh_y, thip_x, thip_y, torso_sh_w, torso_dx_ratio, torso_v_gap_ratio)
            draw_shoulder_level_guides(frame, lsx, lsy, rsx, rsy, shlvl_w, shlvl_dy_ratio)

            # --- Squat rep tracker ---
            lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
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
            draw_squat_rep_overlay(frame, squat_tracker, hip_x, hip_y, knee_x, knee_y, shoulder_w_n)

            # Keep the live rep reference current every frame so the play
            # thread always speaks the REAL rep count, not a stale one.
            if voice is not None:
                voice.update_rep(squat_tracker.count, CFG.squat_max_reps)

            # Voice for squat events (throttled, separate from alignment cues)
            if squat_tracker.too_deep:
                if _squat_too_deep_coach.get_cue("squat_too_deep") and voice is not None:
                    voice.speak("squat_too_deep")
            if squat_tracker.partial_warn:
                if _squat_partial_coach.get_cue("squat_go_deeper") and voice is not None:
                    voice.speak("squat_go_deeper")

            # Speak only the single highest-priority sustained violation.
            # Latest wins: if a new cue replaces the old one mid-playback,
            # the stale clip finishes (~1 s) and the new one plays immediately.
            for cue_key in all_cues:
                if cue_key in _sustained_cues:
                    trainer_msg = draw_trainer_cue(
                        frame, coach, voice, cue_key,
                    )
                    break

            if voice is not None and CFG.squat_max_reps > 0:
                speak_squat_rep_milestones(
                    voice,
                    _prev_rep_count,
                    squat_tracker.count,
                    CFG.squat_max_reps,
                    _rep_milestone_state,
                )
            _prev_rep_count = squat_tracker.count
        else:
            # No pose detected — reset violation tracker so stale cues don't carry over.
            tracker.update(set())
            _sustained_cues = frozenset()
            color = (0, 0, 255)
            sf_label = fi_label = kn_label = hip_label = torso_label = shlvl_label = "--"
            sf_color = fi_color = kn_color = hip_color = torso_color = shlvl_color = (140, 140, 140)
            left_ratio = right_ratio = fi_left_ratio = fi_right_ratio = kn_left_ratio = kn_right_ratio = 0.0
            hip_dx_ratio = torso_dx_ratio = torso_v_gap_ratio = shlvl_dy_ratio = 0.0

        if _record_raw_writer is not None:
            _record_raw_writer.write(raw_frame)
        if _record_writer is not None:
            _record_writer.write(frame)

        render_coach_text_panel(
            _coach_text_panel,
            status,
            color,
            trainer_msg,
            get_voice_status_text(voice),
            squat_tracker,
            sf_label, sf_color, left_ratio, right_ratio,
            fi_label, fi_color, fi_left_ratio, fi_right_ratio,
            kn_label, kn_color, kn_left_ratio, kn_right_ratio,
            hip_label, hip_color, hip_dx_ratio,
            torso_label, torso_color, torso_dx_ratio, torso_v_gap_ratio,
            shlvl_label, shlvl_color, shlvl_dy_ratio,
        )

        # Preview window uses scaled copy; original frame keeps full resolution for drawing.
        cv2.imshow("Front View Detection", resize_for_display(frame))
        cv2.imshow(COACH_TEXT_WINDOW, _coach_text_panel)
        cv2.imshow(ANKLE_TUNE_WINDOW, _ankle_panel)
        cv2.imshow(FOOT_TUNE_WINDOW, _foot_panel)
        cv2.imshow(KNEE_TUNE_WINDOW, _knee_panel)
        cv2.imshow(HIP_TUNE_WINDOW, _hip_panel)
        cv2.imshow(TORSO_TUNE_WINDOW, _torso_panel)
        cv2.imshow(DISPLAY_TUNE_WINDOW, _display_panel)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord("r"), ord("R")):
            reset_pose_filter_to_defaults()
            coach.reset()
            tracker.reset()
            squat_tracker.reset()
            _squat_too_deep_coach.reset()
            _squat_partial_coach.reset()
            if voice is not None:
                voice.reset()

        count += 1

    if _record_writer is not None:
        _record_writer.release()
    if _record_raw_writer is not None:
        _record_raw_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    if voice is not None:
        voice.shutdown()


