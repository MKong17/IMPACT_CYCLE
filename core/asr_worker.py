import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


_ASR_MODEL_CACHE: Dict[Tuple[str, str, str], object] = {}
_ASR_MODEL_CACHE_LOCK = threading.RLock()


# ========== subprocess helpers ==========
def _run(cmd: list) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    return proc.returncode, proc.stdout, proc.stderr


def _bin(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise RuntimeError(f"{name} not found in PATH.")
    return p


def _path_fingerprint(path: str) -> str:
    norm = os.path.abspath(os.path.expanduser(str(path or "")))
    try:
        st = os.stat(norm)
        payload = f"{norm}|{int(st.st_mtime_ns)}|{int(st.st_size)}"
    except Exception:
        payload = norm
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]


# ========== probe & extract ==========
def probe_audio_stream(video_path: str) -> Tuple[bool, str]:
    """Return (has_audio, log)."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        code, out, err = _run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nk=1:nw=1",
                video_path,
            ]
        )
        has = code == 0 and out.strip() != ""
        return has, (out or err)
    ffmpeg = _bin("ffmpeg")
    code, out, err = _run([ffmpeg, "-hide_banner", "-i", video_path, "-f", "null", "-"])
    txt = out + "\n" + err
    has = ("Audio:" in txt) or ("Stream" in txt and "Audio" in txt)
    return has, txt


def extract_wav_16k_mono_verbose(video_path: str, out_wav: str) -> Tuple[bool, str]:
    """Return (ok, log)."""
    ffmpeg = _bin("ffmpeg")
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-map",
        "a:0?",
        "-f",
        "wav",
        out_wav,
    ]
    code, out, err = _run(cmd)
    return (code == 0), (out or err)


def cached_wav_16k_mono_path(video_path: str) -> str:
    token = _path_fingerprint(video_path)
    return os.path.join(tempfile.gettempdir(), f"_cache_audio_16k_{token}.wav")


def ensure_cached_wav_16k_mono_verbose(
    video_path: str,
) -> Tuple[bool, str, str, bool]:
    out_wav = cached_wav_16k_mono_path(video_path)
    try:
        if os.path.isfile(out_wav) and os.path.getsize(out_wav) > 44:
            return True, out_wav, "Reusing cached extracted audio.", True
    except Exception:
        pass
    ok, log = extract_wav_16k_mono_verbose(video_path, out_wav)
    return ok, out_wav, log, False


# compatibility helper for legacy callers
def extract_wav_16k_mono(video_path: str, out_wav: str) -> bool:
    ok, _ = extract_wav_16k_mono_verbose(video_path, out_wav)
    return ok


def load_cached_faster_whisper_model(
    prefer_model: str,
    compute_type: str = "int8_float16",
):
    key = ("faster_whisper", str(prefer_model or "medium"), str(compute_type or ""))
    with _ASR_MODEL_CACHE_LOCK:
        cached = _ASR_MODEL_CACHE.get(key)
        if cached is not None:
            return cached, True
        from faster_whisper import WhisperModel

        model = WhisperModel(key[1], compute_type=key[2] or "int8_float16")
        _ASR_MODEL_CACHE[key] = model
        return model, False


def load_cached_openai_whisper_model(prefer_model: str):
    key = ("openai_whisper", str(prefer_model or "medium"), "")
    with _ASR_MODEL_CACHE_LOCK:
        cached = _ASR_MODEL_CACHE.get(key)
        if cached is not None:
            return cached, True
        import whisper

        model = whisper.load_model(key[1])
        _ASR_MODEL_CACHE[key] = model
        return model, False


# ========== ASR dataclass ==========
@dataclass
class ASRSegment:
    start: float
    end: float
    text: str
    lang: str


# ========== threading version (compat) ==========
class ASRWorker(threading.Thread):
    def __init__(
        self,
        video_path: str,
        fps: float,
        prefer_model: str = "medium",
        lang: Optional[str] = None,
    ):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.fps = max(1.0, float(fps))
        self.prefer_model = prefer_model
        self.lang = lang  # "en"|"de"|"zh"|None
        self.result: List[ASRSegment] = []
        self.error: Optional[str] = None
        self.tmp_wav: Optional[str] = None
        self.on_progress = None

    def run(self):
        try:
            has, _ = probe_audio_stream(self.video_path)
            if not has:
                raise RuntimeError("This video has no audio track.")
            ok, tmp_wav, elog, reused_audio = ensure_cached_wav_16k_mono_verbose(
                self.video_path
            )
            if not ok:
                raise RuntimeError(
                    f"ffmpeg failed to extract audio:\n{elog.strip()[:800]}"
                )
            self.tmp_wav = tmp_wav
            if self.on_progress:
                self.on_progress(
                    "Reusing cached audio."
                    if reused_audio
                    else "Audio extracted."
                )

            try:
                if self.on_progress:
                    self.on_progress("Preparing faster-whisper model...")
                model, reused_model = load_cached_faster_whisper_model(
                    self.prefer_model,
                    compute_type="int8_float16",
                )
                if self.on_progress:
                    self.on_progress(
                        "Reusing faster-whisper model..."
                        if reused_model
                        else "Loading faster-whisper model..."
                    )
                segments, info = model.transcribe(
                    tmp_wav, language=self.lang, vad_filter=True
                )
                det_lang = (info and getattr(info, "language", None)) or (
                    self.lang or "auto"
                )
                self.result = [
                    ASRSegment(float(s.start), float(s.end), s.text.strip(), det_lang)
                    for s in segments
                ]
            except Exception:
                if self.on_progress:
                    self.on_progress("Preparing openai-whisper model...")
                model, reused_model = load_cached_openai_whisper_model(
                    self.prefer_model
                )
                if self.on_progress:
                    self.on_progress(
                        "Reusing openai-whisper model..."
                        if reused_model
                        else "Loading openai-whisper model..."
                    )
                res = model.transcribe(tmp_wav, language=self.lang, task="transcribe")
                det_lang = res.get("language", self.lang or "auto")
                self.result = [
                    ASRSegment(
                        float(s["start"]), float(s["end"]), s["text"].strip(), det_lang
                    )
                    for s in res.get("segments", [])
                ]
            if self.on_progress:
                self.on_progress(f"ASR done: {len(self.result)} segments.")
        except Exception as ex:
            self.error = str(ex)

    def segments_to_tracks(self) -> Dict[str, List[Tuple[int, int, str]]]:
        buckets: Dict[str, List[Tuple[int, int, str]]] = {}
        for seg in self.result:
            code = (seg.lang or "other").lower()
            if code.startswith("en"):
                k = "en"
            elif code.startswith("de"):
                k = "de"
            elif code.startswith("zh"):
                k = "zh"
            else:
                k = "other"
            s = max(0, int(round(seg.start * self.fps)))
            e = max(s, int(round(seg.end * self.fps)) - 1)
            buckets.setdefault(k, []).append((s, e, seg.text))
        return buckets


# ========== Qt version (progress, non-blocking) ==========
class ASRQtWorker(QObject):
    progress = pyqtSignal(int, str)  # percent (-1=unknown), message
    finished = pyqtSignal(object)  # List[ASRSegment]
    error = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        fps: float,
        prefer_model: str = "medium",
        lang: Optional[str] = None,
    ):
        super().__init__()
        self.video_path = video_path
        self.fps = max(1.0, float(fps))
        self.prefer_model = prefer_model
        self.lang = lang

    @pyqtSlot()
    def run(self):
        try:
            has, _ = probe_audio_stream(self.video_path)
            if not has:
                raise RuntimeError("This video has no audio track.")

            ok, tmp_wav, elog, reused_audio = ensure_cached_wav_16k_mono_verbose(
                self.video_path
            )
            if not ok:
                raise RuntimeError(
                    f"ffmpeg failed to extract audio:\n{elog.strip()[:800]}"
                )
            self.progress.emit(
                -1, "Reusing cached audio." if reused_audio else "Audio extracted."
            )

            # compute duration for progress estimation
            duration = 0.0
            try:
                with wave.open(tmp_wav, "rb") as w:
                    duration = w.getnframes() / float(w.getframerate())
            except Exception:
                duration = 0.0

            # faster-whisper: streaming segments => progress possible
            try:
                self.progress.emit(-1, "Preparing faster-whisper model...")
                model, reused_model = load_cached_faster_whisper_model(
                    self.prefer_model,
                    compute_type="int8_float16",
                )
                self.progress.emit(
                    -1,
                    "Reusing faster-whisper model..."
                    if reused_model
                    else "Loading faster-whisper model...",
                )
                segments, info = model.transcribe(
                    tmp_wav, language=self.lang, vad_filter=True
                )
                det_lang = (info and getattr(info, "language", None)) or (
                    self.lang or "auto"
                )
                out: List[ASRSegment] = []
                last_end = 0.0
                for s in segments:
                    out.append(
                        ASRSegment(
                            float(s.start), float(s.end), s.text.strip(), det_lang
                        )
                    )
                    last_end = float(s.end)
                    if duration > 0:
                        pct = int(min(99, round(100.0 * last_end / duration)))
                        self.progress.emit(
                            pct, f"Transcribing... {last_end:.1f}/{duration:.1f}s"
                        )
                    else:
                        self.progress.emit(-1, "Transcribing...")
                self.progress.emit(100, "Finalizing...")
                self.finished.emit(out)
                return
            except Exception:
                pass

            # fallback to openai-whisper: no incremental progress
            try:
                self.progress.emit(-1, "Preparing openai-whisper model...")
                model, reused_model = load_cached_openai_whisper_model(
                    self.prefer_model
                )
                self.progress.emit(
                    -1,
                    "Reusing openai-whisper model..."
                    if reused_model
                    else "Loading openai-whisper model...",
                )
                res = model.transcribe(tmp_wav, language=self.lang, task="transcribe")
                det_lang = res.get("language", self.lang or "auto")
                out = [
                    ASRSegment(
                        float(s["start"]), float(s["end"]), s["text"].strip(), det_lang
                    )
                    for s in res.get("segments", [])
                ]
                self.progress.emit(100, "Done.")
                self.finished.emit(out)
            except Exception as ex:
                raise RuntimeError(str(ex))
        except Exception as ex:
            self.error.emit(str(ex))
