"""
Media processing module for audio and video files.
"""

import logging
import subprocess
import os
import time
import datetime
import warnings
from typing import List, TYPE_CHECKING

from . import system                # ← live module, no frozen flags

if TYPE_CHECKING:                   # for type‑checkers only
    import whisper


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_media_duration(path: str) -> float:
    if not system.ffprobe_available:
        return 0.0

    res = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if res.returncode != 0:
        logging.warning(f"ffprobe error: {res.stderr.strip()}")
        return 0.0
    try:
        return float(res.stdout.strip())
    except ValueError:
        logging.warning(f"Could not parse duration: {res.stdout}")
        return 0.0


def estimate_processing_time(duration: float) -> float:
    if not (system.gpu_available and system.pynvml_available):
        return 0.0
    try:
        h = system.pynvml.nvmlDeviceGetHandleByIndex(0)
        name = system.pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        if "RTX 4070" in name:
            return 0.1894 * duration + 120.2099
        if "RTX 4060 Ti" in name:
            return 0.3162 * duration + 40.9230
    except Exception:
        pass
    return 0.0


# --------------------------------------------------------------------------- #
# Whisper model
# --------------------------------------------------------------------------- #
def load_whisper_model_with_warning_suppression(
    model_name: str, device: str = "cuda", verbose: bool = False
):
    import whisper

    logging.info(f"Loading Whisper model: {model_name}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning,
                                message=".*weights_only=False.*")
        warnings.filterwarnings("ignore", category=UserWarning)

        model = whisper.load_model(model_name)

        tgt = device if device == "cuda" and system.cuda_available else "cpu"
        if tgt == "cuda":
            logging.debug("Moving model to CUDA")
        model = model.to(tgt)

    if verbose:
        logging.info("Model loaded successfully")
    return model


# --------------------------------------------------------------------------- #
# Batch processor
# --------------------------------------------------------------------------- #
def process_audio_video_files(
    files: List[str],
    model: "whisper.Whisper",
    language: str,
    gpu_threshold: int,
    device: str,
    ignore_gpu_threshold: bool,
):
    if not files:
        logging.info("No audio/video files to process.")
        return

    from .utils import format_time_for_display

    if system.gpu_available and system.pynvml_available:
        h = system.pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_model = system.pynvml.nvmlDeviceGetName(h)
        gpu_model = gpu_model.decode() if isinstance(gpu_model, bytes) else gpu_model
    else:
        gpu_model = "Unknown"

    for fp in files:
        base = os.path.splitext(os.path.basename(fp))[0]
        ext  = os.path.splitext(fp)[1].lower().lstrip(".")
        txt  = os.path.join(os.path.dirname(fp), f"{base}_{ext}.txt")
        dump = os.path.join(os.path.dirname(fp), f"{base}_{ext}_dump.txt")

        duration = get_media_duration(fp) if system.ffprobe_available else 0.0
        est_time = estimate_processing_time(duration)

        logging.info(f"Processing {fp}  (duration {duration:.2f}s)")
        if est_time:
            logging.info(f"Estimated time: {format_time_for_display(est_time)}")

        t0 = time.time()
        start_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(dump, "w", encoding="utf-8") as f:
            f.write(f"Start: {start_stamp}\nDuration: {duration:.2f}s\n"
                    f"Estimated: {format_time_for_display(est_time)}\n\n"
                    "--- Output ---\n\n")

        try:
            fp16 = (device == "cuda" and system.cuda_available)
            out = model.transcribe(fp, language=language, fp16=fp16)
            text = out["text"]
            with open(dump, "a", encoding="utf-8") as f:
                f.write(text)
            with open(txt, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            logging.error(f"Failed on {fp}: {e}")
            with open(dump, "a", encoding="utf-8") as f:
                f.write(f"\nERROR: {e}\n")

        elapsed = time.time() - t0
        with open(dump, "a", encoding="utf-8") as f:
            f.write("\n\n--- Summary ---\n"
                    f"End: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"
                    f"Actual: {format_time_for_display(elapsed)}\n")

        logging.info(f"Finished {fp} in {format_time_for_display(elapsed)}")
