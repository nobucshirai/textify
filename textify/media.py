"""
Media processing module for audio and video files.
"""

import logging
import subprocess
import os
import time
import datetime
import warnings
from typing import List, TYPE_CHECKING, Iterable

from . import system                # ← live module, no frozen flags

if TYPE_CHECKING:                   # for type‑checkers only
    import whisper


WHISPER_EXTRA_OUTPUT_FORMATS = ("vtt", "srt", "tsv", "json")


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
            return 0.089 * duration + 15
        if "RTX 4060 Ti" in name:
            return 0.134 * duration + 10.8
    except Exception:
        pass
    return 0.0


def _get_media_output_stem(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return os.path.join(os.path.dirname(path), f"{base}_{ext}")


def write_additional_whisper_outputs(
    result: dict,
    source_path: str,
    output_formats: Iterable[str],
) -> None:
    requested_formats = [fmt for fmt in output_formats if fmt in WHISPER_EXTRA_OUTPUT_FORMATS]
    if not requested_formats:
        return

    if not isinstance(result, dict) or "segments" not in result:
        logging.warning(
            "Skipping additional Whisper outputs for %s because the result "
            "does not contain segments.",
            source_path,
        )
        return

    try:
        from whisper import utils as whisper_utils
    except ImportError as exc:
        logging.error(
            "Additional Whisper output formats were requested but whisper.utils "
            "could not be imported: %s",
            exc,
        )
        return

    writer_classes = {
        "vtt": whisper_utils.WriteVTT,
        "srt": whisper_utils.WriteSRT,
        "tsv": whisper_utils.WriteTSV,
        "json": whisper_utils.WriteJSON,
    }

    output_stem = _get_media_output_stem(source_path)
    for output_format in requested_formats:
        writer = writer_classes[output_format](os.path.dirname(source_path))
        output_path = f"{output_stem}.{output_format}"
        with open(output_path, "w", encoding="utf-8") as f:
            writer.write_result(result, file=f)


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
    whisper_output_formats: Iterable[str] = (),
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
        output_stem = _get_media_output_stem(fp)
        txt = f"{output_stem}.txt"
        dump = f"{output_stem}_dump.txt"

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
            if isinstance(out, dict) and isinstance(out.get("segments"), list):
                lines = []
                for seg in out["segments"]:
                    t = (seg.get("text") or "").strip()
                    if t:
                        lines.append(t)
                text = "\n".join(lines)
            else:
                text = out.get("text", "") if isinstance(out, dict) else ""

            with open(dump, "a", encoding="utf-8") as f:
                f.write(text)
            with open(txt, "w", encoding="utf-8") as f:
                f.write(text)
            write_additional_whisper_outputs(out, fp, whisper_output_formats)
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
