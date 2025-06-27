"""
Core processing logic and main entry point for textify.
"""

import logging
import threading
import time
import os
import datetime
import platform
import sys

from .cli import parse_arguments
from . import system  #  ← live module, not frozen flags
from .utils import (
    has_handler_of_type,
    get_eligible_files,
    categorize_files,
    format_time_for_display,
)
from .media import (
    load_whisper_model_with_warning_suppression,
    process_audio_video_files,
)
from .documents import process_document_files


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    if not has_handler_of_type(logger, logging.StreamHandler):
        ch = logging.StreamHandler()
        ch.setFormatter(log_fmt)
        ch.setLevel(logging.WARNING)
        logger.addHandler(ch)

    return logger, log_fmt


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        logger, log_fmt = setup_logging()
        args = parse_arguments()

        # verbose flag
        if args.verbose:
            logger.setLevel(logging.INFO)
            for h in logger.handlers:
                if isinstance(h, logging.StreamHandler) and not isinstance(
                    h, logging.FileHandler
                ):
                    h.setLevel(logging.INFO)

        # -------------------- system checks -------------------- #
        system.initialize_system_checks(verbose=args.verbose)

        if args.device == "cuda" and not system.cuda_available:
            logging.warning(
                "CUDA requested but not available in this PyTorch build. "
                "Falling back to CPU."
            )
            args.device = "cpu"

        # -------------------- log‑file setup ------------------- #
        if os.path.isdir(args.log_file):
            stamp = datetime.datetime.now().strftime("%Y%m%d")
            host = platform.node()
            log_file = os.path.join(args.log_file, f"batch_process_{host}_{stamp}.log")
        else:
            log_file = args.log_file

        new_log_file = not os.path.exists(log_file)

        # single file handler per path
        fh_id = f"file_handler_{log_file}"
        fh = next(
            (h for h in logger.handlers if getattr(h, "_textify_id", "") == fh_id), None
        )
        if fh is None:
            fh = logging.FileHandler(log_file, mode="a")
            fh.setFormatter(log_fmt)
            fh.setLevel(logging.INFO)
            fh._textify_id = fh_id
            logger.addHandler(fh)

        # -------------------- file discovery ------------------- #
        try:
            eligible = (
                get_eligible_files(input_dir=args.input_dir, verbose=args.verbose)
                if args.input_dir
                else get_eligible_files(file_list=args.files, verbose=args.verbose)
            )
        except Exception as e:
            logging.error(f"Failed to scan for files: {e}")
            sys.exit(1)

        if not eligible and not args.watch:
            msg = (
                "No unprocessed audio/video files found in " f"{args.input_dir}."
                if args.input_dir
                else "No unprocessed files found in the provided file list."
            )
            logging.info(msg)
            print(msg, f"\nLog file: {log_file}")
            return
        elif not eligible and args.watch:
            logging.info("No unprocessed files found. Waiting for new files …")
            av_files, doc_files = [], []
        else:
            av_files, doc_files = categorize_files(eligible)

        logging.info(
            f"Found {len(eligible)} unprocessed files "
            f"({len(av_files)} AV, {len(doc_files)} document/image)"
        )

        # -------------------- one‑time system info -------------- #
        if new_log_file:
            if not system.ffprobe_available:
                logging.warning("ffprobe not available – duration estimation disabled.")
            logging.info(f"System: {platform.system()} {platform.release()}")
            logging.info(f"Python: {platform.python_version()}")
            if system.gpu_available:
                for k, v in system.get_gpu_info().items():
                    logging.info(f"GPU {k}: {v}")

        # -------------------- start resource monitor ------------ #
        stop_evt = threading.Event()
        mon_thr = threading.Thread(
            target=system.monitor_resources,
            args=(stop_evt, args.monitoring_interval),
            daemon=True,
        )
        mon_thr.start()

        # -------------------- processing ------------------------ #
        t0 = time.time()
        model = None

        # --- audio / video
        if av_files:
            if (
                args.device == "cuda"
                and system.gpu_available
                and not args.ignore_gpu_threshold
            ):
                try:
                    h = system.pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = system.pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    if util >= args.gpu_threshold:
                        logging.warning(
                            f"GPU util {util}% ≥ threshold "
                            f"{args.gpu_threshold}% – continuing anyway."
                        )
                except Exception as e:
                    logging.debug(f"Could not query GPU utilisation: {e}")

            logging.info("Loading Whisper model …")
            model = load_whisper_model_with_warning_suppression(
                args.model, args.device, args.verbose
            )
            process_audio_video_files(
                av_files,
                model,
                args.language,
                args.gpu_threshold,
                args.device,
                args.ignore_gpu_threshold,
            )

        # --- documents / images
        if doc_files:
            logging.info("Processing document/image files via OCR …")
            process_document_files(doc_files)

        # -------------------- watch mode ------------------------ #
        if args.watch:
            try:
                from watchdog.observers import Observer
                from watchdog.events import FileSystemEventHandler
            except Exception as e:
                logging.error(f"watchdog is required for --watch: {e}")
                stop_evt.set()
                mon_thr.join()
                sys.exit(1)

            watch_path = os.path.abspath(args.input_dir)

            class NewFileHandler(FileSystemEventHandler):
                # -------------------- helper -------------------- #
                @staticmethod
                def _wait_until_complete(path: str,
                                         timeout: float = 30.0,
                                         interval: float = 0.4) -> None:
                    """
                    Wait until a file finishes growing (size stops changing).

                    Args:
                        path (str): File path to monitor.
                        timeout (float): Max seconds to wait.
                        interval (float): Polling interval.
                    """
                    deadline = time.time() + timeout
                    last = -1
                    while time.time() < deadline:
                        try:
                            size = os.path.getsize(path)
                        except OSError:
                            size = -1
                        if size == last and size > 0:
                            return
                        last = size
                        time.sleep(interval)

                # ------------------- core logic ----------------- #
                def _process(self, path: str):
                    try:
                        el = get_eligible_files(file_list=[path], verbose=args.verbose)
                    except Exception as ex:
                        logging.error(f"Failed to check {path}: {ex}")
                        return
                    if not el:
                        return

                    av, docs = categorize_files(el)
                    nonlocal model

                    if av:
                        if model is None:
                            if (
                                args.device == "cuda"
                                and system.gpu_available
                                and not args.ignore_gpu_threshold
                            ):
                                try:
                                    h = system.pynvml.nvmlDeviceGetHandleByIndex(0)
                                    util = system.pynvml.nvmlDeviceGetUtilizationRates(
                                        h
                                    ).gpu
                                    if util >= args.gpu_threshold:
                                        logging.warning(
                                            f"GPU util {util}% ≥ threshold "
                                            f"{args.gpu_threshold}% – continuing anyway."
                                        )
                                except Exception as ex:
                                    logging.debug(
                                        f"Could not query GPU utilisation: {ex}"
                                    )
                            logging.info("Loading Whisper model …")
                            model = load_whisper_model_with_warning_suppression(
                                args.model, args.device, args.verbose
                            )
                        process_audio_video_files(
                            av,
                            model,
                            args.language,
                            args.gpu_threshold,
                            args.device,
                            args.ignore_gpu_threshold,
                        )
                    if docs:
                        process_document_files(docs)

                # --------------- event callbacks --------------- #
                def on_created(self, event):
                    if not event.is_directory:
                        self._wait_until_complete(event.src_path)
                        self._process(event.src_path)

                def on_modified(self, event):
                    if not event.is_directory:
                        self._wait_until_complete(event.src_path)
                        self._process(event.src_path)

                def on_moved(self, event):
                    if not event.is_directory:
                        self._wait_until_complete(event.dest_path)
                        self._process(event.dest_path)

                # Linux (inotify) emits IN_CLOSE_WRITE – keep this too
                def on_closed(self, event):
                    if not event.is_directory:
                        self._process(event.src_path)

            observer = Observer()
            handler = NewFileHandler()
            observer.schedule(handler, watch_path, recursive=False)
            observer.start()
            logging.info(f"Watching {watch_path} for new files …")

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logging.info("Interrupted by user.")
            finally:
                observer.stop()
                observer.join()

        # -------------------- wrap‑up --------------------------- #
        elapsed = time.time() - t0
        if system.psutil_available:
            logging.info(f"Final CPU util: {system.psutil.cpu_percent()}%")
        if system.gpu_available:
            try:
                h = system.pynvml.nvmlDeviceGetHandleByIndex(0)
                util = system.pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                logging.info(f"Final GPU util: {util}%")
            except Exception:
                pass
        logging.info(f"Total batch time: {elapsed:.2f} s")

        stop_evt.set()
        mon_thr.join()
        logging.info(f"Log file: {log_file}")
        print(f"Log file: {log_file}")

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        print("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logging.exception("Unhandled exception")
        print("Error:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
