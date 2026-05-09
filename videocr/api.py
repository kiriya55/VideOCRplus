from __future__ import annotations

import sys

from . import utils
from .plugin_manager import PLUGIN_REGISTRY, check_plugin_status, get_missing_plugins
from .video import Video


def save_subtitles_to_file(
        video_path: str, file_path: str = 'subtitle.srt', ocr_engine: str = 'google_lens', lang: str = 'en',
        time_start: str = '0:00', time_end: str = '', conf_threshold: int = 75, sim_threshold: int = 80, max_merge_gap_sec: float = 0.1,
        use_fullframe: bool = False, use_gpu: bool = False, use_angle_cls: bool = False, use_server_model: bool = False,
        brightness_threshold: int | None = None, ssim_threshold: int = 92, subtitle_position: str = "center", frames_to_skip: int = 1,
        crop_zones: list[dict[str, int]] | None = None, ocr_image_max_width: int = 720, post_processing: bool = False, min_subtitle_duration_sec: float = 0.2,
        normalize_to_simplified_chinese: bool = True, subtitle_alignments: list[str | None] | None = None,
        llm_api_key: str = '', llm_api_base: str = '', llm_model: str = '', llm_concurrency: int = 4,
        llm_disable_inference: bool = False, llm_max_frames_per_grid: int = 0, llm_image_quality: int = 75) -> None:

    if crop_zones is None:
        crop_zones = []

    if subtitle_alignments is None:
        subtitle_alignments = [None, None]
    elif len(subtitle_alignments) == 1:
        subtitle_alignments.append(None)

    # Check required components for the selected engine
    missing = get_missing_plugins(ocr_engine)
    if missing:
        names = [PLUGIN_REGISTRY[k]["display_name"] for k in missing]
        print(f"Error: Missing required components for '{ocr_engine}':", flush=True)
        for name in names:
            print(f"  - {name}", flush=True)
        print("Download them via the GUI Settings tab or from the GitHub releases.", flush=True)
        sys.exit(1)

    # All engines need PaddleOCR executable for Phase 2 text detection
    paddleocr_path = utils.find_executable("paddleocr")
    try:
        utils.perform_hardware_check(paddleocr_path, use_gpu)
    except SystemExit as e:
        print(e, flush=True)
        sys.exit(1)

    if ocr_engine == 'paddleocr':
        det_model_dir, rec_model_dir, cls_model_dir = utils.resolve_model_dirs(lang, use_server_model)
    else:
        # For the Text-Detection-Only Pass just the default detection model is needed
        det_model_dir, rec_model_dir, cls_model_dir = utils.resolve_model_dirs('en', use_server_model)

    google_lens_path = ""
    if ocr_engine == "google_lens":
        google_lens_path = utils.find_executable("chrome-lens")

    v = Video(video_path, paddleocr_path, det_model_dir, rec_model_dir, cls_model_dir, google_lens_path)
    try:
        v.run_ocr(
            use_gpu, ocr_engine, lang, use_angle_cls, time_start, time_end, conf_threshold,
            use_fullframe, brightness_threshold, ssim_threshold, subtitle_position,
            frames_to_skip, crop_zones, ocr_image_max_width, normalize_to_simplified_chinese,
            llm_api_key, llm_api_base, llm_model, llm_concurrency, llm_disable_inference,
            llm_max_frames_per_grid, llm_image_quality
        )
    except Exception as e:
        print(f"Error: {e}", flush=True)
        sys.exit(1)
    subtitles = v.get_subtitles(sim_threshold, max_merge_gap_sec, lang, post_processing, min_subtitle_duration_sec, subtitle_alignments)

    with open(file_path, 'w+', encoding='utf-8') as f:
        f.write(subtitles)
