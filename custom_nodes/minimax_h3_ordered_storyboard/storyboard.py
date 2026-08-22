"""Ordered image-keyframe planner for MiniMax H3.

The node deliberately has no import-time dependency on ComfyUI, Torch, NumPy,
or Pillow.  This keeps ComfyUI discovery resilient and makes the state parser
unit-testable.  Runtime image loading uses libraries already shipped with
ComfyUI; this package does not add a pip dependency.

``i2v_groups`` follows AIMixer/ComfyUI_MiniMaxH3_Director's public
``MMX_DIR_GROUP`` v1 payload.  AIMixer currently applies one Director seed to
all external groups, so every transition seed is also retained in the neutral
JSON plan and as additive group metadata.  An executor that supports per-shot
seeds can consume that metadata without changing saved storyboard state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import posixpath
from pathlib import Path, PurePosixPath
from typing import Any, Callable


SCHEMA = "minimax_h3_ordered_storyboard/v1"
DIRECTOR_GROUP_TYPE = "MMX_DIR_GROUP"
DEFAULT_DURATION_SEC = 6.5
DEFAULT_STATE = json.dumps(
    {
        "version": 1,
        "loop": False,
        "defaults": {
            "prompt": "",
            "duration_sec": DEFAULT_DURATION_SEC,
            "seed": 0,
        },
        "images": [],
    },
    ensure_ascii=False,
    separators=(",", ":"),
)

MAX_IMAGES = 100
MAX_PROMPT_CHARS = 20_000
MAX_STATE_CHARS = 2_000_000
MIN_DURATION_SEC = 0.2
MAX_DURATION_SEC = 15.0
# Director retains its per-segment endpoint plan and generated segment outputs.
# Bound the actual 24fps 17k+5-aligned frame sum, not just requested seconds:
# many short segments each add alignment padding and could otherwise evade the
# nominal 90-second cap before the memory-bounded 2x exporter gets control.
STORY_FRAME_RATE = 24.0
MAX_TOTAL_ALIGNED_FRAMES = 2_160
MAX_TOTAL_DURATION_SEC = MAX_TOTAL_ALIGNED_FRAMES / STORY_FRAME_RATE
MAX_SEED = 2**64 - 1
MAX_SOURCE_PIXELS = 40_000_000
# H3 Base's practical canvases are below this edge. Bound retained fp32 input
# tensors before Director creates its own fitted endpoint copies.
MAX_LOAD_LONG_EDGE = 1536
SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}


class StoryboardValidationError(ValueError):
    """A user-facing storyboard validation failure."""


def _bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, "0", "false", "False", "off", "OFF", "no", "NO", None):
        return False
    if value in (1, "1", "true", "True", "on", "ON", "yes", "YES"):
        return True
    raise StoryboardValidationError(f"{field} must be true or false.")


def _duration(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise StoryboardValidationError(f"{field} must be a number.") from exc
    if not math.isfinite(parsed):
        raise StoryboardValidationError(f"{field} must be finite.")
    if not MIN_DURATION_SEC <= parsed <= MAX_DURATION_SEC:
        raise StoryboardValidationError(
            f"{field} must be between {MIN_DURATION_SEC:g} and {MAX_DURATION_SEC:g} seconds."
        )
    return round(parsed, 4)


def minimax_aligned_frame_count(
    duration_sec: float, frame_rate: float = STORY_FRAME_RATE
) -> int:
    """Match Director's duration-to-frame 17k+5 alignment."""

    frames = max(5, int(round(max(0.1, float(duration_sec)) * frame_rate)))
    return frames + ((5 - (frames % 17)) % 17)


def _seed(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise StoryboardValidationError(f"{field} must be an integer.")
    try:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError
        parsed = int(str(value).strip()) if isinstance(value, str) else int(value)
    except (TypeError, ValueError) as exc:
        raise StoryboardValidationError(f"{field} must be an integer.") from exc
    if not 0 <= parsed <= MAX_SEED:
        raise StoryboardValidationError(f"{field} must be between 0 and {MAX_SEED}.")
    return parsed


def _prompt(value: Any, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if len(value) > MAX_PROMPT_CHARS:
        raise StoryboardValidationError(
            f"{field} is too long (maximum {MAX_PROMPT_CHARS} characters)."
        )
    return value


def _asset_name(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StoryboardValidationError(f"{field} is required.")
    name = value.strip()
    if "\x00" in name or "/" in name or "\\" in name or name in {".", ".."}:
        raise StoryboardValidationError(f"{field} must be a plain file name.")
    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise StoryboardValidationError(f"{field} has an unsupported extension ({allowed}).")
    return name


def _subfolder(value: Any, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise StoryboardValidationError(f"{field} must be text.")
    raw = value.strip().replace("\\", "/").strip("/")
    if not raw:
        return ""
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise StoryboardValidationError(f"{field} contains an unsafe path.")
    if any(":" in part or "\x00" in part for part in path.parts):
        raise StoryboardValidationError(f"{field} contains an unsafe path.")
    return path.as_posix()


def _parse_payload(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise StoryboardValidationError("storyboard_data must be JSON text.")
    if len(raw) > MAX_STATE_CHARS:
        raise StoryboardValidationError("storyboard_data is too large.")
    if not raw.strip():
        return json.loads(DEFAULT_STATE)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StoryboardValidationError(
            f"storyboard_data is invalid JSON at line {exc.lineno}, column {exc.colno}."
        ) from exc
    if not isinstance(payload, dict):
        raise StoryboardValidationError("storyboard_data must contain a JSON object.")
    return payload


def normalize_storyboard(
    raw: str | dict[str, Any], *, require_ready: bool = False
) -> dict[str, Any]:
    """Return a canonical, validated v1 storyboard state.

    Empty and one-image states are accepted for interactive editing.  Execution
    calls this with ``require_ready=True`` and therefore requires two keyframes.
    """

    payload = _parse_payload(raw)
    version = payload.get("version", 1)
    if version not in (1, "1", None):
        raise StoryboardValidationError(f"Unsupported storyboard version: {version!r}.")

    defaults_raw = payload.get("defaults") or {}
    if not isinstance(defaults_raw, dict):
        raise StoryboardValidationError("defaults must be an object.")
    defaults = {
        "prompt": _prompt(
            defaults_raw.get("prompt", payload.get("default_prompt", "")),
            field="defaults.prompt",
        ),
        "duration_sec": _duration(
            defaults_raw.get(
                "duration_sec", payload.get("default_duration_sec", DEFAULT_DURATION_SEC)
            ),
            field="defaults.duration_sec",
        ),
        "seed": _seed(
            defaults_raw.get("seed", payload.get("default_seed", 0)),
            field="defaults.seed",
        ),
    }

    raw_images = payload.get("images", [])
    if not isinstance(raw_images, list):
        raise StoryboardValidationError("images must be an array.")
    if len(raw_images) > MAX_IMAGES:
        raise StoryboardValidationError(f"A maximum of {MAX_IMAGES} images is supported.")

    legacy_prompts = payload.get("prompts") if isinstance(payload.get("prompts"), list) else []
    legacy_durations = (
        payload.get("durations") if isinstance(payload.get("durations"), list) else []
    )
    legacy_seeds = payload.get("seeds") if isinstance(payload.get("seeds"), list) else []

    images: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, raw_image in enumerate(raw_images):
        if isinstance(raw_image, str):
            raw_image = {"name": raw_image}
        if not isinstance(raw_image, dict):
            raise StoryboardValidationError(f"images[{index}] must be an object.")

        image_type = str(raw_image.get("type", "input")).strip().lower()
        if image_type != "input":
            raise StoryboardValidationError(
                f"images[{index}].type must be 'input'; only uploaded input assets are allowed."
            )
        name = _asset_name(
            raw_image.get("name", raw_image.get("filename")),
            field=f"images[{index}].name",
        )
        subfolder = _subfolder(
            raw_image.get("subfolder", ""), field=f"images[{index}].subfolder"
        )

        raw_id = str(raw_image.get("id") or f"keyframe-{index + 1}").strip()
        if not raw_id or len(raw_id) > 128 or "\x00" in raw_id:
            raw_id = f"keyframe-{index + 1}"
        image_id = raw_id
        suffix = 2
        while image_id in used_ids:
            image_id = f"{raw_id}-{suffix}"
            suffix += 1
        used_ids.add(image_id)

        transition_raw = raw_image.get("transition") or {}
        if not isinstance(transition_raw, dict):
            raise StoryboardValidationError(f"images[{index}].transition must be an object.")
        fallback_prompt = legacy_prompts[index] if index < len(legacy_prompts) else defaults["prompt"]
        fallback_duration = (
            legacy_durations[index]
            if index < len(legacy_durations)
            else defaults["duration_sec"]
        )
        fallback_seed = (
            legacy_seeds[index]
            if index < len(legacy_seeds)
            else min(MAX_SEED, defaults["seed"] + index)
        )
        transition = {
            "prompt": _prompt(
                transition_raw.get("prompt", fallback_prompt),
                field=f"images[{index}].transition.prompt",
            ),
            "duration_sec": _duration(
                transition_raw.get("duration_sec", fallback_duration),
                field=f"images[{index}].transition.duration_sec",
            ),
            "seed": _seed(
                transition_raw.get("seed", fallback_seed),
                field=f"images[{index}].transition.seed",
            ),
        }
        images.append(
            {
                "id": image_id,
                "name": name,
                "subfolder": subfolder,
                "type": "input",
                "transition": transition,
            }
        )

    loop = _bool(payload.get("loop", False), field="loop")
    if require_ready and len(images) < 2:
        raise StoryboardValidationError(
            "Add at least two keyframe images before queueing the storyboard."
        )
    transition_sources = images[:-1]
    if loop and len(images) >= 2:
        transition_sources = [*transition_sources, images[-1]]
    total_aligned_frames = sum(
        minimax_aligned_frame_count(image["transition"]["duration_sec"])
        for image in transition_sources
    )
    if require_ready and total_aligned_frames > MAX_TOTAL_ALIGNED_FRAMES:
        aligned_seconds = total_aligned_frames / STORY_FRAME_RATE
        raise StoryboardValidationError(
            f"Storyboard aligns to {aligned_seconds:.2f}s at 24fps; the memory-safe "
            f"maximum is {MAX_TOTAL_DURATION_SEC:g}s. Remove images or shorten transitions."
        )

    return {
        "version": 1,
        "loop": loop,
        "defaults": defaults,
        "images": images,
    }


def annotated_filename(asset: dict[str, Any]) -> str:
    """Convert a canonical asset descriptor to ComfyUI's annotated input name."""

    relative = (
        posixpath.join(asset["subfolder"], asset["name"])
        if asset.get("subfolder")
        else asset["name"]
    )
    return f"{relative} [input]"


def transition_pairs(state: dict[str, Any]) -> list[tuple[int, int]]:
    count = len(state["images"])
    pairs = [(index, index + 1) for index in range(max(0, count - 1))]
    if state.get("loop") and count >= 2:
        pairs.append((count - 1, 0))
    return pairs


def build_plan(state: dict[str, Any]) -> dict[str, Any]:
    """Build a tensor-free plan suitable for persistence or another executor."""

    images = state["images"]
    transitions: list[dict[str, Any]] = []
    for segment_index, (source_index, target_index) in enumerate(transition_pairs(state)):
        source = images[source_index]
        target = images[target_index]
        settings = source["transition"]
        transitions.append(
            {
                "index": segment_index,
                "from_index": source_index,
                "to_index": target_index,
                "from_id": source["id"],
                "to_id": target["id"],
                "first_frame": {
                    key: source[key] for key in ("name", "subfolder", "type")
                },
                "last_frame": {
                    key: target[key] for key in ("name", "subfolder", "type")
                },
                "prompt": settings["prompt"],
                "duration_sec": settings["duration_sec"],
                "seed": settings["seed"],
            }
        )
    aligned_frames = sum(
        minimax_aligned_frame_count(item["duration_sec"])
        for item in transitions
    )
    return {
        "schema": SCHEMA,
        "director_seed_scope": "global",
        "transition_seeds_applied_by_director": False,
        "loop": bool(state.get("loop")),
        "image_count": len(images),
        "segment_count": len(transitions),
        "total_duration_sec": round(
            sum(item["duration_sec"] for item in transitions), 4
        ),
        "aligned_frame_count_24fps": aligned_frames,
        "aligned_duration_sec_24fps": round(aligned_frames / STORY_FRAME_RATE, 4),
        "images": [
            {key: image[key] for key in ("id", "name", "subfolder", "type")}
            for image in images
        ],
        "transitions": transitions,
    }


def build_director_groups(
    state: dict[str, Any], image_loader: Callable[[dict[str, Any]], Any]
) -> list[dict[str, Any]]:
    """Build AIMixer Director-compatible external FL2V groups."""

    loaded = [image_loader(asset) for asset in state["images"]]

    groups: list[dict[str, Any]] = []
    for segment_index, (source_index, target_index) in enumerate(transition_pairs(state)):
        source = state["images"][source_index]
        target = state["images"][target_index]
        settings = source["transition"]
        groups.append(
            {
                "version": 1,
                "family": "i2v",
                "kind": "fl2v",
                "prompt": settings["prompt"],
                "duration_sec": settings["duration_sec"],
                # Share one tensor per uploaded keyframe across adjacent groups.
                # Director treats these inputs as read-only, then fits/clones its
                # own per-segment endpoints. Cloning here too would duplicate
                # every high-resolution image two or three times before sampling.
                "first_frame": loaded[source_index],
                "last_frame": loaded[target_index],
                "ref_images": {},
                "ref_videos": {},
                "ref_video_audios": {},
                "ref_audios": {},
                # Additive metadata: ignored safely by Director releases that use
                # a global seed, available to a per-transition-aware executor.
                "seed": settings["seed"],
                "storyboard_transition": {
                    "schema": SCHEMA,
                    "index": segment_index,
                    "from_id": source["id"],
                    "to_id": target["id"],
                    "seed": settings["seed"],
                },
            }
        )
    return groups


def _folder_paths_module():
    try:
        import folder_paths
    except ImportError as exc:  # pragma: no cover - only possible outside ComfyUI
        raise RuntimeError("ComfyUI folder_paths is unavailable.") from exc
    return folder_paths


def _asset_path(asset: dict[str, Any]) -> str:
    folder_paths = _folder_paths_module()
    annotated = annotated_filename(asset)
    if hasattr(folder_paths, "get_annotated_filepath"):
        path = folder_paths.get_annotated_filepath(annotated)
    else:  # pragma: no cover - compatibility with unusually old ComfyUI forks
        path = os.path.join(folder_paths.get_input_directory(), asset["subfolder"], asset["name"])
    if not path or not os.path.isfile(path):
        raise StoryboardValidationError(f"Uploaded image not found: {annotated}")
    return path


def bounded_image_size(width: int, height: int) -> tuple[int, int]:
    """Validate an upload and return an aspect-preserving in-memory size."""

    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise StoryboardValidationError("Image dimensions must be positive.")
    if width * height > MAX_SOURCE_PIXELS:
        raise StoryboardValidationError(
            f"Image is too large ({width}x{height}); resize it below "
            f"{MAX_SOURCE_PIXELS:,} pixels before uploading."
        )
    longest = max(width, height)
    if longest <= MAX_LOAD_LONG_EDGE:
        return width, height
    scale = MAX_LOAD_LONG_EDGE / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def _validate_assets_exist(state: dict[str, Any]) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - ComfyUI always supplies Pillow
        raise RuntimeError("ComfyUI image runtime (Pillow) is unavailable.") from exc

    for asset in state["images"]:
        path = _asset_path(asset)
        try:
            # Image.open reads the header without decoding the full raster. Reject
            # oversized uploads while ComfyUI is validating the queued prompt.
            with Image.open(path) as opened:
                bounded_image_size(opened.width, opened.height)
        except StoryboardValidationError:
            raise
        except Exception as exc:
            raise StoryboardValidationError(
                f"Could not read image {asset['name']}: {exc}"
            ) from exc


def _load_image_tensor(asset: dict[str, Any]):
    path = _asset_path(asset)
    try:
        import numpy as np
        import torch
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - ComfyUI always supplies these
        raise RuntimeError("ComfyUI image runtime (Pillow, NumPy, Torch) is unavailable.") from exc

    try:
        with Image.open(path) as opened:
            if getattr(opened, "is_animated", False):
                opened.seek(0)
            # Check the header before any full decode, then orient and retain only a
            # bounded endpoint tensor. Director performs its own final canvas fit.
            bounded_image_size(opened.width, opened.height)
            image = ImageOps.exif_transpose(opened)
            target_size = bounded_image_size(image.width, image.height)
            if image.size != target_size:
                resampling = getattr(Image, "Resampling", Image)
                image = image.resize(target_size, resampling.LANCZOS)
            image = image.convert("RGB")
            pixels = np.asarray(image, dtype=np.float32) / 255.0
    except StoryboardValidationError:
        raise
    except Exception as exc:
        raise StoryboardValidationError(f"Could not read image {asset['name']}: {exc}") from exc
    return torch.from_numpy(pixels).unsqueeze(0).contiguous()


def _state_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    for asset in state["images"]:
        try:
            path = _asset_path(asset)
            stat = os.stat(path)
            # Uploads use overwrite=false, so path + size + nanosecond mtime is
            # enough to invalidate the Comfy cache without rereading every
            # multi-megabyte keyframe on each Queue click.
            digest.update(os.fsencode(os.path.abspath(path)))
            digest.update(f":{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
        except (OSError, RuntimeError, StoryboardValidationError):
            digest.update(annotated_filename(asset).encode("utf-8"))
            digest.update(b":missing")
    return digest.hexdigest()


class MiniMaxH3OrderedStoryboard:
    """Create a variable number of ordered FL2V groups from uploaded images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard_data": (
                    "STRING",
                    {
                        "default": DEFAULT_STATE,
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": (
                            "Managed by the ordered storyboard editor. The JSON remains "
                            "embedded in the workflow for portable, variable-length input."
                        ),
                    },
                ),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, storyboard_data=DEFAULT_STATE, **_kwargs):
        try:
            state = normalize_storyboard(storyboard_data, require_ready=True)
            _validate_assets_exist(state)
        except (StoryboardValidationError, RuntimeError) as exc:
            return str(exc)
        return True

    @classmethod
    def IS_CHANGED(cls, storyboard_data=DEFAULT_STATE, **_kwargs):
        try:
            state = normalize_storyboard(storyboard_data, require_ready=False)
            return _state_digest(state)
        except (StoryboardValidationError, RuntimeError):
            return hashlib.sha256(str(storyboard_data).encode("utf-8")).hexdigest()

    RETURN_TYPES = (DIRECTOR_GROUP_TYPE, "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("groups", "storyboard_plan", "segment_count", "total_seconds")
    FUNCTION = "build"
    CATEGORY = "MiniMaxH3/Storyboard"
    DESCRIPTION = (
        "Upload, remove, and reorder any number of keyframes. Builds adjacent FL2V pairs "
        "for MiniMaxH3Director.i2v_groups; loop mode also builds last-to-first. "
        "Per-transition prompts and durations are applied. Transition seeds are saved in "
        "storyboard_plan for future executors; current Director uses its one global seed."
    )

    def build(self, storyboard_data=DEFAULT_STATE):
        state = normalize_storyboard(storyboard_data, require_ready=True)
        groups = build_director_groups(state, _load_image_tensor)
        plan = build_plan(state)
        return (
            groups,
            json.dumps(plan, ensure_ascii=False, separators=(",", ":")),
            plan["segment_count"],
            plan["aligned_duration_sec_24fps"],
        )


__all__ = [
    "DEFAULT_STATE",
    "DIRECTOR_GROUP_TYPE",
    "MiniMaxH3OrderedStoryboard",
    "MAX_TOTAL_DURATION_SEC",
    "SCHEMA",
    "StoryboardValidationError",
    "annotated_filename",
    "bounded_image_size",
    "build_director_groups",
    "build_plan",
    "normalize_storyboard",
    "minimax_aligned_frame_count",
    "transition_pairs",
]
