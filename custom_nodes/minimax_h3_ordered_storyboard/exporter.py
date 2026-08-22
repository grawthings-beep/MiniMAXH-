"""Memory-bounded RealESRGAN 2x export for Director segment lists.

The node consumes Director's ``OUTPUT_IS_LIST`` IMAGE/AUDIO outputs as one
list (``INPUT_IS_LIST = True``). It trims each source segment, upscales it in
small frame chunks, streams those chunks into one ffmpeg process per segment,
then stream-concats temporary H.264/PCM Matroska files. No combined 2x IMAGE
tensor is produced or retained.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any


UPSCALE_CHUNK_FRAMES = 16
FFMPEG_STDERR_TAIL_BYTES = 64 * 1024


def _scalar(value: Any, default: Any = None) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return default if value is None else value


def _as_segment_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _unpack_node_output(value: Any) -> Any:
    args = getattr(value, "args", None)
    if args:
        return args[0]
    if isinstance(value, (tuple, list)) and value:
        return value[0]
    raise RuntimeError(f"Unexpected ComfyUI node output: {type(value)!r}")


def _parse_storyboard_plan(raw: Any) -> dict[str, Any] | None:
    raw = _scalar(raw, None)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise ValueError("storyboard_plan is not valid JSON.") from exc
    if not isinstance(data, dict):
        raise ValueError("storyboard_plan must contain a JSON object.")
    return data


def _ffmpeg_path() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg was not found in PATH; Story Export 2x cannot encode MP4.")
    return executable


def _interrupt_check() -> None:
    try:
        from comfy import model_management

        check = getattr(model_management, "throw_exception_if_processing_interrupted", None)
        if check is not None:
            check()
    except ImportError:  # pragma: no cover - only possible outside ComfyUI
        return


def _chunk_ranges(start: int, end: int, chunk_frames: int = UPSCALE_CHUNK_FRAMES):
    """Yield half-open source-frame ranges without retaining chunk tensors."""

    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive.")
    cursor = start
    while cursor < end:
        next_cursor = min(end, cursor + chunk_frames)
        yield cursor, next_cursor
        cursor = next_cursor


def _read_stderr_tail(stderr_file, limit: int = FFMPEG_STDERR_TAIL_BYTES) -> str:
    """Read at most the diagnostic tail, never the whole potentially large log."""

    if limit <= 0:
        return ""
    stderr_file.flush()
    stderr_file.seek(0, os.SEEK_END)
    size = stderr_file.tell()
    offset = max(0, size - limit)
    stderr_file.seek(offset)
    tail = stderr_file.read(limit).decode("utf-8", errors="replace").strip()
    if offset:
        return f"[... {offset} stderr bytes omitted ...]\n{tail}"
    return tail


def _terminate_process(process) -> None:
    """Best-effort shutdown which does not hide the exception being handled."""

    if process is None:
        return
    stdin = getattr(process, "stdin", None)
    if stdin is not None and not getattr(stdin, "closed", False):
        try:
            stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
    try:
        if process.poll() is None:
            process.kill()
        process.wait()
    except (OSError, subprocess.SubprocessError):
        pass


def _write_segment_wave(
    audio: dict[str, Any] | None,
    path: Path,
    *,
    frame_count: int,
    fps: float,
    trim_start_frames: int,
) -> None:
    """Write a stereo PCM WAV exactly long enough for the video segment."""

    import torch

    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    sample_rate = int(audio.get("sample_rate") or 48_000) if isinstance(audio, dict) else 48_000
    if sample_rate <= 0:
        raise ValueError("Audio sample_rate must be positive.")
    wanted = max(1, int(round((frame_count / fps) * sample_rate)))
    start = max(0, int(round((trim_start_frames / fps) * sample_rate)))

    if waveform is None:
        samples = torch.zeros((2, wanted), dtype=torch.float32)
    else:
        if not isinstance(waveform, torch.Tensor):
            raise ValueError("AUDIO waveform must be a torch.Tensor.")
        samples = waveform.detach().float().cpu()
        if samples.ndim == 3:
            samples = samples[0]
        elif samples.ndim == 1:
            samples = samples.unsqueeze(0)
        if samples.ndim != 2:
            raise ValueError(
                f"AUDIO waveform must be [B,C,S] or [C,S], got {tuple(samples.shape)}."
            )
        samples = samples[:, start : start + wanted]
        if samples.shape[0] == 1:
            samples = samples.repeat(2, 1)
        elif samples.shape[0] > 2:
            samples = samples[:2]
        if samples.shape[-1] < wanted:
            pad = torch.zeros((samples.shape[0], wanted - samples.shape[-1]), dtype=samples.dtype)
            samples = torch.cat((samples, pad), dim=-1)

    pcm = (
        samples[:, :wanted]
        .clamp(-1.0, 1.0)
        .mul(32767.0)
        .round()
        .to(torch.int16)
        .transpose(0, 1)
        .contiguous()
        .numpy()
        .tobytes()
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


def _encode_segment(
    frames,
    audio: dict[str, Any] | None,
    path: Path,
    *,
    upscale_model,
    upscale_execute,
    model_management,
    fps: float,
    crf: int,
    preset: str,
    trim_start_frames: int,
    trim_end_frames: int,
    ffmpeg: str,
    chunk_frames: int = UPSCALE_CHUNK_FRAMES,
) -> tuple[int, int, int]:
    """Trim, chunk-upscale, and stream one source segment into one ffmpeg process."""

    import torch

    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        shape = tuple(frames.shape) if hasattr(frames, "shape") else None
        raise ValueError(f"Each IMAGE segment must be [N,H,W,C], got {shape!r}.")
    start = max(0, int(trim_start_frames))
    end = int(frames.shape[0]) - max(0, int(trim_end_frames))
    if end <= start:
        raise ValueError("Boundary trimming removed every frame from a segment.")
    if int(frames.shape[-1]) < 3:
        raise ValueError("Story Export 2x requires RGB IMAGE segments.")
    frame_count = end - start

    audio_path = path.with_suffix(".wav")
    _write_segment_wave(
        audio,
        audio_path,
        frame_count=frame_count,
        fps=fps,
        trim_start_frames=start,
    )
    duration = frame_count / fps
    # Keep temporary audio as PCM. Encoding AAC per segment would add encoder
    # priming/padding at every seam and require a second lossy encode later.
    # The concat step stream-copies H.264 and encodes AAC exactly once.
    # An unread stderr PIPE can fill and deadlock a long rawvideo stream. A
    # temporary file keeps diagnostics available without back-pressure.
    process = None
    width: int | None = None
    height: int | None = None
    return_code: int | None = None
    stderr = ""
    with tempfile.TemporaryFile() as stderr_file:
        try:
            for chunk_start, chunk_end in _chunk_ranges(start, end, chunk_frames):
                _interrupt_check()
                # Slice the trimmed source first, so duplicate boundary frames
                # never enter the upscaler or consume 2x VRAM.
                source_chunk = frames[chunk_start:chunk_end]
                upscaled = None
                try:
                    upscaled = _unpack_node_output(
                        upscale_execute(upscale_model, source_chunk)
                    )
                    if not isinstance(upscaled, torch.Tensor) or upscaled.ndim != 4:
                        shape = tuple(upscaled.shape) if hasattr(upscaled, "shape") else None
                        raise ValueError(
                            f"Upscale model must return [N,H,W,C], got {shape!r}."
                        )
                    expected_chunk_frames = chunk_end - chunk_start
                    actual_chunk_frames = int(upscaled.shape[0])
                    if actual_chunk_frames != expected_chunk_frames:
                        raise ValueError(
                            "Upscale model changed the frame count: expected "
                            f"{expected_chunk_frames}, got {actual_chunk_frames}."
                        )
                    if int(upscaled.shape[-1]) < 3:
                        raise ValueError("Upscale model output must contain RGB channels.")

                    chunk_height = int(upscaled.shape[1])
                    chunk_width = int(upscaled.shape[2])
                    if (
                        chunk_width <= 0
                        or chunk_height <= 0
                        or chunk_width % 2
                        or chunk_height % 2
                    ):
                        raise ValueError(
                            "H.264 export requires positive even dimensions, got "
                            f"{chunk_width}x{chunk_height}."
                        )
                    if width is None:
                        width, height = chunk_width, chunk_height
                        command = [
                            ffmpeg,
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-f",
                            "rawvideo",
                            "-pix_fmt",
                            "rgb24",
                            "-video_size",
                            f"{width}x{height}",
                            "-framerate",
                            f"{fps:.9g}",
                            "-i",
                            "pipe:0",
                            "-i",
                            str(audio_path),
                            "-map",
                            "0:v:0",
                            "-map",
                            "1:a:0",
                            "-c:v",
                            "libx264",
                            "-preset",
                            preset,
                            "-crf",
                            str(crf),
                            "-pix_fmt",
                            "yuv420p",
                            "-c:a",
                            "pcm_s16le",
                            "-ar",
                            "48000",
                            "-ac",
                            "2",
                            "-t",
                            f"{duration:.9f}",
                            str(path),
                        ]
                        process = subprocess.Popen(
                            command,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=stderr_file,
                        )
                    elif (chunk_width, chunk_height) != (width, height):
                        raise ValueError(
                            "Upscale chunks changed canvas size: expected "
                            f"{width}x{height}, got {chunk_width}x{chunk_height}."
                        )

                    assert process is not None and process.stdin is not None
                    for index in range(actual_chunk_frames):
                        _interrupt_check()
                        frame = (
                            upscaled[index, :, :, :3]
                            .detach()
                            .clamp(0.0, 1.0)
                            .mul(255.0)
                            .round()
                            .to(device="cpu", dtype=torch.uint8)
                            .contiguous()
                        )
                        process.stdin.write(frame.numpy().tobytes())
                        del frame
                finally:
                    if upscaled is not None:
                        del upscaled
                    del source_chunk
                    model_management.soft_empty_cache()

            if process is None or width is None or height is None:
                raise RuntimeError("No upscaled frames were produced for the segment.")
            assert process.stdin is not None
            process.stdin.close()
            return_code = process.wait()
            stderr = _read_stderr_tail(stderr_file)
        except BrokenPipeError as exc:
            _terminate_process(process)
            stderr = _read_stderr_tail(stderr_file)
            detail = stderr or str(exc) or "ffmpeg exited before accepting every frame"
            raise RuntimeError(
                f"ffmpeg segment encode closed its input early: {detail}"
            ) from exc
        except BaseException:
            _terminate_process(process)
            raise
    if return_code != 0:
        raise RuntimeError(f"ffmpeg segment encode failed ({return_code}): {stderr.strip()}")
    assert width is not None and height is not None
    return width, height, frame_count


def _concat_segments(ffmpeg: str, segments: list[Path], destination: Path) -> None:
    concat_file = segments[0].parent / "concat.txt"
    # Temporary paths are generated by tempfile and contain no single quote,
    # but escape it anyway for ffmpeg concat-demuxer syntax.
    lines = []
    for path in segments:
        escaped = str(path).replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-af",
        "aresample=async=1:first_pts=0",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg concat failed ({result.returncode}): {result.stderr.strip()}"
        )


class MiniMaxH3StoryExport2x:
    """Upscale and encode each Director segment sequentially, then concatenate."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Connect MiniMaxH3Director.images with export mode set to segments."
                        )
                    },
                ),
                "upscale_model": (
                    "UPSCALE_MODEL",
                    {"tooltip": "RealESRGAN_x2plus is the repository default."},
                ),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "filename_prefix": (
                    "STRING",
                    {"default": "video/MiniMax_H3_Story_2x"},
                ),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1}),
                "preset": (["veryfast", "fast", "medium"], {"default": "fast"}),
                "drop_boundary_duplicates": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Drop the first frame after segment 1 (B at A→B / B→C seams).",
                    },
                ),
                "drop_loop_terminal": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "When storyboard_plan says loop=true, drop the final copy of frame 1."
                        ),
                    },
                ),
            },
            "optional": {
                "audio": (
                    "AUDIO",
                    {"tooltip": "Connect MiniMaxH3Director.audio segment list."},
                ),
                "storyboard_plan": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect Ordered Storyboard.storyboard_plan for loop validation.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "saved_path")
    FUNCTION = "export"
    CATEGORY = "MiniMaxH3/Storyboard"
    DESCRIPTION = (
        "Memory-bounded 2x export: processes one Director IMAGE segment with the upscale "
        "model, immediately encodes it, releases the 2x tensor, and ffmpeg-concats "
        "all segments. INPUT_IS_LIST consumes Director segment lists in one execution."
    )
    INPUT_IS_LIST = True
    OUTPUT_NODE = True

    def export(
        self,
        images,
        upscale_model,
        fps,
        filename_prefix,
        crf,
        preset,
        drop_boundary_duplicates,
        drop_loop_terminal,
        audio=None,
        storyboard_plan=None,
    ):
        import folder_paths
        from comfy import model_management
        from comfy_api.latest import InputImpl, io, ui
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

        segments = _as_segment_list(images)
        if not segments:
            raise ValueError("Story Export 2x received no IMAGE segments.")
        model = _scalar(upscale_model)
        if model is None:
            raise ValueError("Story Export 2x requires an UPSCALE_MODEL.")
        fps_value = float(_scalar(fps, 24.0))
        if not 1.0 <= fps_value <= 120.0:
            raise ValueError("fps must be between 1 and 120.")
        prefix = str(_scalar(filename_prefix, "video/MiniMax_H3_Story_2x"))
        crf_value = int(_scalar(crf, 18))
        preset_value = str(_scalar(preset, "fast"))
        if preset_value not in {"veryfast", "fast", "medium"}:
            raise ValueError("preset must be veryfast, fast, or medium.")
        drop_boundaries = bool(_scalar(drop_boundary_duplicates, True))
        drop_loop = bool(_scalar(drop_loop_terminal, True))

        plan = _parse_storyboard_plan(storyboard_plan)
        if plan is not None:
            expected = int(plan.get("segment_count", len(segments)))
            if expected != len(segments):
                raise ValueError(
                    f"storyboard_plan expects {expected} segments, but Director supplied "
                    f"{len(segments)}. Run all storyboard segments before exporting."
                )
        is_loop = bool(plan and plan.get("loop"))

        audio_segments = _as_segment_list(audio)
        if not audio_segments:
            audio_segments = [None] * len(segments)
        elif len(audio_segments) == 1 and len(segments) > 1:
            audio_segments = audio_segments * len(segments)
        elif len(audio_segments) != len(segments):
            raise ValueError(
                f"Story Export 2x received {len(audio_segments)} audio segments for "
                f"{len(segments)} image segments."
            )

        ffmpeg = _ffmpeg_path()
        temp_root = Path(folder_paths.get_temp_directory())
        temp_root.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="minimax_h3_story_", dir=str(temp_root)))
        encoded: list[Path] = []
        output_partial: Path | None = None
        output_path: Path | None = None
        output_info: tuple[str, str] | None = None
        expected_dimensions: tuple[int, int] | None = None

        try:
            progress = None
            try:
                from comfy.utils import ProgressBar

                progress = ProgressBar(len(segments))
            except ImportError:  # pragma: no cover
                pass

            for index, segment in enumerate(segments):
                _interrupt_check()
                trim_start = 1 if drop_boundaries and index > 0 else 0
                trim_end = 1 if drop_loop and is_loop and index == len(segments) - 1 else 0
                segment_path = run_dir / f"segment_{index:04d}.mkv"
                width, height, _frame_count = _encode_segment(
                    segment,
                    audio_segments[index],
                    segment_path,
                    upscale_model=model,
                    upscale_execute=ImageUpscaleWithModel.execute,
                    model_management=model_management,
                    fps=fps_value,
                    crf=crf_value,
                    preset=preset_value,
                    trim_start_frames=trim_start,
                    trim_end_frames=trim_end,
                    ffmpeg=ffmpeg,
                )
                dimensions = (width, height)
                if expected_dimensions is None:
                    expected_dimensions = dimensions
                elif dimensions != expected_dimensions:
                    raise ValueError(
                        "All upscaled segments must share one canvas; got "
                        f"{expected_dimensions[0]}x{expected_dimensions[1]} then {width}x{height}."
                    )
                encoded.append(segment_path)
                if progress is not None:
                    progress.update(1)

            assert expected_dimensions is not None
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                prefix,
                folder_paths.get_output_directory(),
                expected_dimensions[0],
                expected_dimensions[1],
            )
            os.makedirs(full_output_folder, exist_ok=True)
            output_file = f"{filename}_{counter:05}_.mp4"
            output_path = Path(full_output_folder) / output_file
            output_partial = output_path.with_name(output_path.name + ".partial.mp4")
            _concat_segments(ffmpeg, encoded, output_partial)
            os.replace(output_partial, output_path)
            output_info = (output_file, subfolder)
        finally:
            if output_partial is not None and output_partial.exists():
                output_partial.unlink()
            shutil.rmtree(run_dir, ignore_errors=True)
            model_management.soft_empty_cache()

        assert output_path is not None and output_info is not None
        video = InputImpl.VideoFromFile(str(output_path))
        preview = ui.PreviewVideo(
            [ui.SavedResult(output_info[0], output_info[1], io.FolderType.output)]
        )
        return io.NodeOutput(video, str(output_path), ui=preview)


__all__ = ["MiniMaxH3StoryExport2x"]
