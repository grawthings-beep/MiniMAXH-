from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "custom_nodes" / "minimax_h3_ordered_storyboard" / "storyboard.py"
)
SPEC = importlib.util.spec_from_file_location("minimax_h3_ordered_storyboard_test", MODULE_PATH)
assert SPEC and SPEC.loader
storyboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(storyboard)

EXPORTER_PATH = (
    ROOT / "custom_nodes" / "minimax_h3_ordered_storyboard" / "exporter.py"
)
EXPORTER_SPEC = importlib.util.spec_from_file_location(
    "minimax_h3_story_exporter_test", EXPORTER_PATH
)
assert EXPORTER_SPEC and EXPORTER_SPEC.loader
exporter = importlib.util.module_from_spec(EXPORTER_SPEC)
EXPORTER_SPEC.loader.exec_module(exporter)


def asset(name: str, *, prompt: str, duration: float, seed: int) -> dict:
    return {
        "name": name,
        "subfolder": "minimax_h3_storyboard",
        "type": "input",
        "transition": {
            "prompt": prompt,
            "duration_sec": duration,
            "seed": seed,
        },
    }


class _FakeFrame:
    def detach(self):
        return self

    def clamp(self, *_args, **_kwargs):
        return self

    def mul(self, *_args, **_kwargs):
        return self

    def round(self):
        return self

    def to(self, *_args, **_kwargs):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self

    def tobytes(self):
        return b"rgb"


class _FakeImageTensor:
    """Small torch.Tensor stand-in used to test batching without allocating pixels."""

    def __init__(self, shape: tuple[int, ...], *, source_offset: int = 0):
        self.shape = shape
        self.ndim = len(shape)
        self.source_offset = source_offset

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = 0 if key.start is None else key.start
            stop = self.shape[0] if key.stop is None else key.stop
            return _FakeImageTensor(
                (stop - start, *self.shape[1:]),
                source_offset=self.source_offset + start,
            )
        if isinstance(key, tuple) and isinstance(key[0], int):
            return _FakeFrame()
        raise AssertionError(f"Unexpected fake tensor index: {key!r}")


class OrderedStoryboardTests(unittest.TestCase):
    def test_input_images_are_bounded_before_tensor_retention(self) -> None:
        self.assertEqual(storyboard.bounded_image_size(1024, 768), (1024, 768))
        self.assertEqual(storyboard.bounded_image_size(3840, 2160), (1536, 864))
        self.assertEqual(storyboard.bounded_image_size(2160, 3840), (864, 1536))
        with self.assertRaisesRegex(storyboard.StoryboardValidationError, "too large"):
            storyboard.bounded_image_size(10_000, 5_000)
        loader_source = inspect.getsource(storyboard._load_image_tensor)
        self.assertIn("bounded_image_size(opened.width, opened.height)", loader_source)
        self.assertLess(loader_source.index("bounded_image_size"), loader_source.index("np.asarray"))

    def test_editor_limits_and_persists_multi_file_uploads(self) -> None:
        editor = (
            ROOT
            / "custom_nodes"
            / "minimax_h3_ordered_storyboard"
            / "web"
            / "ordered_storyboard.js"
        ).read_text(encoding="utf-8")
        self.assertIn("const MAX_IMAGES = 100;", editor)
        self.assertIn("const MAX_TOTAL_DURATION = 90;", editor)
        self.assertIn("totalDuration > MAX_TOTAL_DURATION", editor)
        self.assertIn("state.images.length + files.length > MAX_IMAGES", editor)
        push_at = editor.index("state.images.push")
        sync_at = editor.index("sync();", push_at)
        render_at = editor.index("render();", sync_at)
        self.assertLess(sync_at, render_at)

    def test_non_loop_builds_n_minus_one_director_groups(self) -> None:
        state = storyboard.normalize_storyboard(
            {
                "version": 1,
                "loop": False,
                "images": [
                    asset("01.png", prompt="walk", duration=6.6, seed=11),
                    asset("02.png", prompt="turn", duration=5.0, seed=22),
                    asset("03.png", prompt="unused", duration=4.0, seed=33),
                ],
            },
            require_ready=True,
        )
        groups = storyboard.build_director_groups(
            state, lambda item: f"tensor:{item['name']}"
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["kind"], "fl2v")
        self.assertEqual(groups[0]["first_frame"], "tensor:01.png")
        self.assertEqual(groups[0]["last_frame"], "tensor:02.png")
        self.assertEqual(groups[0]["prompt"], "walk")
        self.assertEqual(groups[0]["duration_sec"], 6.6)
        self.assertEqual(groups[0]["seed"], 11)
        self.assertEqual(groups[1]["first_frame"], "tensor:02.png")
        self.assertEqual(groups[1]["last_frame"], "tensor:03.png")
        self.assertEqual(groups[1]["seed"], 22)

    def test_loop_adds_last_to_first_transition(self) -> None:
        state = storyboard.normalize_storyboard(
            {
                "loop": True,
                "images": [
                    asset("a.jpg", prompt="a to b", duration=3.0, seed=1),
                    asset("b.jpg", prompt="b to a", duration=4.0, seed=2),
                ],
            },
            require_ready=True,
        )
        groups = storyboard.build_director_groups(
            state, lambda item: f"tensor:{item['name']}"
        )
        plan = storyboard.build_plan(state)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[-1]["first_frame"], "tensor:b.jpg")
        self.assertEqual(groups[-1]["last_frame"], "tensor:a.jpg")
        self.assertEqual(groups[-1]["prompt"], "b to a")
        self.assertEqual(plan["segment_count"], 2)
        self.assertEqual(plan["total_duration_sec"], 7.0)
        self.assertEqual(plan["transitions"][-1]["from_index"], 1)
        self.assertEqual(plan["transitions"][-1]["to_index"], 0)

    def test_adjacent_groups_share_one_loaded_boundary_image(self) -> None:
        state = storyboard.normalize_storyboard(
            {
                "images": [
                    asset("a.png", prompt="a-b", duration=3.0, seed=1),
                    asset("b.png", prompt="b-c", duration=3.0, seed=2),
                    asset("c.png", prompt="", duration=3.0, seed=3),
                ]
            },
            require_ready=True,
        )
        loaded: dict[str, object] = {}

        def load(item: dict) -> object:
            return loaded.setdefault(item["name"], object())

        groups = storyboard.build_director_groups(state, load)
        self.assertIs(groups[0]["last_frame"], groups[1]["first_frame"])
        self.assertEqual(len(loaded), 3)

    def test_defaults_and_legacy_arrays_are_normalized(self) -> None:
        state = storyboard.normalize_storyboard(
            json.dumps(
                {
                    "defaults": {"prompt": "default", "duration_sec": 6.6, "seed": 100},
                    "prompts": ["first override"],
                    "durations": [7.5],
                    "seeds": [999],
                    "images": [
                        {"filename": "one.webp"},
                        {"name": "two.webp"},
                    ],
                }
            ),
            require_ready=True,
        )

        self.assertEqual(state["images"][0]["transition"]["prompt"], "first override")
        self.assertEqual(state["images"][0]["transition"]["duration_sec"], 7.5)
        self.assertEqual(state["images"][0]["transition"]["seed"], 999)
        self.assertEqual(state["images"][1]["transition"]["prompt"], "default")
        self.assertEqual(state["images"][1]["transition"]["seed"], 101)

    def test_rejects_unsafe_or_non_input_assets(self) -> None:
        with self.assertRaises(storyboard.StoryboardValidationError):
            storyboard.normalize_storyboard(
                {"images": [{"name": "../secret.png"}, {"name": "ok.png"}]},
                require_ready=True,
            )
        with self.assertRaises(storyboard.StoryboardValidationError):
            storyboard.normalize_storyboard(
                {
                    "images": [
                        {"name": "one.png", "type": "output"},
                        {"name": "two.png"},
                    ]
                },
                require_ready=True,
            )

    def test_execution_requires_two_images_but_editor_state_does_not(self) -> None:
        empty = storyboard.normalize_storyboard(storyboard.DEFAULT_STATE)
        self.assertEqual(empty["images"], [])
        with self.assertRaisesRegex(storyboard.StoryboardValidationError, "at least two"):
            storyboard.normalize_storyboard(storyboard.DEFAULT_STATE, require_ready=True)

    def test_output_contract_matches_director_group_type(self) -> None:
        self.assertEqual(storyboard.MiniMaxH3OrderedStoryboard.RETURN_TYPES[0], "MMX_DIR_GROUP")
        self.assertEqual(
            storyboard.annotated_filename(
                {
                    "name": "frame 01.png",
                    "subfolder": "minimax_h3_storyboard",
                    "type": "input",
                }
            ),
            "minimax_h3_storyboard/frame 01.png [input]",
        )

    def test_duration_is_limited_to_h3_clip_range(self) -> None:
        with self.assertRaisesRegex(storyboard.StoryboardValidationError, "15"):
            storyboard.normalize_storyboard(
                {
                    "images": [
                        asset("one.png", prompt="", duration=15.1, seed=1),
                        asset("two.png", prompt="", duration=6.5, seed=2),
                    ]
                },
                require_ready=True,
            )

    def test_total_duration_is_limited_before_director_allocates_all_segments(self) -> None:
        self.assertEqual(storyboard.minimax_aligned_frame_count(6.5), 158)
        images = [
            asset(f"{index:02d}.png", prompt="", duration=10.0, seed=index)
            for index in range(10)
        ]
        with self.assertRaisesRegex(storyboard.StoryboardValidationError, "90s"):
            storyboard.normalize_storyboard(
                {"loop": False, "images": images},
                require_ready=True,
            )

        images[8]["transition"]["duration_sec"] = 8.5
        state = storyboard.normalize_storyboard(
            {"loop": False, "images": images},
            require_ready=True,
        )
        plan = storyboard.build_plan(state)
        self.assertEqual(plan["total_duration_sec"], 88.5)
        self.assertEqual(plan["aligned_frame_count_24fps"], 2_153)

    def test_exporter_consumes_list_inputs_and_parses_connected_plan(self) -> None:
        self.assertTrue(exporter.MiniMaxH3StoryExport2x.INPUT_IS_LIST)
        self.assertTrue(exporter.MiniMaxH3StoryExport2x.OUTPUT_NODE)
        self.assertEqual(exporter._as_segment_list(("a", "b")), ["a", "b"])
        plan = exporter._parse_storyboard_plan(
            [json.dumps({"segment_count": 2, "loop": True})]
        )
        self.assertEqual(plan, {"segment_count": 2, "loop": True})

    def test_exporter_orchestrates_each_segment_without_combining_tensors(self) -> None:
        calls: list[tuple] = []

        class FakeNodeOutput:
            def __init__(self, *args, ui=None):
                self.args = args
                self.ui = ui

        class FakeUpscaler:
            @classmethod
            def execute(cls, model, image):
                calls.append(("upscale", model, image))
                return FakeNodeOutput(f"2x:{image}")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            temp_dir = root / "temp"
            output_dir = root / "output"
            temp_dir.mkdir()
            output_dir.mkdir()

            folder_paths = types.ModuleType("folder_paths")
            folder_paths.get_temp_directory = lambda: str(temp_dir)
            folder_paths.get_output_directory = lambda: str(output_dir)
            folder_paths.get_save_image_path = lambda *args: (
                str(output_dir),
                "story",
                1,
                "",
                args[0],
            )

            model_management = types.SimpleNamespace(soft_empty_cache=lambda: None)
            comfy = types.ModuleType("comfy")
            comfy.model_management = model_management

            latest = types.ModuleType("comfy_api.latest")
            latest.InputImpl = types.SimpleNamespace(
                VideoFromFile=lambda path: {"video_path": path}
            )
            latest.io = types.SimpleNamespace(
                FolderType=types.SimpleNamespace(output="output"),
                NodeOutput=FakeNodeOutput,
            )
            latest.ui = types.SimpleNamespace(
                SavedResult=lambda filename, subfolder, kind: {
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": kind,
                },
                PreviewVideo=lambda values: {"preview": values},
            )
            comfy_api = types.ModuleType("comfy_api")
            comfy_api.latest = latest

            upscale_module = types.ModuleType("comfy_extras.nodes_upscale_model")
            upscale_module.ImageUpscaleWithModel = FakeUpscaler
            comfy_extras = types.ModuleType("comfy_extras")

            def fake_encode(frames, audio, path, **kwargs):
                # The encoder now owns bounded upscaling; exercise the injected
                # callable so this orchestration test verifies that wiring too.
                kwargs["upscale_execute"](kwargs["upscale_model"], frames)
                calls.append(
                    (
                        "encode",
                        frames,
                        audio,
                        kwargs["trim_start_frames"],
                        kwargs["trim_end_frames"],
                    )
                )
                path.write_bytes(b"segment")
                return (1728, 960, 157)

            def fake_concat(_ffmpeg, paths, destination):
                calls.append(("concat", len(paths)))
                destination.write_bytes(b"final")

            modules = {
                "folder_paths": folder_paths,
                "comfy": comfy,
                "comfy_api": comfy_api,
                "comfy_api.latest": latest,
                "comfy_extras": comfy_extras,
                "comfy_extras.nodes_upscale_model": upscale_module,
            }
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(exporter, "_ffmpeg_path", return_value="ffmpeg"),
                mock.patch.object(exporter, "_encode_segment", side_effect=fake_encode),
                mock.patch.object(exporter, "_concat_segments", side_effect=fake_concat),
            ):
                result = exporter.MiniMaxH3StoryExport2x().export(
                    images=["segment-a", "segment-b"],
                    upscale_model=["realesrgan-x2"],
                    fps=[24.0],
                    filename_prefix=["video/story"],
                    crf=[18],
                    preset=["fast"],
                    drop_boundary_duplicates=[True],
                    drop_loop_terminal=[True],
                    audio=[{"id": "audio-a"}, {"id": "audio-b"}],
                    storyboard_plan=[json.dumps({"segment_count": 2, "loop": True})],
                )

            self.assertEqual(
                [call for call in calls if call[0] == "upscale"],
                [
                    ("upscale", "realesrgan-x2", "segment-a"),
                    ("upscale", "realesrgan-x2", "segment-b"),
                ],
            )
            encodes = [call for call in calls if call[0] == "encode"]
            self.assertEqual(encodes[0][3:], (0, 0))
            self.assertEqual(encodes[1][3:], (1, 1))
            self.assertIn(("concat", 2), calls)
            self.assertTrue(result.args[0]["video_path"].endswith("story_00001_.mp4"))
            self.assertTrue(Path(result.args[1]).is_file())

    def test_encode_segment_trims_then_upscales_16_frame_chunks_into_one_pipe(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = _FakeImageTensor
        fake_torch.uint8 = object()
        upscaled_chunks: list[tuple[int, int]] = []
        empty_cache_calls: list[None] = []
        processes = []

        class FakeNodeOutput:
            def __init__(self, value):
                self.args = (value,)

        class FakeStdin:
            def __init__(self):
                self.closed = False
                self.writes = []

            def write(self, value):
                self.writes.append(value)
                return len(value)

            def close(self):
                self.closed = True

        class FakeProcess:
            def __init__(self, command):
                self.command = command
                self.stdin = FakeStdin()
                self.returncode = None
                self.killed = False

            def poll(self):
                return self.returncode

            def kill(self):
                self.killed = True
                self.returncode = -9

            def wait(self):
                if self.returncode is None:
                    self.returncode = 0
                return self.returncode

        def fake_upscale(_model, source_chunk):
            upscaled_chunks.append((source_chunk.source_offset, source_chunk.shape[0]))
            return FakeNodeOutput(
                _FakeImageTensor((source_chunk.shape[0], 960, 1728, 3))
            )

        def fake_popen(command, **_kwargs):
            process = FakeProcess(command)
            processes.append(process)
            return process

        model_management = types.SimpleNamespace(
            soft_empty_cache=lambda: empty_cache_calls.append(None)
        )
        with tempfile.TemporaryDirectory() as temporary, (
            mock.patch.dict(sys.modules, {"torch": fake_torch})
        ), (
            mock.patch.object(exporter, "_write_segment_wave")
        ), (
            mock.patch.object(exporter, "_interrupt_check")
        ), (
            mock.patch.object(exporter.subprocess, "Popen", side_effect=fake_popen)
        ):
            width, height, frame_count = exporter._encode_segment(
                _FakeImageTensor((35, 480, 864, 3)),
                None,
                Path(temporary) / "segment.mkv",
                upscale_model="2x-model",
                upscale_execute=fake_upscale,
                model_management=model_management,
                fps=24.0,
                crf=18,
                preset="fast",
                trim_start_frames=1,
                trim_end_frames=2,
                ffmpeg="ffmpeg",
            )

        self.assertEqual((width, height, frame_count), (1728, 960, 32))
        self.assertEqual(upscaled_chunks, [(1, 16), (17, 16)])
        self.assertEqual(len(processes), 1)
        self.assertEqual(len(processes[0].stdin.writes), 32)
        self.assertEqual(len(empty_cache_calls), 2)
        video_size_at = processes[0].command.index("-video_size")
        self.assertEqual(processes[0].command[video_size_at + 1], "1728x960")

    def test_encode_segment_validates_each_chunk_and_reports_broken_pipe_tail(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = _FakeImageTensor
        fake_torch.uint8 = object()

        class FakeNodeOutput:
            def __init__(self, value):
                self.args = (value,)

        class BrokenStdin:
            closed = False

            def write(self, _value):
                raise BrokenPipeError("pipe closed")

            def close(self):
                self.closed = True

        class BrokenProcess:
            def __init__(self):
                self.stdin = BrokenStdin()

            def poll(self):
                return 7

            def kill(self):
                raise AssertionError("an already-exited ffmpeg must not be killed")

            def wait(self):
                return 7

        def broken_popen(_command, **kwargs):
            kwargs["stderr"].write(b"ffmpeg encoder exploded")
            kwargs["stderr"].flush()
            return BrokenProcess()

        good_upscale = lambda _model, chunk: FakeNodeOutput(
            _FakeImageTensor((chunk.shape[0], 960, 1728, 3))
        )
        management = types.SimpleNamespace(soft_empty_cache=lambda: None)
        common = {
            "upscale_model": "2x-model",
            "model_management": management,
            "fps": 24.0,
            "crf": 18,
            "preset": "fast",
            "trim_start_frames": 0,
            "trim_end_frames": 0,
            "ffmpeg": "ffmpeg",
        }
        with tempfile.TemporaryDirectory() as temporary, (
            mock.patch.dict(sys.modules, {"torch": fake_torch})
        ), (
            mock.patch.object(exporter, "_write_segment_wave")
        ), (
            mock.patch.object(exporter, "_interrupt_check")
        ):
            destination = Path(temporary) / "segment.mkv"
            wrong_count = lambda _model, chunk: FakeNodeOutput(
                _FakeImageTensor((chunk.shape[0] - 1, 960, 1728, 3))
            )
            with self.assertRaisesRegex(ValueError, "changed the frame count"):
                exporter._encode_segment(
                    _FakeImageTensor((2, 480, 864, 3)),
                    None,
                    destination,
                    upscale_execute=wrong_count,
                    **common,
                )

            sizes = iter([(960, 1728), (962, 1728)])

            def changing_size(_model, chunk):
                chunk_height, chunk_width = next(sizes)
                return FakeNodeOutput(
                    _FakeImageTensor((chunk.shape[0], chunk_height, chunk_width, 3))
                )

            harmless_process = mock.Mock()
            harmless_process.stdin = mock.Mock(closed=False)
            harmless_process.poll.return_value = None
            harmless_process.wait.return_value = -9
            with (
                mock.patch.object(exporter.subprocess, "Popen", return_value=harmless_process),
                self.assertRaisesRegex(ValueError, "changed canvas size"),
            ):
                exporter._encode_segment(
                    _FakeImageTensor((17, 480, 864, 3)),
                    None,
                    destination,
                    upscale_execute=changing_size,
                    **common,
                )

            with (
                mock.patch.object(exporter.subprocess, "Popen", side_effect=broken_popen),
                self.assertRaisesRegex(RuntimeError, "ffmpeg encoder exploded"),
            ):
                exporter._encode_segment(
                    _FakeImageTensor((1, 480, 864, 3)),
                    None,
                    destination,
                    upscale_execute=good_upscale,
                    **common,
                )

    def test_stderr_reader_only_returns_the_last_64_kib(self) -> None:
        marker = b"important diagnostic"
        with tempfile.TemporaryFile() as stderr_file:
            stderr_file.write(b"x" * (exporter.FFMPEG_STDERR_TAIL_BYTES + 1024))
            stderr_file.write(marker)
            result = exporter._read_stderr_tail(stderr_file)

        self.assertIn("stderr bytes omitted", result)
        self.assertTrue(result.endswith(marker.decode()))
        self.assertNotIn("x" * (exporter.FFMPEG_STDERR_TAIL_BYTES + 1), result)

    def test_final_concat_copies_video_but_reencodes_aac_once(self) -> None:
        captured: list[str] = []

        def fake_run(command, **_kwargs):
            captured.extend(command)
            return types.SimpleNamespace(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segments = [root / "a.mp4", root / "b.mp4"]
            with mock.patch.object(exporter.subprocess, "run", side_effect=fake_run):
                exporter._concat_segments("ffmpeg", segments, root / "final.mp4")

        self.assertIn("-c:v", captured)
        self.assertEqual(captured[captured.index("-c:v") + 1], "copy")
        self.assertEqual(captured[captured.index("-c:a") + 1], "aac")
        self.assertIn("aresample=async=1:first_pts=0", captured)

    def test_temporary_segments_use_pcm_and_nonblocking_stderr(self) -> None:
        source = inspect.getsource(exporter._encode_segment)
        self.assertIn('"pcm_s16le"', source)
        self.assertNotIn('stderr=subprocess.PIPE', source)
        self.assertIn("tempfile.TemporaryFile()", source)
        self.assertIn("FFMPEG_STDERR_TAIL_BYTES", inspect.getsource(exporter._read_stderr_tail))


if __name__ == "__main__":
    unittest.main()
