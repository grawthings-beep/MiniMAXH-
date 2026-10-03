import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("guard_under_test", ROOT / "custom_nodes/minimax_h3_ordered_storyboard/swap_guard_nodes.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class GuardContractTests(unittest.TestCase):
    def test_pinned_manifest_matches_runtime_and_all_graphs(self):
        from prepare_character_swap import build
        from verify_character_swap import verify
        asset = json.loads((ROOT / "manifests/swap_guard.json").read_text())["files"][0]
        self.assertEqual(asset["path"], "swap_detection/" + guard.PERSON_MODEL)
        self.assertEqual(asset["sha256"], guard.PERSON_SHA256)
        self.assertEqual(asset["size"], guard.PERSON_BYTES)
        self.assertTrue(asset["verify_sha256"])
        for p in ("official", "dasiwa-v2"):
            for v in ("int8", "x2-detail"):
                m, w = build(p, vae_profile=v)
                verify(w, m)
                self.assertIn(asset, m["files"])
                self.assertEqual(sum(n["type"] == "MiniMaxH3SwapTarget" for n in w["nodes"]), 1)
                self.assertEqual(sum(n["type"] == "MiniMaxH3SwapFinish" for n in w["nodes"]), 1)
                broken = copy.deepcopy(w)
                next(n for n in broken["nodes"] if n["type"] == "MiniMaxH3SwapFinish")["widgets_values"][0] = False
                with self.assertRaises(RuntimeError): verify(broken, m)

    def test_no_claim_of_identity_verification(self):
        from prepare_character_swap import build
        _, w = build("dasiwa-v2")
        self.assertFalse(w["extra"]["character_r2v"]["copy_check_is_identity_check"])
        self.assertFalse(w["extra"]["character_r2v"]["quality_tested_on_gpu"])
        self.assertTrue(w["extra"]["character_r2v"]["mask_review_required"])

    def test_detector_sha_mandatory_even_with_size_verification(self):
        from verify_models import verify_one
        asset = json.loads((ROOT / "manifests/swap_guard.json").read_text())["files"][0]
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "fixture.pt").write_bytes(b"bad")
            status = verify_one(Path(temp), {**asset, "path": "fixture.pt", "size": 3}, "size")
            self.assertIn("sha256 mismatch", status[1])


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU tensor tests need torch")
class GuardTensorTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.frames = torch.rand(22, 64, 96, 3)
        self.mask = torch.zeros(22, 64, 96)
        self.mask[:, 16:48, 16:48] = 1
        self.source = {"frames": self.frames, "length": 22, "height": 64, "width": 96}

    def test_masks_validate_exact_alignment_and_nonempty(self):
        guard.validate_source(self.source)
        guard.validate_mask(self.mask, self.frames)
        for bad in (self.mask[:1], self.mask+2, self.mask*float("nan"), self.mask*0):
            with self.assertRaises(ValueError): guard.validate_mask(bad, self.frames)

    def test_blurring_changes_only_the_marked_pixels_and_original_retention(self):
        original = self.frames.clone()
        changed = guard.suppress_source_identity(self.frames, self.mask, 0, 8)
        self.assertTrue(self.torch.equal(changed[self.mask == 0], self.frames[self.mask == 0]))
        self.assertFalse(self.torch.equal(changed, self.frames))
        self.assertTrue(self.torch.equal(self.frames, original))
        self.assertTrue(self.torch.equal(guard.suppress_source_identity(self.frames, self.mask, 1, 8), original))

    def test_native_temporal_cells_no_ghost_masks_across_full_exit(self):
        with patch.dict(sys.modules, {
            "comfy.ldm.minimax.model": types.SimpleNamespace(FRAME_PER_TOKEN=[1, 4, 4, 4, 4]),
            "comfy_extras.nodes_minimax_h3": types.SimpleNamespace(video_latent_t=lambda n: 2 + (n-5)//17*5),
        }):
            self.mask[5:18] = 0
            grown, latent, pixels = guard.prepare_masks(self.mask, 0)
        self.assertEqual(tuple(latent.shape), (1, 1, 7, 4, 6))
        self.assertEqual(tuple(pixels.shape), (22, 64, 96))
        self.assertFalse(pixels[5:18].any())
        self.assertTrue(pixels[0].any() and pixels[-1].any())
        self.assertTrue(self.torch.equal(grown, self.mask))

    def test_near_copy_and_colour_grading_are_rejected_changed_frames_not_called_identity_pass(self):
        for output in (self.frames, self.frames*.8+.1):
            scores = guard.copy_scores(self.frames, output, self.mask)
            with self.assertRaisesRegex(ValueError, "SWAP NOT CONFIRMED"):
                guard.reject_source_copy(scores)
        scores = guard.copy_scores(self.frames, self.torch.rand_like(self.frames), self.mask)
        self.assertEqual(guard.reject_source_copy(scores), 0.)
        with self.assertRaises(ValueError): guard.reject_source_copy([])

    def test_original_mode_is_exact_passthrough(self):
        result = guard.MiniMaxH3SwapFinish().finish(self.frames, self.source, {"mode": "original"})
        self.assertIs(result[0], self.frames)

    def test_composite_restores_only_background_and_checks_frame_count(self):
        output = self.torch.rand_like(self.frames)
        edit = {"mode": "masked_replace", "shape": tuple(self.frames.shape), "pixel_mask": self.mask, "raw_mask": self.mask}
        result, report = guard.MiniMaxH3SwapFinish().finish(output, self.source, edit)
        self.assertTrue(self.torch.equal(result[self.mask == 0], self.frames[self.mask == 0]))
        self.assertTrue(self.torch.equal(result[self.mask == 1], output[self.mask == 1]))
        self.assertIn("NOT an identity pass", report)
        with self.assertRaises(ValueError): guard.MiniMaxH3SwapFinish().finish(output[:1], self.source, edit)


if __name__ == "__main__":
    unittest.main()
