import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from foxhunter_batch_gui import match_prompts, parse_prompt_list


class PromptListTests(unittest.TestCase):
    def test_one_file_selects_images_in_manifest_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ("scene 2.PNG", "ラピ[01].jpg"):
                (folder / name).touch()
            manifest = "\ufeff[ラピ[01].jpg]\r\n[Shot 1]\r\nLine 1\r\n\r\nLine 2\r\n[scene 2.PNG]\r\nSunrise."
            rows = match_prompts(parse_prompt_list(manifest), [], folder)
            self.assertEqual([r["path"].name for r in rows], ["ラピ[01].jpg","scene 2.PNG"])
            self.assertEqual([r["prompt"] for r in rows], ["[Shot 1]\nLine 1\n\nLine 2","Sunrise."])
            self.assertTrue(all(r["state"] == "待機" for r in rows))
            (folder / "scene 2.PNG").unlink()
            with self.assertRaisesRegex(ValueError, "scene 2"):
                match_prompts(parse_prompt_list(manifest), [], folder)

    def test_preloaded_rows_match_by_name_not_position(self):
        rows = [{"path":Path(n), "state":s,"prompt":"old"} for n,s in
                (("a.png","停止"),("b.jpg","待機"),("c.webp","保存済み"))]
        before = copy.deepcopy(rows)
        result = match_prompts(parse_prompt_list("[b.JPG]\nB\n[A.png]\nA\n[c.webp]\nC"), rows)
        self.assertEqual([r["prompt"] for r in result], ["A", "B", "old"])
        self.assertEqual([r["state"] for r in result], ["停止", "待機", "保存済み"])
        for bad in ("[a.png]\nA", "[d.png]\nextra"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                match_prompts(parse_prompt_list(bad), rows)
        with self.assertRaisesRegex(ValueError,"同名"):
            match_prompts(parse_prompt_list("[a.png]\nA"), [*rows, rows[0]])
        self.assertEqual(rows, before)

    def test_invalid_manifest_never_silently_mispairs(self):
        for source in ("", "no header", "[a.png]\n  ", "[a.png]\n[a.jpg]\nnext",
                       "[a.png]\nA\n[A.PNG]\nB", "[a.png]\nA\n[b.gif]\nB",
                       "[../x.png]\nA", "[C:\\x.png]\nA", "[NUL.png]\nA"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                parse_prompt_list(source)
