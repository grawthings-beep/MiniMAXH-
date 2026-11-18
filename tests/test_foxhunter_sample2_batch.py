"""Offline contracts, including the two-row HTTP sequence (no GPU needed)."""
import asyncio
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from foxhunter_batch_gui import build_prompt, run_batch, upload


def exported_fixture():
    # Pull node classes and nested execution IDs from the shipped canvas. This
    # is a focused API-format fixture, not a recording or GPU execution.
    wf = json.loads((ROOT / "workflows/foxhunter_seed_hunter_v3.json").read_text(encoding="utf-8"))
    graphs = {g["id"]: g for g in wf["definitions"]["subgraphs"]}
    ns = {str(n["id"]): n for n in wf["nodes"]}
    nested = {f"{k}:{n['id']}": n for k, sg in ns.items() if sg["type"] in graphs
              for n in graphs[sg["type"]]["nodes"]}
    ns.update(nested)
    api = {k: {"class_type": n["type"], "inputs": {}} for k, n in ns.items()}
    api["5"]["inputs"].update({"ref_images.ref_image_1": ["417", 1], "ref_videos.ref_video_0": ["417", 9],
                               "prompt": ["365", 0], "vae": ["171:2", 0], "clip": ["171:3", 0]})
    api["604"]["inputs"].update(samples=["600", 0], model=["605", 0])
    api["600"]["inputs"].update(input_1=["125:12", 1], input_2=["133:129", 1], input_3=["143:139", 1])
    api["133:129"]["inputs"].update(noise=["133:128", 0])
    api["34"]["inputs"].update(save_output=False)
    del api["459"]  # standard (RefMods bypassed) export
    return api


class BatchTests(unittest.TestCase):
    def test_only_sample2_to_upscaled_av_and_file(self):
        template = exported_fixture()
        graph = build_prompt(template, "pics/two.png", "row two", "out/two")
        self.assertEqual({k for k,v in graph.items() if v["class_type"] == "SamplerCustomAdvanced"}, {"133:129", "135:26"})
        self.assertEqual(graph["604"]["inputs"]["samples"], ["133:129", 1])
        self.assertEqual(graph["135:26"]["inputs"]["latent_image"], ["604", 0])
        for name, vae in (("images", "171:2"), ("audio", "171:1")):
            key, _ = graph["34"]["inputs"][name]
            self.assertEqual(graph[key]["inputs"], {"samples":["135:26",0], "vae":[vae,0]})
        self.assertTrue(graph["34"]["inputs"]["save_output"])
        self.assertEqual(graph["900001"]["inputs"]["image"], "pics/two.png")
        self.assertEqual(graph["5"]["inputs"]["prompt"], "row two")
        self.assertEqual([k for k in graph["5"]["inputs"] if k.startswith("ref_")], ["ref_images.ref_image_0"])
        self.assertNotIn("417", graph)
        self.assertFalse(template["34"]["inputs"]["save_output"])
        self.assertEqual(graph["133:129"]["inputs"]["sigmas"], ["22:7", 0])
        template["459"] = {"class_type":"MiniMaxH3RefModApply", "inputs":{"mods":["458",0]}}
        with_mod = build_prompt(template, "a.png", "A", "test")
        self.assertEqual(with_mod["135:29"]["inputs"]["conditioning"], ["459",0])
        self.assertIn("458", with_mod)

    def test_invalid_export_is_not_queued(self):
        template = exported_fixture()
        del template["133:129"]
        with self.assertRaisesRegex(ValueError, "Preview #2"):
            build_prompt(template, "a.png", "a", "x")

    def test_http_upload_serial_save_barrier_and_stop(self):
        for outcome in ("success", "oom", "no_file", "cancel"):
            with self.subTest(outcome=outcome):
                prompts, completed, releases, uploads, status, polled = [], [], [], [], [], set()
                class Server(BaseHTTPRequestHandler):
                    def reply(self, value=None):
                        self.send_response(200); self.end_headers()
                        if value is not None:
                            self.wfile.write(json.dumps(value).encode())

                    def do_POST(self):
                        raw = self.rfile.read(int(self.headers["Content-Length"]))
                        if self.path == "/upload/image":
                            uploads.append(raw)
                            return self.reply({"subfolder":"batch", "name":"a.png"})
                        if self.path == "/free":
                            releases.append(len(completed))
                            return self.reply()
                        assert self.path == "/prompt"
                        assert len(prompts) == len(completed), "second prompt sent before first saved"
                        prompts.append(json.loads(raw)["prompt"])
                        self.reply({"prompt_id":str(len(prompts))})

                    def do_GET(self):
                        if self.path == "/queue":
                            running = [[0,str(len(prompts))]] if len(prompts) > len(completed) and outcome != "cancel" else []
                            return self.reply({"queue_running":running, "queue_pending":[]})
                        assert self.path.startswith("/history/")
                        job = self.path.rsplit("/", 1)[-1]
                        if outcome == "cancel" or job not in polled:
                            polled.add(job)
                            return self.reply({})
                        if outcome == "oom":
                            return self.reply({job:{"status":{"completed":False, "status_str":"error",
                                "messages":[["execution_error", {"exception_message":"OOM"}]]}}})
                        completed.append(job)
                        self.reply({job:{"status":{"completed":True,"status_str":"success"},
                                         "outputs":{"34":{"gifs":[{"filename":"result.mp4",
                                             "type":"temp" if outcome == "no_file" else "output"}]}}}})

                    def log_message(self, *args):
                        pass
                with HTTPServer(("127.0.0.1", 0), Server) as server:
                    t = threading.Thread(target=server.serve_forever, daemon=True); t.start()
                    try:
                        base = f"http://127.0.0.1:{server.server_port}"
                        with tempfile.TemporaryDirectory() as d:
                            p = Path(d) / "同名.png"; p.write_bytes(b"fake-test-only")
                            self.assertEqual(upload(base, p), "batch/a.png")
                        self.assertIn(b'Content-Disposition: form-data; name="image"', uploads[0])
                        rows = [{"image":"batch/a.png", "prompt":"one"}, {"image":"b.png","prompt":"two"}]
                        run_batch(exported_fixture(), rows, base, lambda *event: status.append(event), pause=lambda _: None)
                    finally:
                        server.shutdown(); t.join()
                if outcome == "success":
                    self.assertEqual([g["5"]["inputs"]["prompt"] for g in prompts], ["one","two"])
                    self.assertEqual([g["900001"]["inputs"]["image"] for g in prompts], ["batch/a.png","b.png"])
                    self.assertEqual(releases,[1])
                    self.assertEqual([i for i, s, _ in status if s == "保存済み"], [0,1])
                else:
                    self.assertEqual(len(prompts), 1)
                    self.assertEqual(status[-1][:2], (0, "停止"))
                    self.assertEqual(releases, [])


@unittest.skipUnless(importlib.util.find_spec("aiohttp"), "aiohttp is part of the ComfyUI runtime")
class PodPanelTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_runs_without_browser_and_rejects_second_start(self):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
        from types import SimpleNamespace
        sys.path.insert(0, str(ROOT / "custom_nodes"))
        from foxhunter_compat.batch_routes import register

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "in.png").write_bytes(b"mock")
            app = web.Application()
            routes = web.RouteTableDef()
            register(SimpleNamespace(routes=routes), tmp)
            posted = asyncio.Event()
            finish = asyncio.Event()
            @routes.post("/prompt")
            async def create(request):
                self.assertEqual((await request.json())["prompt"]["900001"]["inputs"]["image"], "in.png")
                posted.set()
                return web.json_response({"prompt_id":"fake"})

            @routes.get("/history/fake")
            async def history(request):
                await asyncio.wait_for(finish.wait(), 5)
                return web.json_response({"fake":{"status":{"completed":True,"status_str":"success"},
                    "outputs":{"34":{"gifs":[{"filename":"final.mp4","type":"output"}]}}}})

            app.add_routes(routes)
            async with TestClient(TestServer(app)) as client:
                row = {"image":"in.png","prompt":"a"}
                graph = exported_fixture()
                bad = await client.post("/foxhunter/batch", json={"rows":[dict(row, image="../other.png")],"graph":graph})
                self.assertEqual(bad.status, 400)
                res = await client.post("/foxhunter/batch", json={"rows":[row],"graph":graph})
                self.assertEqual(res.status, 200)
                job = (await res.json())["id"]
                await asyncio.wait_for(posted.wait(), 5)
                second = await client.post("/foxhunter/batch", json={"rows":[row],"graph":graph})
                self.assertEqual(second.status, 409)
                finish.set()
                for _ in range(100):
                    state = await (await client.get("/foxhunter/batch/current")).json()
                    if not state["running"]:
                        break
                    await asyncio.sleep(.02)
                self.assertEqual(state["id"], job)
                self.assertEqual(state["rows"][0]["state"], "保存済み")


if __name__ == "__main__":
    unittest.main()
