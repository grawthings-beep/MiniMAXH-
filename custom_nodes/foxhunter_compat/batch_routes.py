"""Private, single-user FoxHunter Pod batch queue. Only one prompt in flight."""

import threading
import uuid
from pathlib import Path

from aiohttp import web
from .sample2_batch import build_prompt, run_batch


def register(server, input_directory):
    lock = threading.Lock()
    state = {"id": None, "running": False, "rows": []}
    root = Path(input_directory).resolve()

    @server.routes.post("/foxhunter/batch")
    async def start(request):
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        payload = await request.json()
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="JSONオブジェクトが必要です")
        rows, graph = payload.get("rows"), payload.get("graph")
        if not isinstance(rows, list) or not 1 <= len(rows) <= 100 or not isinstance(graph, dict):
            raise web.HTTPBadRequest(text="画像とプロンプトを1〜100組指定してください")
        try:
            for row in rows:
                path = (root / row["image"]).resolve()
                if (not path.is_relative_to(root) or not path.is_file() or
                        path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"} or
                        not isinstance(row["prompt"], str) or not row["prompt"].strip()):
                    raise ValueError("画像またはプロンプトが不正です")
                row["label"] = Path(str(row.get("label", path.name))).stem[:100]
            build_prompt(graph, rows[0]["image"], rows[0]["prompt"], "foxhunter_batch/check")
        except (TypeError, ValueError, KeyError, OSError) as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc

        batch = uuid.uuid4().hex
        port = request.transport.get_extra_info("sockname")[1]
        with lock:
            if state["running"]:
                raise web.HTTPConflict(text="前の一括生成がまだ実行中です")
            state.update(id=batch, running=True, rows=[{"label": r["label"], "state": "待機", "detail": ""} for r in rows])

        def update(i, stage, detail):
            with lock:
                state["rows"][i].update(state=stage, detail=detail[:1800])

        def work():
            try:
                run_batch(graph, rows, f"http://127.0.0.1:{port}", update)
            finally:
                with lock:
                    state["running"] = False
        threading.Thread(target=work, daemon=True, name="foxhunter-batch").start()
        return web.json_response({"id": batch})

    @server.routes.get("/foxhunter/batch/{batch}")
    async def status(request):
        with lock:
            if not state["id"] or request.match_info["batch"] not in ("current", state["id"]):
                raise web.HTTPNotFound(text="このPodには該当する一括生成がありません")
            return web.json_response(state, headers={"Cache-Control": "no-store"})
