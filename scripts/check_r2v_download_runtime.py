#!/usr/bin/env python3
"""Real aria2 binary smoke: private stdin, pinned filename and HTTP range resume.

Only this isolated loopback fixture relaxes HTTPS validation; production always
uses HTTPS. It needs no Civitai account, external network or model weights.
"""
import json
import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import download_civitai_models as downloader


def main():
    body = b"MiniMax-H3-aria2-fixture-" * 262144
    header = json.dumps({"fixture": {"dtype": "U8", "shape": [len(body)], "data_offsets": [0, len(body)]}}).encode()
    payload = len(header).to_bytes(8, "little") + header + body
    ranges = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.respond(head=True)

        def do_GET(self):
            self.respond(head=False)

        def respond(self, head):
            value = self.headers.get("Range", "")
            start, end = 0, len(payload) - 1
            if value:
                ranges.append(value)
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", value)
                if not match:
                    self.send_error(416)
                    return
                start = int(match[1])
                end = min(int(match[2]) if match[2] else end, end)
            self.send_response(206 if value else 200)
            if value:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            # The server's filename must not override our manifest filename.
            self.send_header("Content-Disposition", 'attachment; filename="server-selected-name.safetensors"')
            self.end_headers()
            if not head:
                try:
                    self.wfile.write(payload[start:end + 1])
                except (BrokenPipeError, ConnectionResetError):
                    pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original_validation = downloader.safe_https
    url = f"http://127.0.0.1:{server.server_port}/wrong-name?fixture=not-a-real-signature"

    def loopback_fixture_only(candidate):
        if candidate != url:
            return original_validation(candidate)

    try:
        with tempfile.TemporaryDirectory(prefix="h3-aria2-smoke-") as temp:
            destination = Path(temp) / "pinned-model.safetensors.part"
            with mock.patch.object(downloader, "safe_https", side_effect=loopback_fixture_only):
                downloader.transfer(url, destination, 4, 30)
                assert destination.read_bytes() == payload
                files = sorted(p.name for p in Path(temp).iterdir())
                # Windows aria2 may leave its temporary progress metadata file.
                assert destination.name in files and set(files) <= {
                    destination.name, destination.name + ".aria2__temp"
                }, files
                # No control file: aria2 retains complete 1 MiB pieces only, so
                # use an aligned prefix, not a sub-piece 4 KiB fragment.
                resume_offset = 1024 * 1024
                destination.write_bytes(payload[:resume_offset])
                ranges.clear()
                downloader.transfer(url, destination, 4, 30)
                assert destination.read_bytes() == payload
                assert any(value.startswith(f"bytes={resume_offset}-") for value in ranges), ranges
                assert not destination.with_name(destination.name + ".aria2").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    print("R2VA download runtime: real aria2 stdin filename + resumed byte range passed (loopback, no secrets)")


if __name__ == "__main__":
    main()
