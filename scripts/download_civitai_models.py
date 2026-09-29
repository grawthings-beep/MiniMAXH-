#!/usr/bin/env python3
"""Pinned checkpoints: authenticated redirect, parallel resume, mandatory SHA256.

Account tokens never enter aria2 arguments, input, logs or redirect headers.
Temporary signed CDN URLs are private stdin data and aria2 output is suppressed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

from download_lora import inspect_safetensors
from verify_models import verify_one


class DownloadError(RuntimeError):
    """Only sanitized messages may be raised as this type."""


class AuthenticationError(DownloadError):
    pass


def safe_https(url: str) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or any(ord(c) < 33 for c in url)):
        raise DownloadError("Invalid HTTPS download location")
    return parsed


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        safe_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def validate_asset(asset: dict) -> None:
    path = PurePosixPath(asset["path"])
    if (len(path.parts) != 2 or path.parts[0] != "diffusion_models"
            or not re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", path.name)):
        raise DownloadError("Checkpoint path must be a filename inside diffusion_models")
    parsed = safe_https(asset["source_url"])
    if (parsed.hostname != "civitai.com" or parsed.port is not None
            or not re.fullmatch(r"/api/download/models/\d+", parsed.path)
            or not re.fullmatch(r"fileId=\d+", parsed.query) or parsed.fragment):
        raise DownloadError("Checkpoint source must pin a Civitai version and fileId without credentials")
    if int(asset["size"]) <= 0 or not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]):
        raise DownloadError("Checkpoint needs a pinned size and SHA256")


def signed_url(url: str, token: str, timeout: int) -> str:
    request = urllib.request.Request(url, headers={
        "Range": "bytes=0-0", "User-Agent": "MiniMAXH-RunPod/DaSiWa",
        "Accept": "application/octet-stream",
    })
    request.add_unredirected_header("Authorization", f"Bearer {token}")
    opener = urllib.request.build_opener(HTTPSRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            final = response.geturl()
            parsed = safe_https(final)
            if parsed.hostname in {"civitai.com", "civitai.red"} or token in final:
                raise DownloadError("Civitai did not return a separate signed CDN location")
            return final
    except urllib.error.HTTPError as error:
        if error.code in {401, 403}:
            raise AuthenticationError(
                f"Civitai HTTP {error.code}; check CIVITAI_API_TOKEN and checkpoint download access"
            ) from None
        raise DownloadError(f"Civitai/CDN HTTP {error.code}; retrying with a new signed link") from None


def transfer(url: str, partial: Path, connections: int, timeout: int) -> None:
    safe_https(url)
    if any(ord(c) < 32 for c in str(partial)):
        raise DownloadError("Invalid checkpoint output path")
    command = [
        "aria2c", "--no-conf=true", "--input-file=-", "--quiet=true", "--console-log-level=error",
        "--summary-interval=0", "--download-result=hide", "--enable-color=false",
        "--continue=true", "--file-allocation=none", "--auto-file-renaming=false",
        "--allow-overwrite=true", "--min-split-size=16M", "--max-tries=3", "--retry-wait=3",
        f"--max-connection-per-server={connections}", f"--split={connections}",
        f"--timeout={timeout}", f"--dir={partial.parent}",
    ]
    # Popen/communicate keeps the signed URL off the command line and process list.
    child_env = {k: v for k, v in os.environ.items()
                 if k not in {"CIVITAI_API_TOKEN", "CIVITAI_TOKEN", "HF_TOKEN"}}
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, text=True, env=child_env)
    try:
        # With --input-file, a global --out is ignored. Pin the name per entry.
        proc.communicate(f"{url}\n  out={partial.name}\n")
        if proc.returncode:
            raise DownloadError(f"Parallel checkpoint transfer interrupted (aria2 exit {proc.returncode})")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def checked(path: Path, asset: dict) -> bool:
    probe = dict(asset, path=path.name)
    if verify_one(path.parent, probe, "sha256")[1] != "ok":
        return False
    try:
        inspect_safetensors(path)
    except (ValueError, RuntimeError, UnicodeError):
        return False
    return True


def download(asset: dict, root: Path, token: str, *, retries: int, connections: int, timeout: int) -> None:
    validate_asset(asset)
    destination = root / asset["path"]
    # Reject symlinks escaping the intended model tree before any file operation.
    if not destination.resolve().is_relative_to(root.resolve()):
        raise DownloadError("Checkpoint destination escapes the model directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    control = partial.with_name(partial.name + ".aria2")
    if any(p.is_symlink() for p in (destination, partial, control)):
        raise DownloadError("Checkpoint download paths must not be symlinks")
    if destination.is_file() and checked(destination, asset):
        print(f"[r2v-checkpoint] {destination.name} ready; size + SHA256 verified", flush=True)
        return
    for attempt in range(1, retries + 1):
        try:
            print(f"[r2v-checkpoint] {destination.name}: attempt {attempt}/{retries}, {connections} range connections", flush=True)
            # A fully transferred .part can survive an interrupted verification/rename.
            complete = partial.is_file() and not control.exists() and checked(partial, asset)
            if not complete:
                # Discard a completed corrupt transfer, not an incomplete resumable one.
                if partial.exists() and not control.exists() and partial.stat().st_size >= asset["size"]:
                    partial.unlink()
                transfer(signed_url(asset["source_url"], token, timeout), partial, connections, timeout)
                if not checked(partial, asset):
                    partial.unlink(missing_ok=True)
                    control.unlink(missing_ok=True)
                    raise DownloadError("Checkpoint size, SHA256 or safetensors verification failed")
            os.replace(partial, destination)
            control.unlink(missing_ok=True)
            print(f"[r2v-checkpoint] {destination.name} ready; size + SHA256 verified", flush=True)
            return
        except AuthenticationError:
            raise
        except Exception as error:
            # Exceptions from urllib/subprocess can contain signed URLs: never stringify them.
            message = str(error) if isinstance(error, DownloadError) else "Checkpoint download interrupted (details withheld)"
            print(f"[r2v-checkpoint] {message}", file=sys.stderr, flush=True)
            if attempt < retries:
                time.sleep(3 * attempt)
    raise DownloadError("Checkpoint download failed; ComfyUI must not start")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assets = [f for f in manifest["files"] if f.get("auth") == "civitai"]
    if not assets:
        return
    token = os.environ.get("CIVITAI_API_TOKEN", "").strip()
    if not token or any(ord(c) < 32 for c in token):
        raise AuthenticationError("CIVITAI_API_TOKEN is required for the selected DaSiWa checkpoint")
    for asset in assets:
        download(asset, args.root, token, retries=max(1, int(os.environ.get("DOWNLOAD_RETRIES", "3"))),
                 connections=max(1, min(16, int(os.environ.get("ARIA2_CONNECTIONS_PER_FILE", "16")))),
                 timeout=max(30, int(os.environ.get("CIVITAI_DOWNLOAD_TIMEOUT", "120"))))


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        main()
    except Exception as error:
        message = str(error) if isinstance(error, DownloadError) else "Checkpoint setup failed (details withheld)"
        print(f"[r2v-checkpoint] ERROR: {message}", file=sys.stderr)
        raise SystemExit(1) from None
