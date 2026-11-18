"""Portable desktop client for an already-running FoxHunter Pod. Python stdlib only."""

import json
import mimetypes
import queue
import re
import sys
import threading
import uuid
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "custom_nodes" / "foxhunter_compat"))
from sample2_batch import build_prompt, run_batch


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_prompt_list(text):
    """[filename.png] on its own line starts a prompt; retain multiline content."""
    blocks = {}
    filename = None
    for line in text.lstrip("\ufeff").splitlines():
        header = re.fullmatch(r"\[(.+)\]", line.strip())
        if header and re.search(r"\.[\w-]+\s*$", header[1]):
            name = header[1]
            path = PureWindowsPath(name)
            if path.suffix.lower() not in IMAGE_EXTS:
                raise ValueError(f"対応画像は PNG / JPG / JPEG / WEBP です：{name}")
            if (name != path.name or name != name.strip()
                    or re.match(r"^(?:CON|PRN|AUX|NUL|(?:COM|LPT)[0-9¹²³])\s*\.", name, re.I)
                    or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)):
                raise ValueError(f"見出しにはフォルダを含まない画像名だけを指定してください：{name}")
            if name.casefold() in blocks:
                raise ValueError(f"一覧に同じ画像名が2回あります：{name}")
            filename = name.casefold()
            blocks[filename] = (name, [])
        elif filename is not None:
            blocks[filename][1].append(line)
        elif line.strip():
            raise ValueError("一覧の先頭行を [画像名.png] にして、次の行にプロンプトを書いてください。")
    if not blocks:
        raise ValueError("一覧に [画像名.png] とプロンプトがありません。")
    result = {}
    for key, (name, lines) in blocks.items():
        prompt = "\n".join(lines).strip()
        if not prompt:
            raise ValueError(f"プロンプトが空です：{name}")
        result[key] = (name, prompt)
    return result


def match_prompts(blocks, rows, directory=None):
    """Validate everything before changing any row. Never match by list position."""
    if not rows:
        folder = Path(directory).resolve(strict=True)
        paths = [folder / name for name, _ in blocks.values()]
        for path in paths:
            if not path.is_file() or path.resolve().parent != folder:
                raise ValueError(f"選択フォルダに画像がありません：{path.name}")
        return [{"path": p, "state": "待機", "prompt": blocks[p.name.casefold()][1]} for p in paths]

    names = [r["path"].name.casefold() for r in rows]
    if len(set(names)) != len(names):
        raise ValueError("同名の画像があるため照合できません。名前を分けるか重複行を外してください。")
    unknown = blocks.keys() - set(names)
    if unknown:
        raise ValueError("未登録の画像名：" + "、".join(blocks[k][0] for k in sorted(unknown)))
    missing = [r["path"].name for r, k in zip(rows, names) if r["state"] != "保存済み" and k not in blocks]
    if missing:
        raise ValueError("一覧にプロンプトがない画像：" + "、".join(missing))
    return [dict(r, prompt=blocks[k][1]) if r["state"] != "保存済み" else dict(r)
            for r, k in zip(rows, names)]


def upload(base, path):
    # UUID names also prevent overwriting previous uploads with the same basename.
    boundary = "fox" + uuid.uuid4().hex
    name = uuid.uuid4().hex + path.suffix.lower()
    start = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
             f"Content-Type: {mimetypes.guess_type(name)[0] or 'application/octet-stream'}\r\n\r\n").encode()
    req = Request(base + "/upload/image", data=start + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode(),
                  headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urlopen(req, timeout=120) as response:
        saved = json.load(response)
    return "/".join(x for x in (saved.get("subfolder"), saved["name"]) if x)


def main():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    win = tk.Tk()
    win.title("FoxHunter  |  Sample #2 → Upscale")
    win.geometry("800x590")
    win.columnconfigure(1, weight=1)
    win.rowconfigure(4, weight=1)
    base = tk.StringVar()
    template_path = tk.StringVar()
    ttk.Label(win, text="ComfyUIのURL").grid(row=0, column=0, sticky="w")
    ttk.Entry(win, textvariable=base).grid(row=0, column=1, sticky="ew")
    buttons = []
    def button(text, command, row, col):
        b = ttk.Button(win, text=text, command=command)
        b.grid(row=row, column=col, sticky="ew")
        buttons.append(b)
    def pick_template():
        path = filedialog.askopenfilename(filetypes=[("FoxHunter API JSON", "*.json")])
        if path:
            template_path.set(path)
    button("① API JSON を選ぶ", pick_template, 1, 0)
    ttk.Label(win, textvariable=template_path).grid(row=1, column=1, sticky="w")
    ttk.Label(win, text="一覧形式： [画像名.png] を1行に書き、次の行からプロンプト。画像と一覧が同じフォルダなら②だけで全件登録。").grid(
        row=3, columnspan=2, sticky="w")
    listing = tk.Listbox(win, width=32, exportselection=False)
    listing.grid(row=4, column=0, sticky="nsew")
    editor = tk.Text(win, wrap="word", undo=True)
    editor.grid(row=4, column=1, sticky="nsew")
    status = tk.StringVar(value="待機中。画像1枚につき動画1本。")
    ttk.Label(win, textvariable=status, wraplength=740).grid(row=6, columnspan=2, sticky="w")
    rows, selected, active = [], None, False
    events = queue.Queue()

    def commit():
        if selected is not None:
            rows[selected]["prompt"] = editor.get("1.0", "end-1c")

    def select(event=None):
        nonlocal selected
        choice = listing.curselection()
        if not choice:
            return
        commit()
        selected = choice[0]
        editor.delete("1.0", "end")
        editor.insert("1.0", rows[selected]["prompt"])
        status.set(rows[selected].get("detail") or str(rows[selected]["path"]))

    def show(i):
        row = rows[i]
        listing.insert(i, f"{i+1:03}  {row['state']}  {row['path'].name}")

    def add():
        for f in filedialog.askopenfilenames(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")]):
            p = Path(f)
            txt = p.with_suffix(".txt")
            try:
                prompt = txt.read_text(encoding="utf-8-sig") if txt.exists() else ""
            except (OSError, UnicodeError) as exc:
                messagebox.showerror("プロンプト読み込み失敗", str(exc))
                prompt = ""
            rows.append({"path": p, "prompt": prompt, "state": "待機"})
            show(len(rows)-1)
        if selected is None and rows:
            listing.selection_set(0)
            select()

    def bulk(from_file):
        nonlocal selected
        folder = None
        try:
            if from_file:
                filename = filedialog.askopenfilename(title="画像別プロンプト一覧",
                                                       filetypes=[("UTF-8 テキスト", "*.txt *.md")])
                if not filename:
                    return
                source = Path(filename)
                text = source.read_text(encoding="utf-8-sig")
                folder = source.parent
            else:
                try:
                    text = win.clipboard_get()
                except tk.TclError:
                    raise ValueError("画像別プロンプト一覧を先にコピーしてください。") from None
            blocks = parse_prompt_list(text)
            if not rows and (folder is None or any(not (folder / n).is_file() for n, _ in blocks.values())):
                folder = filedialog.askdirectory(title="一覧に書いた画像が入っているフォルダを選択")
                if not folder:
                    return
            commit()
            new_rows = match_prompts(blocks, rows, folder)
        except (OSError, UnicodeError, ValueError) as exc:
            messagebox.showerror("一括登録できません", str(exc))
            return
        if any(old["prompt"].strip() and old["prompt"] != new["prompt"] for old, new in zip(rows, new_rows)):
            if not messagebox.askyesno("入力済みのプロンプト", "一致した未完了行のプロンプトを一覧の内容に置き換えますか？"):
                return
        current = selected if selected is not None else 0
        selected = None  # do not commit stale editor text over the imported value
        rows[:] = new_rows
        listing.delete(0, "end")
        for i in range(len(rows)):
            show(i)
        listing.selection_set(current)
        select()
        status.set(f"画像とプロンプトを照合しました。未完了 {sum(r['state'] != '保存済み' for r in rows)} 件。③で生成開始。")

    def remove():
        nonlocal selected
        if selected is None:
            return
        i = selected
        selected = None
        rows.pop(i)
        listing.delete(0, "end")
        for n in range(len(rows)):
            show(n)
        editor.delete("1.0", "end")
        if rows:
            listing.selection_set(min(i, len(rows)-1))
            select()

    def lock(on):
        nonlocal active
        active = on
        for w in [*buttons, listing, editor]:
            w.configure(state="disabled" if on else "normal")

    def start():
        commit()
        try:
            url = base.get().strip().rstrip("/")
            parsed = urlsplit(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment or parsed.username:
                raise ValueError("RunPodのComfyUIポート8188の接続URLを指定してください。")
            template = json.loads(Path(template_path.get()).read_text(encoding="utf-8-sig"))
            build_prompt(template, "preflight.png", "preflight", "foxhunter_batch/preflight")
            todo = [(i, r["path"], r["prompt"].strip()) for i, r in enumerate(rows) if r["state"] != "保存済み"]
            if not todo:
                raise ValueError("未生成の行がありません。")
            for i, p, prompt in todo:
                if not prompt or not p.is_file():
                    raise ValueError(f"{i+1}行目：画像またはプロンプトがありません。")
            if any(rows[i]["state"] == "停止" for i, *_ in todo) and not messagebox.askyesno(
                    "停止した行を再実行", "通信切断の場合、その行はPodで実行中または保存済みかもしれません。履歴を確認しましたか？"):
                return
        except Exception as exc:
            messagebox.showerror("実行前チェック", str(exc))
            return
        lock(True)
        status.set("画像をPodへ登録中…")

        def worker():
            try:
                jobs = [{"image": upload(url, p), "label": p.stem, "prompt": prompt} for _, p, prompt in todo]
                run_batch(template, jobs, url, lambda j, s, d: events.put((todo[j][0], s, d)))
            except Exception as exc:
                events.put((None, "登録失敗", str(exc)))
            finally:
                events.put((None, "終了", ""))
        threading.Thread(target=worker, daemon=True).start()

    def poll():
        while not events.empty():
            i, state, detail = events.get_nowait()
            if state == "終了":
                lock(False)
                continue
            if i is not None:
                rows[i].update(state=state, detail=detail)
                listing.configure(state="normal")
                listing.delete(i)
                show(i)
                listing.configure(state="disabled")
            status.set(f"{i+1 if i is not None else ''} {state}  {detail}")
        win.after(200, poll)

    listing.bind("<<ListboxSelect>>", select)
    button("② 一覧ファイルを開く", lambda: bulk(True), 2, 0)
    button("または：コピーした一覧を一括登録", lambda: bulk(False), 2, 1)
    button("個別編集用：画像を追加", add, 7, 0)
    button("選択行を外す", remove, 7, 1)
    button("③ 未完了の行を順番に生成", start, 5, 1)
    def close():
        if not active or messagebox.askyesno("実行中", "ウィンドウを閉じると後続行は送信されません。閉じますか？"):
            win.destroy()
    win.protocol("WM_DELETE_WINDOW", close)
    poll()
    win.mainloop()


if __name__ == "__main__":
    main()
