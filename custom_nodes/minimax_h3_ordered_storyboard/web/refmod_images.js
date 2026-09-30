import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const MAX_IMAGES = 8;
function element(tag, text) {
    const item = document.createElement(tag);
    if (text) item.textContent = text;
    return item;
}
function viewUrl(name, type = "input") {
    const parts = name.split("/");
    const filename = parts.pop();
    return api.apiURL(`/view?${new URLSearchParams({ filename, subfolder: parts.join("/"), type })}`);
}

function installImages(node) {
    if (node.__refmodEditor) return;
    const field = node.widgets?.find(w => w.name === "images_json");
    if (!field || !node.addDOMWidget) return;
    field.type = "hidden";
    // Current ComfyUI DOM widgets check `hidden`, not just the canvas type.
    // Keep the widget/value for workflow serialization, but never show raw JSON.
    field.hidden = true;
    field.computeSize = () => [0, -4];
    let files = [], busy = false, error = "";
    const root = element("div");
    Object.assign(root.style, { overflowY: "auto", height: "100%", boxSizing: "border-box",
        padding: "12px", background: "#202329", color: "#e8edf7", font: "14px/1.5 sans-serif" });
    const picker = element("input");
    picker.type = "file";
    picker.accept = "image/png,image/jpeg,image/webp,image/bmp,image/tiff";
    picker.multiple = true;
    picker.style.display = "none";

    function commit() {
        field.value = JSON.stringify(files);
        field.callback?.(field.value);
        node.graph?.setDirtyCanvas?.(true, true);
        render();
    }
    function button(text, action) {
        const b = element("button", text);
        b.type = "button";
        b.disabled = busy;
        Object.assign(b.style, { cursor: "pointer", padding: "6px 10px", margin: "3px",
            color: "#eef2ff", background: "#334155", border: "1px solid #64748b", borderRadius: "5px" });
        b.onclick = action;
        return b;
    }
    async function upload(incoming) {
        if (busy || !incoming.length) return;
        if (files.length + incoming.length > MAX_IMAGES) {
            error = "参照は最大8枚です。不要な画像を外してから追加してください。";
            render(); return;
        }
        busy = true; error = ""; render();
        try {
            for (const file of incoming) {
                if (!/\.(png|jpe?g|webp|bmp|tiff?)$/i.test(file.name)) throw new Error("静止画のPNG/JPEG/WebP/BMP/TIFFを選んでください。");
                const body = new FormData();
                body.append("image", file, file.name);
                body.append("type", "input");
                body.append("subfolder", "minimax_h3_character");
                body.append("overwrite", "false");
                const response = await api.fetchApi("/upload/image", { method: "POST", body });
                if (!response.ok) throw new Error(`画像アップロード失敗 (${response.status})`);
                const asset = await response.json();
                if (typeof asset.name !== "string" || !asset.name) throw new Error("アップロード応答にファイル名がありません。");
                const name = [asset.subfolder, asset.name].filter(Boolean).join("/");
                if (!files.includes(name)) files.push(name);
                commit();
            }
        } catch (e) { error = String(e.message || e); }
        finally { busy = false; picker.value = ""; render(); }
    }
    picker.onchange = () => upload(Array.from(picker.files || []));
    root.ondragover = event => { event.preventDefault(); event.stopPropagation(); };
    root.ondrop = event => {
        event.preventDefault(); event.stopPropagation();
        upload(Array.from(event.dataTransfer?.files || []));
    };

    function render() {
        root.replaceChildren(picker);
        root.append(element("div", "同じキャラの画像を1〜8枚。画像は開始フレームではありません。"));
        root.append(button(busy ? "アップロード中…" : "＋画像を追加（複数選択可）", () => picker.click()));
        root.append(element("div", `${files.length} / ${MAX_IMAGES} 枚 · ここへドラッグ＆ドロップも可能`));
        if (error) { const e = element("div", error); e.style.color = "#ffb4ab"; root.append(e); }
        files.forEach((name, i) => {
            const row = element("div");
            Object.assign(row.style, { display: "grid", gridTemplateColumns: "76px 1fr", gap: "10px",
                padding: "10px 0", borderTop: "1px solid #475569" });
            const img = element("img"); img.src = viewUrl(name); img.alt = `Picture ${i + 1}`;
            Object.assign(img.style, { width: "76px", height: "90px", objectFit: "contain" });
            const body = element("div");
            const label = element("div", `<Picture ${i + 1}> · ${name.split("/").pop()}`);
            label.style.overflowWrap = "anywhere"; body.append(label);
            body.append(button("外す", () => { files.splice(i, 1); commit(); }));
            if (i > 0) body.append(button("↑", () => { [files[i-1], files[i]] = [files[i], files[i-1]]; commit(); }));
            if (i + 1 < files.length) body.append(button("↓", () => { [files[i+1], files[i]] = [files[i], files[i+1]]; commit(); }));
            row.append(img, body); root.append(row);
        });
        node.graph?.setDirtyCanvas?.(true, true);
    }
    node.addDOMWidget("refmod_image_editor", "div", root, { serialize: false, hideOnZoom: false });
    node.__refmodEditor = () => {
        try {
            const value = JSON.parse(field.value || "[]");
            if (!Array.isArray(value) || value.some(x => typeof x !== "string") || value.length > MAX_IMAGES) throw new Error();
            files = value; error = "";
        } catch (_) { files = []; error = "保存された画像リストが不正です。画像を追加し直してください。"; }
        render();
    };
    node.__refmodEditor();
}

app.registerExtension({
    name: "MiniMaxH3.CharacterRefMod",
    async beforeRegisterNodeDef(nodeType, data) {
        if (data.name === "MiniMaxH3RefModImages") {
            const created = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () { const r = created?.apply(this, arguments); installImages(this); return r; };
            const configure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () { const r = configure?.apply(this, arguments); setTimeout(() => this.__refmodEditor?.(), 0); return r; };
        }
        if (data.name === "MiniMaxH3SaveCharacterRefMod") {
            const executed = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                executed?.apply(this, arguments);
                if (!this.__refmodDownloads) {
                    this.__refmodDownloads = element("div");
                    this.addDOMWidget("refmod_downloads", "div", this.__refmodDownloads, { serialize: false });
                }
                this.__refmodDownloads.replaceChildren();
                for (const file of message.refmod_files || []) {
                    const a = element("a", `RefModを保存: ${file.filename}`);
                    a.href = viewUrl(`${file.subfolder}/${file.filename}`, "output");
                    a.download = file.filename;
                    a.style.color = "#9bd5ff"; a.style.overflowWrap = "anywhere";
                    this.__refmodDownloads.append(a);
                }
            };
        }
    },
});
