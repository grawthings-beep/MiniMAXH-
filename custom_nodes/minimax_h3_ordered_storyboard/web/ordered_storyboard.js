import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "MiniMaxH3OrderedStoryboard";
const DEFAULT_DURATION = 6.5;
const INPUT_SUBFOLDER = "minimax_h3_storyboard";
const MAX_IMAGES = 100;
const MAX_TOTAL_DURATION = 90;
const STORY_FPS = 24;

function alignedFrameCount(durationSec) {
    const frames = Math.max(5, Math.round(Math.max(0.1, Number(durationSec) || 0.1) * STORY_FPS));
    return frames + ((5 - (frames % 17)) % 17);
}

function freshId() {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    return `keyframe-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function asBoolean(value) {
    return value === true || value === 1 || value === "1" || value === "true" || value === "on";
}

function seedBigInt(value) {
    const text = String(value ?? "0").trim();
    return /^\d+$/.test(text) ? BigInt(text) : 0n;
}

function normalizeClientState(value) {
    let raw = value;
    if (typeof raw === "string") {
        try {
            raw = JSON.parse(raw || "{}");
        } catch (_) {
            raw = {};
        }
    }
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) raw = {};
    const defaults = raw.defaults && typeof raw.defaults === "object" ? raw.defaults : {};
    const defaultSeed = String(defaults.seed ?? raw.default_seed ?? "0");
    const state = {
        version: 1,
        loop: asBoolean(raw.loop),
        defaults: {
            prompt: String(defaults.prompt ?? raw.default_prompt ?? ""),
            duration_sec: Number(defaults.duration_sec ?? raw.default_duration_sec ?? DEFAULT_DURATION),
            seed: defaultSeed,
        },
        images: [],
    };
    if (!Number.isFinite(state.defaults.duration_sec)) state.defaults.duration_sec = DEFAULT_DURATION;
    const images = Array.isArray(raw.images) ? raw.images : [];
    state.images = images.map((entry, index) => {
        if (typeof entry === "string") entry = { name: entry };
        if (!entry || typeof entry !== "object") entry = {};
        const transition = entry.transition && typeof entry.transition === "object" ? entry.transition : {};
        return {
            id: String(entry.id || freshId()),
            name: String(entry.name ?? entry.filename ?? `missing-${index + 1}.png`),
            subfolder: String(entry.subfolder ?? ""),
            type: "input",
            transition: {
                prompt: String(transition.prompt ?? state.defaults.prompt),
                duration_sec: Number(transition.duration_sec ?? state.defaults.duration_sec),
                seed: String(transition.seed ?? (seedBigInt(defaultSeed) + BigInt(index))),
            },
        };
    });
    return state;
}

function imageViewUrl(asset) {
    const query = new URLSearchParams({ filename: asset.name, type: "input" });
    if (asset.subfolder) query.set("subfolder", asset.subfolder);
    return api.apiURL(`/view?${query.toString()}`);
}

function el(tag, props = {}, children = []) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
        if (key === "className") node.className = value;
        else if (key === "text") node.textContent = value;
        else if (key === "style") Object.assign(node.style, value);
        else if (key.startsWith("on") && typeof value === "function") {
            node.addEventListener(key.slice(2).toLowerCase(), value);
        } else if (value !== undefined) {
            node[key] = value;
        }
    }
    for (const child of children) node.append(child);
    return node;
}

function installEditor(node) {
    if (node.__h3OrderedStoryboardInstalled) return;
    const stateWidget = node.widgets?.find((widget) => widget.name === "storyboard_data");
    if (!stateWidget || typeof node.addDOMWidget !== "function") return;
    node.__h3OrderedStoryboardInstalled = true;

    stateWidget.type = "hidden";
    stateWidget.computeSize = () => [0, -4];

    let state = normalizeClientState(stateWidget.value);
    const root = el("div", {
        className: "h3-ordered-storyboard",
        style: {
            boxSizing: "border-box",
            width: "100%",
            minWidth: "520px",
            maxHeight: "720px",
            overflowY: "auto",
            padding: "8px",
            color: "var(--fg-color, #ddd)",
            font: "12px sans-serif",
        },
    });
    const fileInput = el("input", {
        type: "file",
        accept: "image/png,image/jpeg,image/webp,image/bmp,image/gif,image/tiff",
        multiple: true,
        style: { display: "none" },
    });
    root.append(fileInput);

    const sync = () => {
        stateWidget.value = JSON.stringify(state);
        stateWidget.callback?.(stateWidget.value);
        node.graph?.setDirtyCanvas?.(true, true);
    };

    const showError = (message) => {
        node.__h3OrderedStoryboardError = message ? String(message) : "";
        render();
    };

    async function uploadFiles(files) {
        if (!files?.length) return;
        if (state.images.length + files.length > MAX_IMAGES) {
            const remaining = Math.max(0, MAX_IMAGES - state.images.length);
            throw new Error(`最大${MAX_IMAGES}枚です。現在あと${remaining}枚追加できます。`);
        }
        showError("");
        for (const file of files) {
            const body = new FormData();
            body.append("image", file, file.name);
            body.append("type", "input");
            body.append("subfolder", INPUT_SUBFOLDER);
            body.append("overwrite", "false");
            const response = await api.fetchApi("/upload/image", { method: "POST", body });
            if (!response.ok) {
                const detail = await response.text();
                throw new Error(`Upload failed (${response.status}): ${detail}`);
            }
            const asset = await response.json();
            const seedBase = seedBigInt(state.defaults.seed);
            state.images.push({
                id: freshId(),
                name: String(asset.name || file.name),
                subfolder: String(asset.subfolder || INPUT_SUBFOLDER),
                type: "input",
                transition: {
                    prompt: state.defaults.prompt,
                    duration_sec: state.defaults.duration_sec,
                    seed: String(seedBase + BigInt(state.images.length)),
                },
            });
            // Persist every successful upload immediately. If a later file in
            // the same selection fails, the visible cards and queued JSON agree.
            sync();
        }
        render();
    }

    fileInput.addEventListener("change", async () => {
        try {
            await uploadFiles(Array.from(fileInput.files || []));
        } catch (error) {
            showError(error?.message || error);
        } finally {
            fileInput.value = "";
        }
    });

    function transitionEditor(image, index) {
        const targetIndex = index + 1 < state.images.length ? index + 1 : 0;
        const wrap = el("div", {
            style: {
                margin: "6px 0 2px 44px",
                padding: "7px",
                borderLeft: "2px solid #5c8bd6",
                background: "rgba(80,120,180,.08)",
            },
        });
        wrap.append(el("div", {
            text: `Transition ${index + 1}: #${index + 1} → #${targetIndex + 1}`,
            style: { marginBottom: "4px", color: "#9fc4ff", fontWeight: "600" },
        }));
        const prompt = el("textarea", {
            value: image.transition.prompt,
            placeholder: "この区間のプロンプト",
            rows: 2,
            style: {
                boxSizing: "border-box",
                width: "100%",
                resize: "vertical",
                color: "inherit",
                background: "#171717",
                border: "1px solid #555",
                borderRadius: "4px",
                padding: "5px",
            },
            oninput: (event) => {
                image.transition.prompt = event.target.value;
                sync();
            },
        });
        wrap.append(prompt);

        const settings = el("div", {
            style: { display: "grid", gridTemplateColumns: "45px 1fr 110px 1fr", gap: "5px", marginTop: "5px", alignItems: "center" },
        });
        settings.append(el("label", { text: "秒数" }));
        settings.append(el("input", {
            type: "number",
            value: image.transition.duration_sec,
            min: 0.2,
            max: 15,
            step: 0.1,
            style: { minWidth: "0", color: "inherit", background: "#171717", border: "1px solid #555", borderRadius: "4px", padding: "4px" },
            onchange: (event) => {
                const value = Number(event.target.value);
                if (Number.isFinite(value)) image.transition.duration_sec = value;
                sync();
                render();
            },
        }));
        settings.append(el("label", {
            text: "Seed（保存のみ）",
            title: "現行AIMixer Directorは区間別seedを適用せず、Director本体の全体seedを使います。",
        }));
        settings.append(el("input", {
            type: "text",
            inputMode: "numeric",
            value: image.transition.seed,
            pattern: "[0-9]*",
            title: "将来の区間seed対応用に保存。現在の生成はDirector本体の全体seedです。",
            style: { minWidth: "0", color: "inherit", background: "#171717", border: "1px solid #555", borderRadius: "4px", padding: "4px" },
            onchange: (event) => {
                image.transition.seed = event.target.value.trim();
                sync();
            },
        }));
        wrap.append(settings);
        return wrap;
    }

    function render() {
        const preservedInput = fileInput;
        root.replaceChildren(preservedInput);
        const toolbar = el("div", {
            style: { display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px", position: "sticky", top: "0", zIndex: "2", background: "#222", padding: "4px" },
        });
        toolbar.append(el("button", {
            text: "＋ 画像を追加",
            onclick: () => fileInput.click(),
            style: { cursor: "pointer", padding: "5px 10px" },
        }));
        const loop = el("input", {
            type: "checkbox",
            checked: state.loop,
            onchange: (event) => {
                state.loop = event.target.checked;
                sync();
                render();
            },
        });
        toolbar.append(el("label", { style: { display: "flex", alignItems: "center", gap: "4px" } }, [loop, document.createTextNode("最後→最初を追加（ループ）")]));
        const segmentCount = Math.max(0, state.images.length - 1) + (state.loop && state.images.length >= 2 ? 1 : 0);
        let totalFrames = state.images.slice(0, Math.max(0, state.images.length - 1))
            .reduce((sum, image) => sum + alignedFrameCount(image.transition.duration_sec), 0);
        if (state.loop && state.images.length >= 2) {
            totalFrames += alignedFrameCount(state.images[state.images.length - 1].transition.duration_sec);
        }
        const totalDuration = totalFrames / STORY_FPS;
        const overDurationLimit = totalDuration > MAX_TOTAL_DURATION;
        toolbar.append(el("span", {
            text: `${state.images.length}枚 / ${segmentCount}区間 / ${totalDuration.toFixed(1)}秒（上限${MAX_TOTAL_DURATION}秒）`,
            title: "24fps・17k+5整列後の尺。Directorの全区間tensor保持によるRAM超過を防ぐ安全上限です。",
            style: { marginLeft: "auto", opacity: ".9", color: overDurationLimit ? "#ff8b8b" : "inherit" },
        }));
        root.append(toolbar);

        if (node.__h3OrderedStoryboardError) {
            root.append(el("div", { text: node.__h3OrderedStoryboardError, style: { color: "#ff8b8b", padding: "5px", whiteSpace: "pre-wrap" } }));
        }
        if (!state.images.length) {
            root.append(el("div", {
                text: "画像を2枚以上追加してください。順番は ↑ ↓ で自由に変更できます。",
                style: { padding: "18px 8px", textAlign: "center", opacity: ".75" },
            }));
        }

        state.images.forEach((image, index) => {
            const card = el("div", {
                style: { border: "1px solid #484848", borderRadius: "6px", padding: "6px", marginBottom: "6px", background: "rgba(0,0,0,.12)" },
            });
            const header = el("div", { style: { display: "flex", alignItems: "center", gap: "7px" } });
            header.append(el("span", { text: `#${index + 1}`, style: { width: "28px", fontWeight: "700" } }));
            header.append(el("img", {
                src: imageViewUrl(image),
                alt: image.name,
                style: { width: "58px", height: "42px", objectFit: "cover", borderRadius: "3px", background: "#111" },
            }));
            header.append(el("span", { text: image.subfolder ? `${image.subfolder}/${image.name}` : image.name, title: image.name, style: { flex: "1", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" } }));
            const move = (delta) => {
                const next = index + delta;
                if (next < 0 || next >= state.images.length) return;
                [state.images[index], state.images[next]] = [state.images[next], state.images[index]];
                sync();
                render();
            };
            header.append(el("button", { text: "↑", title: "上へ", disabled: index === 0, onclick: () => move(-1), style: { cursor: "pointer" } }));
            header.append(el("button", { text: "↓", title: "下へ", disabled: index === state.images.length - 1, onclick: () => move(1), style: { cursor: "pointer" } }));
            header.append(el("button", {
                text: "−",
                title: "削除",
                onclick: () => {
                    state.images.splice(index, 1);
                    sync();
                    render();
                },
                style: { cursor: "pointer", color: "#ff9b9b" },
            }));
            card.append(header);
            if (index < state.images.length - 1 || (state.loop && state.images.length >= 2)) {
                card.append(transitionEditor(image, index));
            }
            root.append(card);
        });
        node.setSize?.([Math.max(node.size?.[0] || 0, 560), Math.max(node.size?.[1] || 0, 520)]);
        node.graph?.setDirtyCanvas?.(true, true);
    }

    node.addDOMWidget("ordered_storyboard_editor", "div", root, {
        serialize: false,
        hideOnZoom: false,
    });
    node.__h3OrderedStoryboardRefresh = () => {
        state = normalizeClientState(stateWidget.value);
        render();
    };
    sync();
    render();
}

app.registerExtension({
    name: "MiniMaxH3.OrderedStoryboard",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            installEditor(this);
            return result;
        };

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            setTimeout(() => this.__h3OrderedStoryboardRefresh?.(), 0);
            return result;
        };
    },
});
