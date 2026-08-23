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
    const element = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
        if (key === "className") element.className = value;
        else if (key === "text") element.textContent = value;
        else if (key === "style") Object.assign(element.style, value);
        else if (key.startsWith("on") && typeof value === "function") {
            element.addEventListener(key.slice(2).toLowerCase(), value);
        } else if (value !== undefined) {
            element[key] = value;
        }
    }
    for (const child of children) element.append(child);
    return element;
}

const inputStyle = {
    boxSizing: "border-box",
    minWidth: "0",
    color: "inherit",
    background: "#15171b",
    border: "1px solid #515866",
    borderRadius: "6px",
    padding: "7px",
};

const buttonStyle = {
    cursor: "pointer",
    color: "#e8edf7",
    background: "#303744",
    border: "1px solid #566174",
    borderRadius: "6px",
    padding: "6px 10px",
};

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
            minWidth: "700px",
            maxHeight: "850px",
            overflowY: "auto",
            padding: "10px",
            color: "var(--fg-color, #e3e7ef)",
            background: "#202329",
            font: "13px/1.4 sans-serif",
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
        node.__h3OrderedStoryboardError = "";
        for (const file of files) {
            const body = new FormData();
            body.append("image", file, file.name);
            body.append("type", "input");
            body.append("subfolder", INPUT_SUBFOLDER);
            body.append("overwrite", "false");
            const response = await api.fetchApi("/upload/image", { method: "POST", body });
            if (!response.ok) {
                const detail = await response.text();
                throw new Error(`アップロード失敗 (${response.status}): ${detail}`);
            }
            const asset = await response.json();
            state.images.push({
                id: freshId(),
                name: String(asset.name || file.name),
                subfolder: String(asset.subfolder || INPUT_SUBFOLDER),
                type: "input",
                transition: {
                    prompt: state.defaults.prompt,
                    duration_sec: state.defaults.duration_sec,
                    seed: String(seedBigInt(state.defaults.seed) + BigInt(state.images.length)),
                },
            });
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
                margin: "10px 0 2px 76px",
                padding: "10px",
                borderLeft: "3px solid #6aa3ff",
                borderRadius: "0 7px 7px 0",
                background: "rgba(72, 116, 184, .12)",
            },
        });
        wrap.append(el("div", {
            text: `区間 ${index + 1}: 画像 #${index + 1} → #${targetIndex + 1}`,
            style: { marginBottom: "7px", color: "#a9ceff", fontWeight: "700" },
        }));
        wrap.append(el("textarea", {
            value: image.transition.prompt,
            placeholder: "この区間で起きる動作・カメラ・音を記述。境界で止めず、次へ続く動きを明記します。",
            rows: 4,
            style: { ...inputStyle, width: "100%", minHeight: "92px", resize: "vertical", lineHeight: "1.5" },
            oninput: (event) => {
                image.transition.prompt = event.target.value;
                sync();
            },
        }));

        const settings = el("div", {
            style: {
                display: "grid",
                gridTemplateColumns: "70px minmax(90px, 1fr) 150px minmax(150px, 1.5fr)",
                gap: "7px",
                marginTop: "8px",
                alignItems: "center",
            },
        });
        settings.append(el("label", { text: "秒数" }));
        settings.append(el("input", {
            type: "number",
            value: image.transition.duration_sec,
            min: 0.2,
            max: 15,
            step: 0.1,
            style: inputStyle,
            onchange: (event) => {
                const value = Number(event.target.value);
                if (Number.isFinite(value)) image.transition.duration_sec = value;
                sync();
                render();
            },
        }));
        settings.append(el("label", {
            text: "区間Seed（保存用）",
            title: "現行Directorは全区間でDirectorノード本体のseedを使います。",
        }));
        settings.append(el("input", {
            type: "text",
            inputMode: "numeric",
            value: image.transition.seed,
            pattern: "[0-9]*",
            title: "将来の区間別seed対応用。現在の生成seedはDirectorノードで変更します。",
            style: inputStyle,
            onchange: (event) => {
                image.transition.seed = event.target.value.trim();
                sync();
            },
        }));
        wrap.append(settings);
        return wrap;
    }

    function render() {
        root.replaceChildren(fileInput);
        const toolbar = el("div", {
            style: {
                display: "flex",
                flexWrap: "wrap",
                alignItems: "center",
                gap: "9px",
                marginBottom: "10px",
                position: "sticky",
                top: "0",
                zIndex: "3",
                background: "#202329",
                borderBottom: "1px solid #414754",
                padding: "6px 4px 10px",
            },
        });
        toolbar.append(el("button", {
            text: "＋ 画像を追加",
            onclick: () => fileInput.click(),
            style: { ...buttonStyle, background: "#315f9e", borderColor: "#5f96de", fontWeight: "700" },
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
        toolbar.append(el("label", {
            title: "最後の画像から最初の画像へ戻る区間も生成します。",
            style: { display: "flex", alignItems: "center", gap: "5px", fontWeight: "600" },
        }, [loop, document.createTextNode("Loop（最後 → 最初）")]));

        const segmentCount = Math.max(0, state.images.length - 1) + (state.loop && state.images.length >= 2 ? 1 : 0);
        let totalFrames = state.images.slice(0, Math.max(0, state.images.length - 1))
            .reduce((sum, image) => sum + alignedFrameCount(image.transition.duration_sec), 0);
        if (state.loop && state.images.length >= 2) {
            totalFrames += alignedFrameCount(state.images[state.images.length - 1].transition.duration_sec);
        }
        const totalDuration = totalFrames / STORY_FPS;
        const overDurationLimit = totalDuration > MAX_TOTAL_DURATION;
        toolbar.append(el("span", {
            text: `${state.images.length}枚 · ${segmentCount}区間 · ${totalDuration.toFixed(1)}秒 / 上限${MAX_TOTAL_DURATION}秒`,
            title: "24fps・17k+5整列後。DirectorのRAM保護上限です。",
            style: { marginLeft: "auto", color: overDurationLimit ? "#ff8b8b" : "#c9d3e3", fontWeight: "600" },
        }));
        root.append(toolbar);

        const help = el("details", {
            style: { marginBottom: "10px", padding: "8px 10px", background: "#191c21", borderRadius: "7px" },
        });
        help.append(el("summary", { text: "使い方 / Motion Context", style: { cursor: "pointer", fontWeight: "700" } }));
        help.append(el("div", {
            text: "上から順に画像を通過します。各プロンプトには『ポーズで停止せず動作とカメラの勢いを次へ継続』と書くのが有効です。前区間のlatent＋音声22フレームは自動で次区間へ継承されます。",
            style: { marginTop: "7px", color: "#b9c5d8" },
        }));
        root.append(help);

        if (node.__h3OrderedStoryboardError) {
            root.append(el("div", {
                text: node.__h3OrderedStoryboardError,
                style: { color: "#ff9e9e", background: "#482629", borderRadius: "6px", padding: "8px", marginBottom: "8px", whiteSpace: "pre-wrap" },
            }));
        }
        if (!state.images.length) {
            root.append(el("div", {
                text: "画像を2枚以上追加してください。追加後に ↑ / ↓ で順番を変更できます。",
                style: { padding: "34px 12px", textAlign: "center", color: "#aeb8c8", border: "1px dashed #596273", borderRadius: "8px" },
            }));
        }

        state.images.forEach((image, index) => {
            const card = el("div", {
                style: { border: "1px solid #4a5260", borderRadius: "8px", padding: "9px", marginBottom: "10px", background: "#242831" },
            });
            const header = el("div", { style: { display: "flex", alignItems: "center", gap: "9px", minHeight: "58px" } });
            header.append(el("span", { text: `#${index + 1}`, style: { width: "34px", fontSize: "15px", fontWeight: "800", color: "#a9ceff" } }));
            header.append(el("img", {
                src: imageViewUrl(image),
                alt: image.name,
                style: { width: "76px", height: "54px", objectFit: "cover", borderRadius: "5px", background: "#111", border: "1px solid #555e6c" },
            }));
            header.append(el("span", {
                text: image.subfolder ? `${image.subfolder}/${image.name}` : image.name,
                title: image.name,
                style: { flex: "1", minWidth: "0", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontWeight: "600" },
            }));
            const move = (delta) => {
                const next = index + delta;
                if (next < 0 || next >= state.images.length) return;
                [state.images[index], state.images[next]] = [state.images[next], state.images[index]];
                sync();
                render();
            };
            header.append(el("button", { text: "↑", title: "上へ移動", disabled: index === 0, onclick: () => move(-1), style: buttonStyle }));
            header.append(el("button", { text: "↓", title: "下へ移動", disabled: index === state.images.length - 1, onclick: () => move(1), style: buttonStyle }));
            header.append(el("button", {
                text: "削除",
                title: "この画像を削除",
                onclick: () => {
                    state.images.splice(index, 1);
                    sync();
                    render();
                },
                style: { ...buttonStyle, color: "#ffb1b1", background: "#43282d", borderColor: "#70434b" },
            }));
            card.append(header);
            if (index < state.images.length - 1 || (state.loop && state.images.length >= 2)) {
                card.append(transitionEditor(image, index));
            } else {
                card.append(el("div", {
                    text: "終点（Loop OFFのため、この画像から次の区間は生成しません）",
                    style: { margin: "8px 0 2px 76px", color: "#919bab" },
                }));
            }
            root.append(card);
        });
        node.setSize?.([Math.max(node.size?.[0] || 0, 760), Math.max(node.size?.[1] || 0, 860)]);
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
