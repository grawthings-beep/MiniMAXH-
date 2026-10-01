// DOM contract tests only; not a live ComfyUI/browser playback benchmark.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

class Element {
    constructor(tag) { this.tag = tag; this.style = {}; this.children = []; this.handlers = {}; this.readyState = 3; this.currentTime = 0; this.duration = 5; this.paused = true; }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
    addEventListener(name, handler) { this.handlers[name] = handler; }
    removeEventListener(name) { delete this.handlers[name]; }
    removeAttribute(name) { delete this[name]; }
    load() {}
    pause() { this.paused = true; }
    async play() { this.paused = false; }
}
const extensions = [];
const app = { registerExtension: e => extensions.push(e), graph: { setDirtyCanvas() {} } };
const source = (await readFile(new URL("../custom_nodes/minimax_h3_ordered_storyboard/web/upscale_compare.js", import.meta.url), "utf8"))
    .replace(/^import .*;\r?\n/gm, "").replace("export function", "function");
vm.runInNewContext(source, { app, api: { apiURL: url => url }, document: { createElement: tag => new Element(tag) },
    URLSearchParams, setTimeout, clearTimeout, console });
class CompareNode {
    constructor() { this.size = [1200, 1350]; this.widgetCount = 0; }
    addDOMWidget(name, type, root, options) { assert.equal(options.serialize, false); this.dom = root; this.widgetCount++; return {}; }
    setSize(size) { this.size = size; }
}
await extensions[0].beforeRegisterNodeDef(CompareNode, { name: "MiniMaxH3CompareUpscale" });
const node = new CompareNode();
const find = (root, tag) => [root, ...root.children.flatMap(n => find(n, tag))].filter(n => n.tag === tag);
const message = {
    comparison_files: ["A_X2_VAE", "B_INT8_SPAN", "C_INT8_AnimeSharp"].map(label => ({ label, filename: label + " &?.mp4", subfolder: "video", type: "output", seconds: { total: 1.23 } })),
    comparison_report: [{ filename: "report.json", subfolder: "video", type: "output" }],
};
node.onExecuted(message);
let videos = find(node.dom, "video");
assert.equal(videos.length, 3);
assert.deepEqual(videos.map(v => v.muted), [false, true, true]);
assert.match(videos[0].src, /%26%3F/);
assert.equal(find(node.dom, "a").length, 4);
await find(node.dom, "button")[0].onclick();
assert(videos.every(v => !v.paused));
videos[0].currentTime = 1.5;
videos[0].handlers.timeupdate();
assert.equal(videos[2].currentTime, 1.5);
const slider = find(node.dom, "input")[0];
slider.value = "500";
slider.oninput();
assert(videos.every(v => v.paused && v.currentTime === 2.5));
node.onExecuted(message);
assert.equal(node.widgetCount, 1);
assert(videos.every(v => !v.src && v.paused));
videos = find(node.dom, "video");
node.onRemoved();
assert(videos.every(v => !v.src && v.paused));
console.log("Comparison UI fixtures passed: three previews, one audio, safe links, shared playback/seek, rerun/removal cleanup");
