// DOM/API fixtures for upload, restore, reorder, remove, and download contracts.
// No ComfyUI server or browser is impersonated; visual GPU UI QA is separate.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

class Element {
    constructor(tag) { this.tag = tag; this.style = {}; this.children = []; this.textContent = ""; }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
    click() { this.onclick?.(); }
}
const extensions = [];
let uploads = 0;
const app = { registerExtension: e => extensions.push(e) };
const api = {
    apiURL: path => path,
    fetchApi: async (path, { body }) => {
        assert.equal(path, "/upload/image");
        assert.equal(body.values.get("overwrite"), "false");
        uploads++;
        return { ok: true, json: async () => ({ name: body.values.get("image").name, subfolder: "minimax_h3_character" }) };
    },
};
class FormData { constructor() { this.values = new Map(); } append(k, v) { this.values.set(k, v); } }
const source = (await readFile(new URL("../custom_nodes/minimax_h3_ordered_storyboard/web/refmod_images.js", import.meta.url), "utf8"))
    .replace(/^import .*;\r?\n/gm, "");
vm.runInNewContext(source, { app, api, document: { createElement: tag => new Element(tag) },
    URLSearchParams, FormData, setTimeout, console });
class ImageNode {
    constructor(value = "[]") { this.widgets = [{ name: "images_json", value }]; this.graph = { setDirtyCanvas() {} }; }
    addDOMWidget(name, type, root, options) { this.dom = root; assert.equal(options.serialize, false); }
}
const extension = extensions[0];
await extension.beforeRegisterNodeDef(ImageNode, { name: "MiniMaxH3RefModImages" });
const node = new ImageNode(); node.onNodeCreated();
assert.equal(node.widgets[0].hidden, true);
assert.equal(node.widgets[0].type, "hidden");
const find = (root, predicate) => [root, ...root.children.flatMap(n => find(n, predicate))].filter(predicate);
const textButton = text => find(node.dom, n => n.tag === "button" && n.textContent === text)[0];
const picker = find(node.dom, n => n.tag === "input")[0];
picker.files = [{ name: "face.png" }, { name: "body.jpg" }, { name: "side.webp" }];
await picker.onchange();
assert.equal(uploads, 3);
assert.equal(JSON.parse(node.widgets[0].value).length, 3);
textButton("↓").click();
assert.match(JSON.parse(node.widgets[0].value)[0], /body.jpg$/);
textButton("外す").click();
assert.equal(JSON.parse(node.widgets[0].value).length, 2);
assert.equal(find(node.dom, n => n.tag === "img")[0].alt, "Picture 1");
const restored = new ImageNode(node.widgets[0].value); restored.onNodeCreated(); restored.onConfigure();
await new Promise(resolve => setTimeout(resolve, 5));
assert.equal(find(restored.dom, n => n.tag === "img").length, 2);
const before = uploads;
picker.files = Array.from({ length: 8 }, (_, i) => ({ name: `${i}.png` }));
await picker.onchange();
assert.equal(uploads, before);
assert.match(find(node.dom, n => n.textContent.includes("最大8枚"))[0].textContent, /最大8枚/);
class SaveNode { addDOMWidget(name, type, root) { this.dom = root; } }
await extension.beforeRegisterNodeDef(SaveNode, { name: "MiniMaxH3SaveCharacterRefMod" });
const save = new SaveNode();
save.onExecuted({ refmod_files: [{ filename: "character_123.safetensors", subfolder: "refmods", type: "output" }] });
assert.equal(save.dom.children[0].download, "character_123.safetensors");
assert.match(save.dom.children[0].href, /type=output/);
console.log("RefMod UI fixtures passed: multi-upload, order, removal, restore, count limit, output download");
