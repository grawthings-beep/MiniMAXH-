import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { pairFiles } from "./pair_files.mjs";

const label = "サンプル2 → アップスケール 一括生成";
let dialog;
function el(tag, text) {
    const node = document.createElement(tag);
    if (text != null) node.textContent = text;
    return node;
}
async function json(path, options) {
    const r = await api.fetchApi(path, options);
    if (!r.ok) throw new Error((await r.text()).slice(0, 1800) || r.statusText);
    return r.json();
}

function openBatch() {
    if (dialog) { if (!dialog.open) dialog.showModal(); return; }
    dialog = el("dialog");
    dialog.style.cssText = "width:min(740px,90vw);max-height:82vh;overflow:auto;padding:20px;background:#262a30;color:#eee;border:1px solid #777";
    const title = el("h3", label);
    const help = el("p", "① Preview #2 と Final Upscale Pass をON、Single PassをOFF。②「ファイル」から例：ラピ.png と ラピ.txt を全件まとめて選択。③ 生成開始。Pod受付まではSafariを開いてください。");
    const add = el("input"); add.type = "file"; add.multiple = true; add.accept = ".png,.jpg,.jpeg,.webp,.txt";
    const list = el("div"), start = el("button", "登録した順に生成"), close = el("button", "閉じる");
    const status = el("pre"); status.style.whiteSpace = "pre-wrap";
    const last = el("button", "直近の一括生成の進捗");
    const jobs = [];
    let timer, busy = false;
    const setLocked = on => {
        busy = on; add.disabled = start.disabled = on;
        for (const j of jobs) j.prompt.disabled = j.remove.disabled = on;
    };
    close.onclick = () => dialog.close();
    add.onchange = async () => {
        setLocked(true); last.disabled = true;
        try {
            const pairs = await pairFiles(Array.from(add.files), jobs.map(j => j.file.name));
            for (const {file, text} of pairs) {
                const box = el("section"), name = el("strong", file.name), prompt = el("textarea"), remove = el("button", "外す");
                prompt.rows = 3; prompt.placeholder = "この画像のプロンプト（全文）"; prompt.value = text;
                prompt.style.cssText = "display:block;width:100%;box-sizing:border-box;margin:4px 0 12px";
                const job = {file, prompt, remove};
                remove.onclick = () => { jobs.splice(jobs.indexOf(job), 1); box.remove(); };
                box.append(name, remove, prompt); list.append(box); jobs.push(job);
            }
            status.textContent = pairs.length + "件を追加しました。合計 " + jobs.length + " 件。";
        } catch (err) { status.textContent = err.message; }
        finally { add.value = ""; setLocked(false); last.disabled = false; }
    };
    const observe = async id => {
        clearTimeout(timer);
        const state = await json("/foxhunter/batch/" + id);
        status.textContent = (state.running ? "Pod受付済み。Safariは閉じても構いません。\n" : "一括処理終了\n") +
            state.rows.map((r, i) => (i+1) + ". " + r.label + "　" + r.state + "　" + r.detail).join("\n");
        if (state.running) {
            setLocked(true);
            timer = setTimeout(() => observe(state.id).catch(err => {
                status.textContent += "\n進捗を取得できません。再送せず「直近の一括生成の進捗」で確認：" + err.message;
            }), 2500);
        } else setLocked(false);
    };
    last.onclick = async () => {
        const wasBusy = busy;
        last.disabled = true; setLocked(true);
        try { await observe("current"); }
        catch (e) { status.textContent = e.message; setLocked(wasBusy); }
        finally { last.disabled = false; }
    };
    start.onclick = async () => {
        if (busy) return;
        if (!jobs.length || jobs.length > 100 || jobs.some(j => !j.prompt.value.trim())) {
            status.textContent = "画像1〜100枚と、それぞれのプロンプトを入力してください。"; return;
        }
        setLocked(true); last.disabled = true;
        let submitted = false;
        try {
            const {output: graph} = await app.graphToPrompt();
            if (!graph["133:129"] || !graph["135:26"] || !graph["604"]) {
                throw new Error("FoxHunterの Preview #2 と Final Upscale Pass をON、Single PassをOFFにしてください。");
            }
            const rows = [];
            for (const job of jobs) {
                status.textContent = "画像を登録中 " + (rows.length+1) + " / " + jobs.length;
                const body = new FormData();
                body.append("image", job.file, job.file.name);
                body.append("subfolder", "foxhunter_batch/" + crypto.randomUUID());
                const saved = await json("/upload/image", {method:"POST", body});
                rows.push({image:[saved.subfolder, saved.name].filter(Boolean).join("/"),
                           label:job.file.name, prompt:job.prompt.value.trim()});
            }
            const {id} = await json("/foxhunter/batch", {method:"POST", headers:{"Content-Type":"application/json"},
                                                         body:JSON.stringify({rows, graph})});
            submitted = true;
            list.replaceChildren(); jobs.length = 0;
            status.textContent = "Podで開始しました。このタブは閉じても構いません。";
            last.disabled = false;
            await observe(id);
        } catch (e) {
            status.textContent = e.message + "\n通信が途切れた場合は、再登録前に「直近の一括生成の進捗」で確認してください。";
            if (!submitted) setLocked(false);
            last.disabled = false;
        }
    };
    dialog.append(title, help, add, list, start, last, close, status);
    document.body.append(dialog); dialog.showModal();
}

app.registerExtension({
    name:"FoxHunter.Sample2Batch",
    setup() {
        if (!window.matchMedia("(pointer: coarse)").matches || document.getElementById("fox-batch-open")) return;
        const launch = el("button", "画像を一括登録");
        launch.id = "fox-batch-open";
        launch.style.cssText = "position:fixed;right:12px;bottom:calc(12px + env(safe-area-inset-bottom));z-index:90;padding:10px 14px;border-radius:10px;color:white;background:#384759";
        launch.onclick = openBatch;
        document.body.append(launch);
    },
    commands:[{id:"foxhunter.sample2Batch", label, function:openBatch}],
    menuCommands:[{path:["FoxHunter"], commands:["foxhunter.sample2Batch"]}],
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3PromptIDE") return;
        const previous = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(canvas, options) {
            const ret = previous?.apply(this, arguments);
            options.unshift({content: label, callback: openBatch});
            return ret;
        };
    }
});
