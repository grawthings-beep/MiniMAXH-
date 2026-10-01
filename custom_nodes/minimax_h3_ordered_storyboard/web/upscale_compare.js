import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const viewURL = (file) => api.apiURL(`/view?${new URLSearchParams({
    filename: file.filename, subfolder: file.subfolder || "", type: file.type || "output",
})}`);

export function createComparisonPanel(files, report) {
    const root = document.createElement("div");
    Object.assign(root.style, { background: "#17191e", color: "#eee", padding: "12px", overflow: "auto" });
    const bar = document.createElement("div");
    const grid = document.createElement("div");
    Object.assign(grid.style, { display: "grid", gridTemplateColumns: "repeat(3,minmax(0,1fr))", gap: "12px", marginTop: "12px" });
    const videos = [];
    for (const [index, file] of files.entries()) {
        const card = document.createElement("div");
        const title = document.createElement("div");
        title.textContent = `${file.label} · ${Number(file.seconds?.total || 0).toFixed(1)} s`;
        const video = document.createElement("video");
        video.src = viewURL(file);
        video.preload = "metadata";
        video.playsInline = true;
        video.muted = index !== 0; // The same audio is decoded once; hear it only once.
        video.controls = false;
        Object.assign(video.style, { width: "100%", maxHeight: "800px", background: "#000", objectFit: "contain" });
        const download = document.createElement("a");
        download.href = viewURL(file);
        download.download = file.filename;
        download.textContent = "MP4を保存 / 原寸で比較";
        download.style.color = "#8cd8ff";
        card.append(title, video, download);
        grid.append(card);
        videos.push(video);
    }
    const pause = () => videos.forEach(v => v.pause());
    let generation = 0;
    const seek = (time) => videos.forEach(v => { if (v.readyState >= 1) v.currentTime = time; });
    const ready = (v) => v.readyState >= 3 ? Promise.resolve() : new Promise((resolve, reject) => {
        const timer = setTimeout(() => finish(new Error("Video load timeout")), 15000);
        const loaded = () => finish();
        const failed = () => finish(new Error("Video unavailable"));
        function finish(error) {
            clearTimeout(timer);
            v.removeEventListener("canplay", loaded);
            v.removeEventListener("error", failed);
            error ? reject(error) : resolve();
        }
        v.addEventListener("canplay", loaded);
        v.addEventListener("error", failed);
        v.load();
    });
    const status = document.createElement("span");
    const button = (label, fn) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.onclick = fn;
        bar.append(b);
    };
    button("3本を先頭から再生", async () => {
        const token = ++generation;
        pause();
        status.textContent = " 読込中…";
        try {
            await Promise.all(videos.map(ready));
            if (token !== generation) return;
            seek(0);
            await Promise.all(videos.map(v => v.play()));
            if (token !== generation) { pause(); return; }
            status.textContent = " 音声はAのみ / ブラウザ同期は近似";
        } catch {
            pause();
            status.textContent = " 再生できません。MP4保存リンクで確認してください。";
        }
    });
    button("停止", () => { generation++; pause(); });
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "1000";
    slider.value = "0";
    slider.title = "3本を同じ時刻へシーク（停止して比較）";
    slider.oninput = () => {
        generation++;
        pause();
        if (Number.isFinite(videos[0]?.duration)) seek(videos[0].duration * Number(slider.value) / 1000);
    };
    videos[0]?.addEventListener("timeupdate", () => {
        const first = videos[0];
        if (first.duration) slider.value = String(first.currentTime / first.duration * 1000);
        if (!first.paused) {
            for (const other of videos.slice(1)) {
                if (Math.abs(other.currentTime - first.currentTime) > 0.08) other.currentTime = first.currentTime;
            }
        }
    });
    videos[0]?.addEventListener("ended", pause);
    bar.append(slider, status);
    if (report) {
        const link = document.createElement("a");
        link.href = viewURL(report);
        link.download = report.filename;
        link.textContent = " 処理時間JSON";
        link.style.color = "#8cd8ff";
        bar.append(link);
    }
    const notice = document.createElement("p");
    notice.textContent = "同一latent・音声・縦横2倍・H264 CRF18。表示は縮小。原寸MP4で顔・髪・輪郭・ちらつきを確認。AnimeSharp: CC-BY-NC-SA-4.0（非商用）。";
    root.append(bar, notice, grid);
    return { root, dispose() { generation++; pause(); videos.forEach(v => { v.removeAttribute("src"); v.load(); }); } };
}

app.registerExtension({
    name: "MiniMaxH3.UpscaleCompare",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MiniMaxH3CompareUpscale") return;
        const previous = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            previous?.apply(this, arguments);
            if (!message?.comparison_files?.length) return;
            this.h3ComparePanel?.dispose();
            const panel = createComparisonPanel(message.comparison_files, message.comparison_report?.[0]);
            if (!this.h3CompareWidget) {
                const host = document.createElement("div");
                this.h3CompareWidget = this.addDOMWidget("h3Compare", "div", host, { serialize: false });
                this.h3CompareHost = host;
                this.h3CompareWidget.computeSize = () => [1100, 950];
            }
            this.h3CompareHost.replaceChildren(panel.root);
            this.h3ComparePanel = panel;
            this.setSize([Math.max(this.size[0], 1200), Math.max(this.size[1], 1350)]);
            app.graph.setDirtyCanvas(true, true);
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function() {
            this.h3ComparePanel?.dispose();
            return removed?.apply(this, arguments);
        };
    },
});
