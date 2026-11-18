// Pair locally; only image files and decoded prompt strings go to the Pod.
export async function pairFiles(files, existing = []) {
    const images = new Map(), texts = new Map();
    const key = name => name.slice(0, name.lastIndexOf(".")).normalize("NFC").toLowerCase();
    const used = new Set(existing.map(key));
    for (const file of files) {
        const ext = /\.(png|jpe?g|webp|txt)$/i.exec(file.name)?.[1]?.toLowerCase();
        if (!ext || !key(file.name)) throw Error(file.name + "：PNG/JPG/WEBP画像とUTF-8の.txtを選んでください");
        const map = ext === "txt" ? texts : images;
        const stem = key(file.name);
        if (map.has(stem) || used.has(stem)) throw Error(file.name + "：同名のファイルまたは登録済みの画像があります");
        map.set(stem, file);
    }
    if (images.size + existing.length > 100) throw Error("画像は合計100枚までです");
    for (const [stem, file] of texts) {
        if (!images.has(stem)) throw Error(file.name + " に対応する画像を一緒に選んでください");
    }
    const rows = [];
    for (const [stem, file] of images) {
        const sidecar = texts.get(stem);
        if (!sidecar && texts.size) throw Error(file.name + " に対応する .txt がありません");
        let text = "";
        if (sidecar) {
            if (sidecar.size > 128 * 1024) throw Error(sidecar.name + "：128 KiB以下のテキストを指定してください");
            try { text = new TextDecoder("utf-8", {fatal:true}).decode(await sidecar.arrayBuffer()).trim(); }
            catch { throw Error(sidecar.name + " をUTF-8として読み込めません"); }
            if (!text) throw Error(sidecar.name + " が空です");
        }
        rows.push({file, text});
    }
    return rows;
}
