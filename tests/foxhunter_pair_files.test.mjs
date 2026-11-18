import assert from "node:assert/strict";
import test from "node:test";
import {pairFiles} from "../custom_nodes/foxhunter_compat/web/pair_files.mjs";

function file(name, data = "") {
    const blob = new Blob([data]);
    return Object.assign(blob, {name});
}

test("mixed picker order, unicode, multiline UTF-8 BOM", async () => {
    const rows = await pairFiles([file("B.txt","second"), file("ラピ.png"), file("b.WeBP"),
        file("ラピ".normalize("NFD") + ".TXT","\uFEFFfirst\nline two\n")]);
    assert.deepEqual(rows.map(r=>[r.file.name,r.text]), [["ラピ.png","first\nline two"],["b.WeBP","second"]]);
});
test("mismatch, duplicates, invalid encoding or size reject the complete selection", async () => {
    for (const files of [
        [file("a.txt","p")], [file("a.png"),file("a.txt","p"),file("b.jpg")],
        [file("a.png"),file("A.jpg")], [file("a.png"),file("a.txt","p"),file("A.TXT","p")],
        [file("a.png"),file("a.txt","  ")], [file("a.png"),file("a.txt",new Uint8Array([255]))],
        [file("a.png"),file("a.txt","p".repeat(131073))], [file("a.heic")]
    ]) await assert.rejects(pairFiles(files));
    await assert.rejects(pairFiles([file("A.png")], ["a.jpg"]));
});
test("manual image-only input remains available and total is capped", async () => {
    assert.equal((await pairFiles([file("a.png")]))[0].text, "");
    await assert.rejects(pairFiles([file("next.png")], Array.from({length:100}, (_,i)=>i+".png")));
});
