#!/usr/bin/env python3
"""
Merge two transcription passes of the same notes (e.g. whisper small and medium) into a
consensus transcript. Words both passes agree on are kept. Where they disagree the span is
written as [unclear: "pass A heard" / "pass B heard"] so nothing is silently guessed.

Usage: ./venv/bin/python consensus.py --primary <dir-with-transcripts> --secondary <dir-with-transcripts> --out transcripts
Rebuilds ALL_TRANSCRIPTS.md in --out afterwards.
"""
import argparse
import difflib
import json
import re
from pathlib import Path

import transcribe as T

WORD = re.compile(r"\S+")


def norm(w):
    return re.sub(r"[^\w']", "", w.lower())


def merge_text(a: str, b: str):
    wa, wb = WORD.findall(a), WORD.findall(b)
    sm = difflib.SequenceMatcher(a=[norm(w) for w in wa], b=[norm(w) for w in wb], autojunk=False)
    out, n_unclear = [], 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out.extend(wa[i1:i2])
        else:
            n_unclear += 1
            out.append(f'[unclear: "{" ".join(wa[i1:i2]) or "-"}" / "{" ".join(wb[j1:j2]) or "-"}"]')
    return " ".join(out), n_unclear


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, help="dir containing transcripts/ from the better model")
    ap.add_argument("--secondary", required=True, help="dir containing transcripts/ from the other model")
    ap.add_argument("--out", default="transcripts")
    ap.add_argument("--notes-dir", default=".", help="where the .opus and _chat.txt live")
    args = ap.parse_args()

    pdir, sdir, out = Path(args.primary) / "transcripts", Path(args.secondary) / "transcripts", Path(args.out)
    out.mkdir(exist_ok=True)
    notes = sorted(Path(args.notes_dir).glob("*.opus"), key=T.note_time)
    for src in notes:
        pj, sj = pdir / (src.stem + ".json"), sdir / (src.stem + ".json")
        if not (pj.exists() and sj.exists()):
            T.log(f"{src.name}: missing a pass, skipped")
            continue
        P, S = json.loads(pj.read_text()), json.loads(sj.read_text())
        segs, total_unclear = [], 0
        ps, ss = P["segments"], S["segments"]
        if len(ps) != len(ss):            # chunk boundaries differ: merge whole-note text instead
            ps = [{"start": ps[0]["start"], "end": ps[-1]["end"], "text": " ".join(s["text"] for s in ps)}] if ps else []
            ss = [{"start": 0, "end": 0, "text": " ".join(s["text"] for s in ss)}] if ss else []
        for p, s in zip(ps, ss):
            text, n = merge_text(p["text"], s["text"])
            total_unclear += n
            segs.append({"start": p["start"], "end": p["end"], "text": text, "unclear": False})  # inline markers carry it
        meta = dict(P["meta"])
        meta["model"] = f"{P['meta']['model']}+{S['meta']['model']} consensus"
        meta["unclear_note"] = (f"{total_unclear} span(s) where the two models disagree are written as "
                                f"[unclear: \"{P['meta']['model']} heard\" / \"{S['meta']['model']} heard\"]")
        T.write_note_transcript(out, src, segs, meta)
        T.log(f"{src.name}: {total_unclear} unclear span(s)")
    T.build_all(notes, out, T.load_chat(Path(args.notes_dir) / "_chat.txt"), T.DEFAULT_ME)
    T.log(f"wrote {out / 'ALL_TRANSCRIPTS.md'}")


if __name__ == "__main__":
    main()
