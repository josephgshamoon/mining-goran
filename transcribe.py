#!/usr/bin/env python3
"""
Local voice-note transcription pipeline for the MrCrypto Mining due diligence.

Usage:
  ./venv/bin/python transcribe.py            # transcribe new .opus files, rebuild ALL_TRANSCRIPTS.md
  ./venv/bin/python transcribe.py --force    # re-transcribe everything
  ./venv/bin/python transcribe.py --print    # also print ALL_TRANSCRIPTS.md to the terminal
  ./venv/bin/python transcribe.py --model medium --backend faster-whisper

Backends (auto order): faster-whisper (needs huggingface.co reachable once to fetch the
model) -> sherpa-onnx Whisper (model from GitHub releases, kept under ./models/).
Nothing here sends data anywhere; model downloads are the only network use.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TS_RE = re.compile(r"AUDIO-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})")
DEFAULT_ME = {"00000052"}          # note numbers that are JS; everything else is Goran
SPEAKER_ME, SPEAKER_THEM = "JS (Joseph)", "Goran (Mr.Crypto)"
SHERPA_MODELS = {
    "small": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.tar.bz2",
    "medium": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-medium.tar.bz2",
}


def log(msg):
    print(f"[transcribe] {msg}", file=sys.stderr, flush=True)


def note_time(path: Path):
    m = TS_RE.search(path.name)
    return datetime(*map(int, m.groups())) if m else datetime.fromtimestamp(path.stat().st_mtime)


def speaker_for(path: Path, me_prefixes):
    return SPEAKER_ME if any(path.name.startswith(p) for p in me_prefixes) else SPEAKER_THEM


def fmt(sec):
    sec = max(0.0, float(sec))
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


# ---------------------------------------------------------------- audio
def to_wav(src: Path, wav_dir: Path, force=False) -> Path:
    dst = wav_dir / (src.stem + ".wav")
    if dst.exists() and not force and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def read_wav(path: Path):
    import numpy as np
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getframerate() == 16000 and w.getsampwidth() == 2, "expected 16k mono s16"
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype("float32") / 32768.0
    return data, 16000


# ---------------------------------------------------------------- quality check
def looks_like_garbage(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    letters = sum(c.isalpha() for c in t)
    ascii_letters = sum(c.isascii() and c.isalpha() for c in t)
    if letters and ascii_letters / letters < 0.7:      # mostly non-latin script for an English note
        return True
    words = t.lower().split()
    if len(words) >= 8 and len(set(words)) / len(words) < 0.25:   # heavy repetition loops
        return True
    return False


# ---------------------------------------------------------------- backends
class FasterWhisperBackend:
    name = "faster-whisper"
    has_confidence = True

    def __init__(self, model_size):
        from faster_whisper import WhisperModel
        log(f"loading faster-whisper '{model_size}' (first run downloads from huggingface.co)")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.model_size = model_size

    def transcribe(self, wav: Path, language):
        segs, info = self.model.transcribe(str(wav), language=language, beam_size=5,
                                           vad_filter=True, condition_on_previous_text=False)
        out = []
        for s in segs:
            unclear = (s.avg_logprob is not None and s.avg_logprob < -1.0) or (s.no_speech_prob or 0) > 0.6
            out.append({"start": s.start, "end": s.end, "text": s.text.strip(), "unclear": unclear})
        return out, info.language


class SherpaWhisperBackend:
    name = "sherpa-onnx whisper"
    has_confidence = False

    def __init__(self, model_size):
        import sherpa_onnx
        mdir = ROOT / "models" / f"sherpa-onnx-whisper-{model_size}"
        if not mdir.exists():
            self._fetch(model_size, mdir)
        enc = mdir / f"{model_size}-encoder.int8.onnx"
        dec = mdir / f"{model_size}-decoder.int8.onnx"
        self.tokens = mdir / f"{model_size}-tokens.txt"
        self.model_size = model_size
        self._mk = lambda lang: sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(enc), decoder=str(dec), tokens=str(self.tokens),
            language=lang or "", task="transcribe", num_threads=max(1, os.cpu_count() or 1))
        self.recognizers = {}

    @staticmethod
    def _fetch(model_size, mdir):
        url = SHERPA_MODELS[model_size]
        mdir.parent.mkdir(exist_ok=True)
        tarball = mdir.parent / (mdir.name + ".tar.bz2")
        log(f"downloading {url}")
        subprocess.run(["curl", "-sS", "-L", "--fail", "-o", str(tarball), url], check=True)
        subprocess.run(["tar", "-xjf", str(tarball), "-C", str(mdir.parent)], check=True)
        tarball.unlink()

    def _chunks(self, audio, sr, max_len=28.0, min_len=4.0):
        """Whisper via sherpa handles <=30 s per call. Split long notes at the quietest point
        before the limit so we do not cut mid-word."""
        import numpy as np
        n = len(audio)
        max_n, min_n, win = int(max_len * sr), int(min_len * sr), int(0.05 * sr)
        pos = 0
        while pos < n:
            if n - pos <= max_n:
                yield pos, n
                break
            lo, hi = pos + min_n, pos + max_n
            frames = [(np.abs(audio[i:i + win]).mean(), i) for i in range(lo, hi - win, win)]
            cut = min(frames)[1] + win // 2 if frames else hi
            yield pos, cut
            pos = cut

    def transcribe(self, wav: Path, language):
        audio, sr = read_wav(wav)
        key = language or ""
        if key not in self.recognizers:
            self.recognizers[key] = self._mk(language)
        rec = self.recognizers[key]
        out = []
        for a, b in self._chunks(audio, sr):
            stream = rec.create_stream()
            stream.accept_waveform(sr, audio[a:b])
            rec.decode_stream(stream)
            text = stream.result.text.strip()
            if text:
                out.append({"start": a / sr, "end": b / sr, "text": text, "unclear": False})
        detected = getattr(getattr(stream, "result", None), "lang", None) or language or "auto"
        return out, detected


def pick_backend(choice, model_size):
    order = {"auto": ["faster-whisper", "sherpa"], "faster-whisper": ["faster-whisper"], "sherpa": ["sherpa"]}[choice]
    errors = []
    for b in order:
        try:
            return FasterWhisperBackend(model_size) if b == "faster-whisper" else SherpaWhisperBackend(model_size)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{b}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
            log(f"backend {b} unavailable -> {errors[-1]}")
    raise SystemExit("no transcription backend could be initialised:\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------- chat context
def load_chat(chat_path: Path):
    if not chat_path.exists():
        return None
    raw = chat_path.read_text(encoding="utf-8", errors="replace")
    return [ln.replace("‎", "").rstrip("\n") for ln in raw.splitlines()]


def chat_context(lines, basename, before=8, after=4):
    if lines is None:
        return None
    idx = [i for i, ln in enumerate(lines) if basename in ln]
    if not idx:
        return []
    i = idx[0]
    return lines[max(0, i - before): i + after + 1]


# ---------------------------------------------------------------- outputs
def write_note_transcript(tdir: Path, src: Path, segs, meta):
    lines = [f"# {src.name}", f"speaker: {meta['speaker']}", f"sent: {meta['sent']}",
             f"backend: {meta['backend']} / model: {meta['model']} / language: {meta['language']}",
             f"duration: {fmt(meta['duration'])}", f"unclear-marking: {meta['unclear_note']}", ""]
    if not segs:
        lines.append("[no speech detected]")
    for s in segs:
        tag = " [unclear]" if s["unclear"] else ""
        lines.append(f"[{fmt(s['start'])} - {fmt(s['end'])}] {s['text']}{tag}")
    (tdir / (src.stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tdir / (src.stem + ".json")).write_text(json.dumps({"meta": meta, "segments": segs}, indent=1), encoding="utf-8")


def build_all(opus_files, tdir: Path, chat_lines, me_prefixes):
    out = ["# ALL_TRANSCRIPTS", "",
           f"generated: {datetime.now().isoformat(timespec='seconds')}",
           f"notes: {len(opus_files)} voice note(s), chronological by filename timestamp.",
           "speaker labels come from the note number (00000052 = JS), not from the audio.",
           "chat context is quoted verbatim from _chat.txt around the attachment line." if chat_lines is not None
           else "chat context unavailable: _chat.txt not found next to the notes.", ""]
    for src in opus_files:
        meta_path = tdir / (src.stem + ".json")
        if not meta_path.exists():
            out += [f"## {src.name}", "", "_not transcribed (missing transcript file)_", ""]
            continue
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        meta, segs = data["meta"], data["segments"]
        out += [f"## {meta['sent']}  -  {src.name}", "",
                f"speaker: {meta['speaker']}  |  duration: {fmt(meta['duration'])}  |  "
                f"backend: {meta['backend']} ({meta['model']}, lang={meta['language']})", ""]
        ctx = chat_context(chat_lines, src.name)
        if ctx:
            out += ["chat context:", ""] + [f"> {ln}" if ln.strip() else ">" for ln in ctx] + [""]
        elif chat_lines is not None:
            out += ["chat context: attachment line not found in _chat.txt", ""]
        out.append("transcript:")
        out.append("")
        if not segs:
            out.append("[no speech detected]")
        for s in segs:
            out.append(f"[{fmt(s['start'])} - {fmt(s['end'])}] {s['text']}{' [unclear]' if s['unclear'] else ''}")
        out.append("")
    text = "\n".join(out)
    (tdir / "ALL_TRANSCRIPTS.md").write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT), help="directory holding the .opus files and _chat.txt")
    ap.add_argument("--model", default="small", choices=["small", "medium"])
    ap.add_argument("--backend", default="auto", choices=["auto", "faster-whisper", "sherpa"])
    ap.add_argument("--language", default="en", help="'auto' to let the model detect")
    ap.add_argument("--force", action="store_true", help="re-transcribe notes that already have a transcript")
    ap.add_argument("--print", action="store_true", help="print ALL_TRANSCRIPTS.md when done")
    ap.add_argument("--me", default=",".join(sorted(DEFAULT_ME)), help="comma list of note prefixes spoken by JS")
    args = ap.parse_args()

    base = Path(args.dir).resolve()
    wav_dir, tdir = base / "wav", base / "transcripts"
    wav_dir.mkdir(exist_ok=True)
    tdir.mkdir(exist_ok=True)
    me_prefixes = {p.strip() for p in args.me.split(",") if p.strip()}
    language = None if args.language == "auto" else args.language

    opus_files = sorted(base.glob("*.opus"), key=note_time)
    if not opus_files:
        log(f"no .opus files found in {base}")
    todo = [p for p in opus_files if args.force or not (tdir / (p.stem + ".json")).exists()]
    log(f"{len(opus_files)} note(s) found, {len(todo)} to transcribe")

    if todo:
        backend = pick_backend(args.backend, args.model)
        for src in todo:
            wav = to_wav(src, wav_dir, args.force)
            audio_len = wave.open(str(wav)).getnframes() / 16000
            log(f"{src.name}: {fmt(audio_len)} -> transcribing ({backend.name})")
            segs, detected = backend.transcribe(wav, language)
            joined = " ".join(s["text"] for s in segs)
            used_lang = language or "auto"
            if language and looks_like_garbage(joined):
                log(f"{src.name}: output looks wrong with language={language}, retrying with auto-detect")
                segs2, detected2 = backend.transcribe(wav, None)
                if not looks_like_garbage(" ".join(s["text"] for s in segs2)):
                    segs, detected, used_lang = segs2, detected2, f"auto->{detected2}"
            meta = {"file": src.name, "speaker": speaker_for(src, me_prefixes),
                    "sent": note_time(src).strftime("%Y-%m-%d %H:%M:%S"), "duration": audio_len,
                    "backend": backend.name, "model": backend.model_size, "language": used_lang,
                    "unclear_note": ("segments with low model confidence are tagged [unclear]"
                                     if backend.has_confidence else
                                     "backend exposes no confidence score; nothing auto-tagged, read with care")}
            write_note_transcript(tdir, src, segs, meta)

    text = build_all(opus_files, tdir, load_chat(base / "_chat.txt"), me_prefixes)
    log(f"wrote {tdir / 'ALL_TRANSCRIPTS.md'}")
    if args.print:
        print(text)


if __name__ == "__main__":
    main()
