# mining-goran

local transcription + due diligence tooling. nothing here sends data anywhere.

setup (done once):

    python3 -m venv venv
    ./venv/bin/pip install faster-whisper sherpa-onnx
    # ffmpeg must be on PATH (brew/apt/winget, user-space is fine)

run:

    ./venv/bin/python transcribe.py --print           # new .opus files only
    ./venv/bin/python transcribe.py --force --print   # redo everything
    ./venv/bin/python transcribe.py --model medium    # if the machine can take it

drop the whatsapp export (`_chat.txt` + `*.opus`) into this directory first.
outputs: `wav/` (16 kHz mono), `transcripts/<note>.txt|json`, `transcripts/ALL_TRANSCRIPTS.md`.

backend order: faster-whisper (model from huggingface.co) then sherpa-onnx whisper
(model from github releases into `models/`). both give the same whisper "small" weights.

this repository is public. `.gitignore` keeps the chat export, audio, invoices,
screenshots, transcripts and analysis out of git. do not force-add them.

second opinion and consensus:

    ./venv/bin/python transcribe.py --dir <copy-of-notes> --model medium --backend sherpa   # second pass
    ./venv/bin/python consensus.py --primary <copy-of-notes> --secondary <dir-with-first-pass> --out transcripts

words both passes agree on are kept, disagreements are written as [unclear: "a" / "b"].
`--chunk 20` shortens the decode window if a model drops part of a long note.
