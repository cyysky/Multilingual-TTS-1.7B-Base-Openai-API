"""OpenAI-compatible TTS server for Scicom-intl/Multilingual-TTS-1.7B-Base.

Hosts the model on a single GPU (set CUDA_VISIBLE_DEVICES before starting)
and exposes:
    GET  /healthz                 health check
    GET  /v1/models               model list (OpenAI format)
    POST /v1/audio/speech         text -> speech (OpenAI format)

Voice handling:
    * a bare name -> multilingual TTS speaker (e.g. "husein", any speaker in
      https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS)
    * a name matching voices/<name>.{wav,mp3,...} or a data URI
      (data:audio/wav;base64,...) -> voice cloning from that reference audio.
      For cloning, pass the transcript of the reference audio in "voice_text".
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import subprocess
import threading
import time
import urllib.parse
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from neucodec import NeuCodec
from pydantic import BaseModel, Field, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "/home/gpusvr/hf-models/Multilingual-TTS-1.7B-Base")
VOICES_DIR = os.environ.get("TTS_VOICES_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices"))
CODEC_ID = "neuphonic/neucodec"
DEFAULT_MODEL = "Scicom-intl/Multilingual-TTS-1.7B-Base"
DEFAULT_VOICE = "husein"
SAMPLE_RATE = 24000
REF_SAMPLE_RATE = 16000
MAX_TEXT_CHARS = 4096
MAX_REF_SECONDS = 120
MODEL_CREATED = 1743117578  # model card lastModified epoch

SPEAKER_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
AUDIO_TOKEN_RE = re.compile(r"<\|s_(\d+)\|>")
DATA_URI_RE = re.compile(r"^data:audio/[a-zA-Z0-9.+-]+;base64,(.+)$", re.S)

# response_format -> (media type, ffmpeg encode args)
ENCODERS = {
    "mp3": ("audio/mpeg", ["-c:a", "libmp3lame", "-b:a", "192k", "-f", "mp3", "pipe:1"]),
    "opus": ("audio/opus", ["-c:a", "libopus", "-b:a", "96k", "-f", "opus", "pipe:1"]),
    "aac": ("audio/aac", ["-c:a", "aac", "-b:a", "192k", "-f", "adts", "pipe:1"]),
    "flac": ("audio/flac", ["-c:a", "flac", "-f", "flac", "pipe:1"]),
    "wav": ("audio/wav", ["-c:a", "pcm_s16le", "-f", "wav", "pipe:1"]),
    "pcm": ("audio/pcm", ["-f", "s16le", "pipe:1"]),
}

_model = None
_tokenizer = None
_codec = None
_device = None
_gen_lock = threading.Lock()
_gen_gate: Optional[asyncio.Semaphore] = None  # set in lifespan
_mon = {
    "requests_total": 0,
    "errors_total": 0,
    "queue": 0,     # requests waiting for the generation lock
    "inflight": 0,  # request currently synthesizing
    "current": None,
    "last": [],     # ring buffer of completed requests
}
_mon_lock = threading.Lock()
_next_id = 0


def log(msg: str) -> None:
    print(f"[tts] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def _load_models() -> None:
    global _model, _tokenizer, _codec, _device
    if _model is not None:
        return
    _device = torch.device("cuda")
    log(f"loading tokenizer + model from {MODEL_DIR} ...")
    t0 = time.time()
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    ).eval()
    log(f"model loaded in {time.time() - t0:.1f}s ({torch.cuda.memory_allocated() / 1e9:.2f} GB)")
    t0 = time.time()
    _codec = NeuCodec.from_pretrained(CODEC_ID).eval().to(_device)
    log(f"neucodec loaded in {time.time() - t0:.1f}s")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def atempo_filter(speed: float) -> str:
    """ffmpeg atempo filter chain preserving pitch for speed in [0.25, 4.0]."""
    filters: List[str] = []
    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5
    filters.append(f"atempo={speed:.5f}")
    return ",".join(filters)


def _run_ffmpeg(args: List[str], data: bytes) -> bytes:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", *args],
        input=data,
        capture_output=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='ignore')[-500:]}")
    return proc.stdout


def run_ffmpeg(args: List[str], data: bytes) -> bytes:
    return _run_ffmpeg(args, data)


def decode_audio_16k(data: bytes) -> np.ndarray:
    """Decode any ffmpeg-readable audio to mono float32 at 16 kHz."""
    raw = _run_ffmpeg(
        ["-i", "pipe:0", "-vn", "-f", "f32le", "-ac", "1", "-ar", str(REF_SAMPLE_RATE), "pipe:1"],
        data,
    )
    return np.frombuffer(raw, dtype=np.float32).copy()


def _encode_for_response(wave: np.ndarray, fmt: str, speed: float) -> Tuple[str, bytes]:
    media_type, enc_args = ENCODERS[fmt]
    wav_buf = io.BytesIO()
    sf.write(wav_buf, wave, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    wav_bytes = wav_buf.getvalue()

    if fmt == "pcm" and abs(speed - 1.0) < 1e-6:
        return media_type, wav_bytes[44:]

    fargs = ["-f", "wav", "-i", "pipe:0"]
    if abs(speed - 1.0) > 1e-6:
        fargs += ["-af", atempo_filter(speed)]
    out = _run_ffmpeg([*fargs, *enc_args], wav_bytes)
    if fmt == "pcm" and out.startswith(b"RIFF"):
        out = out[44:]
    return media_type, out


def encode_ref_audio(data: bytes) -> List[int]:
    """Encode reference audio into neucodec audio tokens."""
    y = decode_audio_16k(data)
    if len(y) / REF_SAMPLE_RATE > MAX_REF_SECONDS:
        raise HTTPException(status_code=400, detail=f"Reference audio longer than {MAX_REF_SECONDS}s")
    with torch.no_grad():
        codes = _codec.encode_code(torch.tensor(y, device=_device, dtype=torch.float32)[None, None])
    return [int(i) for i in codes[0, 0].tolist()]


def synthesize(prompt: str, text_len: int) -> np.ndarray:
    inputs = _tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(_device)
    input_len = inputs.input_ids.shape[1]
    max_new = min(8192, max(512, int(text_len * 3.5) + 256))
    with _gen_lock, torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.8,
            repetition_penalty=1.15,
        )
        generated = _tokenizer.decode(out[0, input_len:], skip_special_tokens=False)
        tokens = [int(t) for t in AUDIO_TOKEN_RE.findall(generated)]
        if not tokens:
            raise RuntimeError(
                "model produced no audio tokens "
                f"(generated: {generated[:200]!r}); try again with different text"
            )
        codes = torch.tensor(tokens, device=_device, dtype=torch.long)[None, None]
        wave = _codec.decode_code(codes)[0, 0].detach().cpu().numpy()
    return wave


def tts_prompt(speaker: str, text: str) -> str:
    if not SPEAKER_NAME_RE.match(speaker):
        raise HTTPException(status_code=400, detail=f"Invalid speaker name: {speaker!r}")
    return f"<|im_start|>{speaker}: {text}<|speech_start|>"


def _voice_clone_prompt(ref_bytes: bytes, ref_text: str) -> str:
    tokens = encode_ref_audio(ref_bytes)
    return (
        f"<|im_start|>{ref_text}<|speech_start|>"
        + "".join(f"<|s_{i}|>" for i in tokens)
        + "<|im_end|>"
    )


def resolve_voice(voice: str, voice_text: Optional[str]) -> Optional[str]:
    """Return a prompt prefix for voice cloning, or None for plain speaker TTS."""
    m = DATA_URI_RE.match(voice)
    if m:
        if not voice_text or not voice_text.strip():
            raise HTTPException(status_code=400, detail="voice_text (transcript of reference audio) is required for voice cloning")
        ref_bytes = base64.b64decode(m.group(1))
        return _voice_clone_prompt(ref_bytes, voice_text.strip())

    name = urllib.parse.unquote(voice)
    for ext in ("wav", "mp3", "flac", "ogg", "m4a", "aac", "opus"):
        path = os.path.join(VOICES_DIR, f"{name}.{ext}")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                ref_bytes = f.read()
            if not voice_text or not voice_text.strip():
                txt_path = os.path.join(VOICES_DIR, f"{name}.txt")
                if os.path.isfile(txt_path):
                    voice_text = open(txt_path, encoding="utf-8").read().strip()
            if not voice_text:
                raise HTTPException(status_code=400, detail=f"voice_text required: no transcript sidecar for voice {name!r}")
            return _voice_clone_prompt(ref_bytes, voice_text.strip())

    return None


class SpeechRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL, description="Model id (accepted but ignored)")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default=DEFAULT_VOICE, description="Speaker name, registered voice, or data URI of reference audio")
    response_format: str = Field(default="mp3", description="mp3, opus, aac, flac, wav, or pcm")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed (0.25-4.0)")
    voice_text: Optional[str] = Field(default=None, description="Transcript of reference audio (voice cloning)")

    @field_validator("response_format")
    @classmethod
    def _check_fmt(cls, v: str) -> str:
        if v not in ENCODERS:
            raise ValueError(f"Unsupported response_format {v!r}; choose from {', '.join(ENCODERS)}")
        return v

    @field_validator("input")
    @classmethod
    def _check_input(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("input must not be empty")
        if len(v) > MAX_TEXT_CHARS:
            raise ValueError(f"input must be at most {MAX_TEXT_CHARS} characters")
        return v


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _load_models)
    log("TTS service ready")

    global _gen_gate
    _gen_gate = asyncio.Semaphore(1)

    async def watchdog():
        while True:
            await asyncio.sleep(30)
            with _mon_lock:
                cur = _mon["current"]
                q = _mon["queue"]
            if cur is not None:
                elapsed = time.time() - cur["started"]
                log(
                    f"watchdog: still busy req#{cur['id']} elapsed={elapsed:.0f}s "
                    f"client={cur['client']} text={cur['text_len']}ch voice={cur['voice']} "
                    f"fmt={cur['fmt']} waiting={q}"
                )

    task = asyncio.create_task(watchdog())
    yield
    task.cancel()


app = FastAPI(title="OpenAI-compatible Multilingual TTS", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    if _device is None:
        return {"status": "loading", "model": DEFAULT_MODEL}
    free_mb = int(torch.cuda.mem_get_info()[0] / 1e6)
    with _mon_lock:
        if _mon["current"] is not None:
            cur = {**_mon["current"], "elapsed_sec": round(time.time() - _mon["current"]["started"], 1)}
            del cur["started"]
        else:
            cur = None
        snapshot = {
            "requests_total": _mon["requests_total"],
            "errors_total": _mon["errors_total"],
            "queue_waiting": _mon["queue"],
            "inflight": _mon["inflight"],
            "current": cur,
            "last": _mon["last"][:5],
        }
    return {"status": "ok", "model": DEFAULT_MODEL, "device": str(_device), "gpu_free_mb": free_mb, "monitor": snapshot}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": MODEL_CREATED,
                "owned_by": "scicom-intl",
            }
        ],
    }


@app.get("/")
def root():
    return {
        "service": "OpenAI-compatible TTS",
        "endpoints": ["GET /healthz", "GET /v1/models", "POST /v1/audio/speech"],
        "model": DEFAULT_MODEL,
        "voice": DEFAULT_VOICE,
    }


@app.post("/v1/audio/speech")
async def audio_speech(req: SpeechRequest, request: Request):
    global _next_id
    if _model is None or _gen_gate is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    text = req.input.strip()
    voice = req.voice
    loop = asyncio.get_running_loop()
    client = request.client.host if request.client else "?"

    with _mon_lock:
        _next_id += 1
        req_id = _next_id
        _mon["requests_total"] += 1
        _mon["queue"] += 1
    log(f"req#{req_id}: {client} voice={voice} fmt={req.response_format} "
        f"speed={req.speed} text={len(text)}ch queue={_mon['queue']}")

    t_start = time.time()

    def _work():
        with _mon_lock:
            _mon["inflight"] = 1
            _mon["current"] = {
                "id": req_id,
                "client": client,
                "voice": voice,
                "fmt": req.response_format,
                "speed": req.speed,
                "text_len": len(text),
                "started": time.time(),
                "stage": "generate",
            }
        prefix = resolve_voice(voice, req.voice_text)
        prompt = f"{prefix}{text}<|speech_start|>" if prefix else tts_prompt(voice, text)
        return synthesize(prompt, len(text))

    def _encode():
        return _encode_for_response(wave, req.response_format, req.speed)

    acquired = False
    try:
        async with _gen_gate:
            acquired = True
            with _mon_lock:
                _mon["queue"] = max(0, _mon["queue"] - 1)
            wave = await loop.run_in_executor(None, _work)
            with _mon_lock:
                if _mon["current"] is not None:
                    _mon["current"]["stage"] = "encode"
            media_type, payload = await loop.run_in_executor(None, _encode)
    except Exception:
        with _mon_lock:
            _mon["errors_total"] += 1
        raise
    finally:
        with _mon_lock:
            if not acquired:
                _mon["queue"] = max(0, _mon["queue"] - 1)
            _mon["inflight"] = 0
            _mon["current"] = None
    with _mon_lock:
        _mon["last"].insert(0, {
            "id": req_id,
            "client": client,
            "voice": voice,
            "fmt": req.response_format,
            "text_len": len(text),
            "audio_bytes": len(payload),
            "duration_sec": round((time.time() - t_start), 1),
        })
        _mon["last"] = _mon["last"][:20]
    log(f"req#{req_id}: done in {time.time() - t_start:.1f}s -> {len(payload)} bytes {req.response_format}")
    return Response(content=payload, media_type=media_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8998")), workers=1)
