# Multilingual-TTS-1.7B-Base — OpenAI-compatible TTS server

Serves [Scicom-intl/Multilingual-TTS-1.7B-Base](https://huggingface.co/Scicom-intl/Multilingual-TTS-1.7B-Base)
(Qwen3-1.7B + neucodec, 24 kHz output) behind an OpenAI-compatible API.

- URL: `http://<host>:8998`
- GPU: physical GPU 1 (set via `CUDA_VISIBLE_DEVICES=1`), ~7.5 GB VRAM
- Model files: `/home/gpusvr/hf-models/Multilingual-TTS-1.7B-Base`

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/healthz` | Health + GPU free memory |
| GET | `/v1/models` | Model list (OpenAI format) |
| POST | `/v1/audio/speech` | Text → speech (OpenAI format) |

## Usage

curl:

```bash
curl -X POST http://localhost:8998/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Scicom-intl/Multilingual-TTS-1.7B-Base",
    "input": "Hello! Selamat petang. 你好。",
    "voice": "husein",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  -o out.mp3
```

OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8998/v1", api_key="not-needed")
with client.audio.speech.with_streaming_response.create(
    model="Scicom-intl/Multilingual-TTS-1.7B-Base",
    input="Hello! Selamat petang.",
    voice="husein",
    response_format="mp3",
) as resp:
    resp.stream_to_file("out.mp3")
```

Request fields (OpenAI-compatible):

- `model` — any value, accepted and ignored.
- `input` — text, max 4096 chars (multi-language: en, ms, zh, ta, and more).
- `voice` — speaker name (default `husein`; any speaker in the
  `malaysia-ai/Multilingual-TTS` dataset), **or** a voice-cloning reference:
  - name of a file in `voices/` (e.g. `jenny`) → clones from `voices/jenny.wav`,
    transcript auto-read from `voices/jenny.txt`;
  - data URI `data:audio/wav;base64,...` → clones from that audio.
- `voice_text` — transcript of the reference audio (required when cloning from a
  data URI or when no `voices/<name>.txt` sidecar exists).
- `response_format` — `mp3` (default), `wav`, `pcm`, `opus`, `aac`, `flac`.
- `speed` — 0.25–4.0, pitch-preserving (ffmpeg `atempo`).

## Monitoring

View live processing state from `GET /healthz` -> `monitor`:

```bash
curl -s http://localhost:8998/healthz | python3 -m json.tool
```

```json
"monitor": {
  "requests_total": 12,
  "errors_total": 0,
  "queue_waiting": 1,     // requests waiting behind the current one
  "inflight": 1,          // 1 = currently synthesizing, 0 = idle
  "current": { "id": 5, "client": "192.168.33.33", "voice": "vivian",
               "fmt": "mp3", "text_len": 90, "stage": "generate",
               "elapsed_sec": 1.6 },
  "last": [ ... ]         // last 5 completed requests
}
```

Per-request timing goes to `server.log` (`tail -f /home/gpusvr/tts-server/server.log`):

```
[tts] req#1: 127.0.0.1 voice=husein fmt=mp3 speed=1.0 text=3743ch queue=1
[tts] 15:05:03 watchdog: still busy req#1 elapsed=16s ... waiting=1
[tts] req#1: done in 96.1s -> 1767405 bytes mp3
```

A request that runs longer than 30s triggers a watchdog line every 30s showing
elapsed time and how many requests are queued behind it, so you can distinguish
"long generation" from "stuck".

Deep inspection if you suspect a real hang:

```bash
sudo /home/gpusvr/tts-server/venv/bin/py-spy dump --pid $(ss -tlnp | grep 8998 | grep -oP 'pid=\K[0-9]+')
# live:  sudo .../py-spy top --pid <pid>
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader   # GPU busy but no Python progress => CUDA-level wait
```

Note: requests are serialized (one at a time), so later requests queue behind a
long one; `queue_waiting`/`current.elapsed_sec` show that clearly.

## Operations

Start / stop:

```bash
./start.sh                 # starts on port 8998 (GPU 1), writes server.pid / server.log
kill $(cat server.pid)     # stop
```

Add a voice: drop `voice.wav` (or mp3/flac/ogg/m4a) + `voice.txt` (its
transcript) into `/home/gpusvr/tts-server/voices/`, then use `"voice": "voice"`.

Notes:

- Single worker with serialized generation (thread-safe GPU use), so concurrent
  requests queue rather than run in parallel.
- Model is loaded bf16; peak GPU usage ~7.5 GB.
- `neucodec` (NeuCodec 24 kHz) weights are cached in `~/.cache/huggingface`.
