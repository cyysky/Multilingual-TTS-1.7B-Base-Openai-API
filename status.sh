#!/usr/bin/env bash
# Live snapshot: what the TTS server is processing right now
echo "== current state =="
curl -s http://localhost:8998/healthz | python3 -c "
import json, sys
d = json.load(sys.stdin)
m = d['monitor']
print(f\"status: {d['status']}  |  inflight: {m['inflight']}  |  queue_waiting: {m['queue_waiting']}\")
print(f\"requests_total: {m['requests_total']}  |  errors: {m['errors_total']}\")
if m['current']:
    c = m['current']
    print(f\"NOW PROCESSING req#{c['id']}: voice={c['voice']} text={c['text_len']}ch fmt={c['fmt']} stage={c['stage']} elapsed={c['elapsed_sec']}s from {c['client']}\")
else:
    print('idle - nothing processing')
print()
print('last completed:')
for r in m['last']:
    print(f\"  req#{r['id']} {r['client']} voice={r['voice']} {r['text_len']}ch -> {r['audio_bytes']}B {r['fmt']} in {r['duration_sec']}s\")
"
