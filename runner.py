#!/usr/bin/env python3
"""Vision runner: connects to the relay over a WebSocket, receives {id, text, urls}, fetches the photos,
asks a local open-weights vision model (ollama) for the JSON and sends {id, verdict} back.
No API keys for the model: the weights run here. Only RELAY_URL + RELAY_KEY (the relay's shared secret)."""
import base64, json, os, sys, time, threading, urllib.request
import websocket  # websocket-client

RELAY = os.environ["RELAY_URL"].rstrip("/").replace("https://", "wss://").replace("http://", "ws://") + "/relay/ws"
KEY = os.environ["RELAY_KEY"]
MODEL = os.environ.get("VISION_MODEL", "gemma3:4b")
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MAX_PHOTOS = int(os.environ.get("MAX_PHOTOS", "2"))
RESIZE = int(os.environ.get("RESIZE", "448"))   # longest side sent to the model (vision-encoder cost scales with pixels on qwen-style models)
STATS = {"jobs": 0, "errors": 0, "last_s": None, "avg_s": None, "model": MODEL}
DEADLINE = float(os.environ.get("JOB_SECONDS", "40"))
STOP_AT = time.time() + float(os.environ.get("RUN_SECONDS", str(5 * 3600 + 40 * 60)))  # leave the 6h job limit early


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def fetch(u):
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://tinder.com/"})
    with urllib.request.urlopen(req, timeout=15) as r:
        b = r.read()
    if RESIZE:
        try:
            from PIL import Image
            import io
            im = Image.open(io.BytesIO(b)).convert("RGB")
            if max(im.size) > RESIZE:
                im.thumbnail((RESIZE, RESIZE))
                out = io.BytesIO(); im.save(out, "JPEG", quality=85); b = out.getvalue()
        except Exception as e:
            log(f"resize skipped: {str(e)[:60]}")
    return b


def judge(text, urls):
    t0 = time.time()
    imgs = []
    for u in urls[:MAX_PHOTOS]:
        try:
            b = fetch(u)
            if len(b) >= 12 * 1024:
                imgs.append(base64.b64encode(b).decode())
        except Exception as e:
            log(f"photo fetch failed: {str(e)[:60]}")
    if not imgs:
        return {"error": "no usable photos"}
    body = {"model": MODEL, "stream": False, "format": "json", "options": {"temperature": 0, "num_predict": 300},
            "messages": [{"role": "user", "content": text, "images": imgs}]}
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=DEADLINE) as r:
        d = json.load(r)
    v = json.loads(d["message"]["content"])
    v["_model"] = "runner/" + MODEL
    v["_timing"] = f"{len(imgs)} photos, {time.time() - t0:.1f}s"
    return v


def main():
    while time.time() < STOP_AT:
        try:
            ws = websocket.create_connection(RELAY, header=[f"X-Key: {KEY}", "User-Agent: Mozilla/5.0 swiper-vision"], timeout=60)
            log("connected to the relay")
            ws.send(json.dumps({"type": "hello", "stats": STATS, "seat": os.environ.get("SEAT", "?")}))
            ws.settimeout(30)
            last_ping = time.time()
            while time.time() < STOP_AT:
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    if time.time() - last_ping > 20:
                        ws.send(json.dumps({"type": "ping"})); last_ping = time.time()
                    continue
                if not raw:
                    continue
                try:
                    m = json.loads(raw)
                except ValueError:
                    continue
                if m.get("type") != "job":
                    continue
                t = time.time()
                try:
                    v = judge(m.get("text") or "", m.get("urls") or [])
                except Exception as e:
                    v = {"error": f"runner: {str(e)[:100]}"}
                took = time.time() - t
                STATS["jobs"] += 1; STATS["errors"] += 1 if "error" in v else 0; STATS["last_s"] = round(took, 1)
                STATS["avg_s"] = round(took if STATS["avg_s"] is None else STATS["avg_s"] * 0.8 + took * 0.2, 1)
                ws.send(json.dumps({"type": "result", "id": m["id"], "verdict": v, "stats": STATS})); last_ping = time.time()
                log(f"{m['id']} -> {('ERR ' + v['error']) if 'error' in v else v.get('_timing')} ({took:.1f}s)")
            ws.close()
        except Exception as e:
            log(f"relay error: {str(e)[:100]}; reconnecting in 5s")
            time.sleep(5)
    log("run window over, exiting so the next job takes the seat")


if __name__ == "__main__":
    main()
