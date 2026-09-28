#!/usr/bin/env python3
"""Vision runner: connects to the relay over a WebSocket, receives {id, text, urls}, fetches the photos,
asks a local open-weights vision model (ollama) for the JSON and sends {id, verdict} back.
No API keys for the model: the weights run here. Only RELAY_URL + RELAY_KEY (the relay's shared secret)."""
import base64, json, os, sys, time, threading, urllib.request
import websocket  # websocket-client

RELAY = os.environ["RELAY_URL"].rstrip("/") + "/relay/ws"
KEY = os.environ["RELAY_KEY"]
MODEL = os.environ.get("VISION_MODEL", "gemma3:4b")
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MAX_PHOTOS = int(os.environ.get("MAX_PHOTOS", "3"))
DEADLINE = float(os.environ.get("JOB_SECONDS", "40"))
STOP_AT = time.time() + float(os.environ.get("RUN_SECONDS", str(5 * 3600 + 40 * 60)))  # leave the 6h job limit early


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def fetch(u):
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://tinder.com/"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read()


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
    body = {"model": MODEL, "stream": False, "format": "json", "options": {"temperature": 0, "num_predict": 400},
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
            ws = websocket.create_connection(RELAY, header=[f"X-Key: {KEY}"], timeout=60)
            log("connected to the relay")
            ws.settimeout(30)
            last_ping = time.time()
            while time.time() < STOP_AT:
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    if time.time() - last_ping > 25:
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
                ws.send(json.dumps({"type": "result", "id": m["id"], "verdict": v}))
                log(f"{m['id']} -> {('ERR ' + v['error']) if 'error' in v else v.get('_timing')} ({time.time() - t:.1f}s)")
            ws.close()
        except Exception as e:
            log(f"relay error: {str(e)[:100]}; reconnecting in 5s")
            time.sleep(5)
    log("run window over, exiting so the next job takes the seat")


if __name__ == "__main__":
    main()
