import base64, json, os, time, urllib.request
MODEL = os.environ["MODEL"]
URLS = ["https://picsum.photos/id/1027/640/800", "https://picsum.photos/id/64/640/800"]
PROMPT = ("Rate the person in the photos. Return ONLY a JSON object with keys: is_woman (true/false), feminine (0-10), body (one of slim, athletic, average, curvy, plus), "
          "body_confidence (0-1), full_body_visible (true/false), swimwear (true/false), curves (0-10), photo_quality (0-10), grainy (true/false), group_photo (true/false), "
          "face (0-10), dyed_hair (true/false), bust (0-10), sexy_vibe (0-10), in_shape (true/false), facial_piercings (true/false), alt_style (true/false), glutes (0-10), gym_selfie (true/false).")
imgs = [base64.b64encode(urllib.request.urlopen(u, timeout=20).read()).decode() for u in URLS]
def run(n, tag):
    body = {"model": MODEL, "stream": False, "format": "json", "options": {"temperature": 0, "num_predict": 300},
            "messages": [{"role": "user", "content": PROMPT, "images": imgs[:n]}]}
    t = time.time()
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600)
    d = json.load(r)
    print(f"RESULT {MODEL} {tag} photos={n} {time.time()-t:.1f}s prompt_eval={d.get('prompt_eval_count')} eval={d.get('eval_count')} :: {d['message']['content'][:220].replace(chr(10),' ')}", flush=True)
run(1, "warm")
run(1, "one")
run(2, "two")
run(2, "two-again")
