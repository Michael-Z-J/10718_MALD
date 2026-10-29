"""client for local llama.cpp servers (openai-compatible chat api).

chat(model, messages, schema) --> post /v1/chat/completions with a json schema
  (grammar-constrained decoding) --> parsed dict.
each reply is cached on disk under a hash of (model, messages, schema, params),
so a rerun makes no new calls and gives the same outputs.
server url of a model: env MALD_<MODEL>_URL, else the local port below.
"""
import hashlib, json, os, time

import httpx

from mald.paths import CACHE

PORTS = {"gemma": 8091, "glimmer": 8092, "nemotron": 8093, "qwen": 8094}
URLS = {m: os.environ.get(f"MALD_{m.upper()}_URL", f"http://127.0.0.1:{p}") for m, p in PORTS.items()}


def key(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def chat(model, messages, schema=None, temp=0.0, seed=0, max_tokens=4096, tag=""):
    """one chat call. returns (parsed json or text, usage dict)."""
    k = key(model, messages, schema, temp, seed, max_tokens)
    p = CACHE / model / k[:2] / f"{k}.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            return d["out"], d["usage"]
        except json.JSONDecodeError:
            pass
    body = dict(model=model, messages=messages, temperature=temp, seed=seed, max_tokens=max_tokens,
                chat_template_kwargs={"enable_thinking": False})
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    t0 = time.time()
    if model in RAW:
        j = raw(model, messages, schema, temp, seed, max_tokens)
        txt = j["content"]
        j["usage"] = dict(prompt_tokens=j["tokens_evaluated"], completion_tokens=j["tokens_predicted"])
    else:
        j = post(f"{URLS[model]}/v1/chat/completions", body)
        txt = j["choices"][0]["message"]["content"]
    try:
        out = json.loads(txt) if schema else txt
    except json.JSONDecodeError:
        out = {"_raw": txt, "_error": "json"}
    usage = dict(j.get("usage", {}), secs=time.time() - t0, tag=tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(out=out, usage=usage)))
    return out, usage


# models whose chat template opens a reasoning channel that the grammar cannot
# constrain: render the template, open the answer channel, and decode with
# the json_schema grammar on /completion.
RAW = {"glimmer": (" to=user<|message|>", ["<|eot|>", "<|eom|>"]),
       "nemotron": ("<|channel>thought\n<channel|>", ["<turn|>"])}


def post(url, body):
    for i in range(3):
        try:
            r = httpx.post(url, json=body, timeout=1800)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError:
            if i == 2:
                raise
            time.sleep(5)


def raw(model, messages, schema, temp, seed, max_tokens):
    pre, stop = RAW[model]
    prompt = post(f"{URLS[model]}/apply-template", {"messages": messages})["prompt"] + pre
    body = dict(prompt=prompt, temperature=temp, seed=seed, n_predict=max_tokens, stop=stop,
                cache_prompt=True)
    if schema:
        body["json_schema"] = schema
    return post(f"{URLS[model]}/completion", body)


def tokens(model, text):
    r = httpx.post(f"{URLS[model]}/tokenize", json={"content": text}, timeout=120)
    return len(r.json()["tokens"])


def clip(model, text, n):
    """truncate text to at most n tokens of `model` (keeps the head)."""
    t = tokens(model, text)
    if t <= n:
        return text
    return text[: int(len(text) * n / t * 0.98)]
