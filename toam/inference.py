"""
TOAM inference helper: builds the exact prompt the model was trained on,
runs greedy decoding, and parses the JSON action.

This is the same code as `tiny_agent.py` in the Hugging Face model repos.

    from toam.inference import TinyAgent
    agent = TinyAgent("ksg91/toam-100m-v0.2")   # or a local folder
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    agent.decide(tools, "What's the weather in Pune?")
    # {'type': 'tool_call', 'tool': 'get_weather', 'arguments': {'city': 'Pune'}}

Actions the model can return:
    tool_call   {"type":"tool_call","tool":..,"arguments":{..}}
    tool_calls  {"type":"tool_calls","calls":[{"tool":..,"arguments":{..}}, ..]}   (parallel)
    ask_user    {"type":"ask_user","question":..}     a required value is missing
    escalate    {"type":"escalate", ..}               no tool can do it / policy says hand off
    finish / retry / wait                             multi-step control (pass history/state)

Tool schemas may be OpenAI-style, plain {name, description, parameters}, JSON Schema,
Python-style types ("dict"/"float") or flat parameter maps; they are rendered the way training saw them.
The prompt must fit the 2048-token context: keep menus short (~10 tools).
"""

from __future__ import annotations

import json
import re
from typing import Any

SYSTEM_TEXT = "You are an agent controller. Choose the correct next action."
SINGLE_STEP_STATE = {"step": 0}
MAX_CONTEXT = 2048

J = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))  # noqa: E731


# ---------------------------------------------------------------- schema rendering (as in training)

_PY_TYPES = {
    "str": "string", "string": "string", "text": "string", "any": "string",
    "int": "integer", "integer": "integer", "long": "integer",
    "float": "number", "double": "number", "number": "number", "decimal": "number",
    "bool": "boolean", "boolean": "boolean",
    "dict": "object", "object": "object", "map": "object",
    "list": "array", "array": "array", "tuple": "array", "set": "array",
    "null": "null", "none": "null",
}


def norm_type(t: Any) -> tuple[str | None, str | None, bool]:
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        jt, it, _ = norm_type(non_null[0] if non_null else "string")
        return jt, it, "null" in t
    if not isinstance(t, str) or not t.strip():
        return None, None, False
    s = t.strip()
    optional = bool(re.search(r",\s*optional\s*$", s, re.I)) or s.lower().startswith("optional[")
    s = re.sub(r",\s*optional\s*$", "", s, flags=re.I)
    s = re.sub(r"^optional\[(.*)\]$", r"\1", s, flags=re.I)
    m = re.match(r"^(list|array|tuple|set|sequence)\s*\[\s*([^\]]+)\s*\]$", s, re.I)
    if m:
        inner, _, _ = norm_type(m.group(2).split(",")[0])
        return "array", inner, optional
    m = re.match(r"^(dict|map|mapping)\s*\[.*\]$", s, re.I)
    if m:
        return "object", None, optional
    if "|" in s:
        return norm_type([p.strip() for p in s.split("|")])[0], None, optional or "none" in s.lower()
    return _PY_TYPES.get(s.lower(), "string"), None, optional


def _alt_types(spec: dict) -> Any:
    for alt in ("anyOf", "oneOf"):
        for sub in spec.get(alt) or []:
            if isinstance(sub, dict) and sub.get("type") not in (None, "null"):
                return sub.get("type"), sub
    return None, None


def normalize_schema(schema: Any) -> dict:
    if not isinstance(schema, dict):
        return {"properties": {}, "required": []}
    if "properties" in schema or schema.get("type") in ("object", "dict"):
        props = schema.get("properties") or {}
        req = [str(r) for r in (schema.get("required") or []) if r in props]
        flat = False
    else:
        props, req, flat = schema, None, True
    out, implied_req = {}, []
    for name, spec in props.items():
        if not isinstance(spec, dict):
            continue
        t = spec.get("type")
        sub = None
        if t is None:
            t, sub = _alt_types(spec)
        jt, it, optional = norm_type(t)
        s: dict[str, Any] = {"type": jt or "string"}
        if isinstance(spec.get("description"), str) and spec["description"].strip():
            s["description"] = re.sub(r"\s+", " ", spec["description"]).strip()
        for key in ("enum", "format", "minimum", "maximum"):
            if key in spec:
                s[key] = spec[key]
            elif sub and key in sub:
                s[key] = sub[key]
        if "default" in spec and spec["default"] not in ("", None) and (not flat or optional):
            s["default"] = spec["default"]
        items = spec.get("items") if isinstance(spec.get("items"), dict) else (sub or {}).get("items")
        if s["type"] == "array":
            if isinstance(items, dict):
                ijt, _, _ = norm_type(items.get("type"))
                s["items"] = {"type": ijt or "string"}
                if ijt == "object" and isinstance(items.get("properties"), dict):
                    s["items"] = {"type": "object", **normalize_schema(items)}
            elif it:
                s["items"] = {"type": it}
        if s["type"] == "object" and isinstance(spec.get("properties"), dict):
            s.update(normalize_schema(spec))
        out[str(name)] = s
        if flat and not optional:
            implied_req.append(str(name))
    return {"properties": out, "required": req if req is not None else implied_req}


def _trim(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _render_spec(spec: dict, desc_len: int, depth: int) -> dict:
    e: dict[str, Any] = {"type": spec.get("type", "string")}
    if spec.get("description"):
        e["description"] = _trim(spec["description"], desc_len)
    if isinstance(spec.get("enum"), list):
        e["enum"] = spec["enum"][:10]
    for key in ("format", "default", "minimum", "maximum"):
        if key in spec and len(J(spec[key])) <= 40:
            e[key] = spec[key]
    if e["type"] == "array" and isinstance(spec.get("items"), dict):
        it = spec["items"]
        e["items"] = ({"type": "object", "properties": {k: {"type": v.get("type", "string")}
                                                      for k, v in list((it.get("properties") or {}).items())[:8]}}
                      if it.get("type") == "object" and it.get("properties") and depth < 1
                      else {"type": it.get("type", "string")})
    if e["type"] == "object" and spec.get("properties") and depth < 1:
        e["properties"] = {k: _render_spec(v, 60, depth + 1) for k, v in list(spec["properties"].items())[:8]}
        if spec.get("required"):
            e["required"] = spec["required"]
    return e


def render_tool(name: str, description: str | None, schema: Any,
                desc_len: int = 240, param_desc_len: int = 100, max_props: int = 12) -> dict:
    ns = normalize_schema(schema)
    props, req = ns["properties"], ns["required"]
    order = req + [k for k in props if k not in req]
    keep = order[: max(max_props, len(req))]
    return {
        "name": name,
        "description": _trim(description or "", desc_len),
        "parameters": {
            "properties": {k: _render_spec(props[k], param_desc_len, 0) for k in keep},
            "required": req,
        },
    }


def to_rendered_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        f = t.get("function", t) if t.get("type") == "function" else t
        out.append(render_tool(f["name"], f.get("description"), f.get("parameters") or {}))
    return out


# ---------------------------------------------------------------- prompt (v1) and parsing

def format_prompt(tools: list[dict], goal: str, policy: str | None = None, history: list | None = None,
                  state: dict | None = None, observation: Any = None) -> str:
    sections = ["<|system|>\n" + SYSTEM_TEXT, "<|tools|>\n" + J(to_rendered_tools(tools))]
    if policy:
        sections.append("<|policy|>\n" + policy)
    return "\n".join(sections + [
        "<|goal|>\n" + goal,
        "<|history|>\n" + J(history or []),
        "<|state|>\n" + J(state or SINGLE_STEP_STATE),
        "<|observation|>\n" + J(observation),
        "<|action|>\n",
    ])


def parse_action(text: str) -> dict:
    try:
        a = json.loads(text)
        if isinstance(a, dict):
            return a
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"type": "invalid", "raw": text}


class TinyAgent:
    def __init__(self, model: str, device: str | None = None, dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        td = {"auto": (torch.bfloat16 if self.device == "cuda" else torch.float32),
              "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=td).to(self.device).eval()

    def decide(self, tools: list[dict], goal: str, policy: str | None = None, history: list | None = None,
               state: dict | None = None, observation: Any = None, max_new_tokens: int = 256,
               return_text: bool = False):
        prompt = format_prompt(tools, goal, policy, history, state, observation)
        ids = self.tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if ids.shape[1] + 32 > MAX_CONTEXT:
            raise ValueError(f"prompt is {ids.shape[1]} tokens; the model's context is {MAX_CONTEXT}. "
                             "Offer fewer tools or shorter descriptions.")
        with self.torch.no_grad():
            out = self.model.generate(ids.to(self.device), do_sample=False,
                                      max_new_tokens=min(max_new_tokens, MAX_CONTEXT - ids.shape[1]),
                                      eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.pad_token_id)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        action = parse_action(text)
        return (action, text) if return_text else action


if __name__ == "__main__":
    import sys
    agent = TinyAgent(sys.argv[1] if len(sys.argv) > 1 else ".")
    tools = [
        {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
                                          "parameters": {"type": "object", "properties": {
                                              "city": {"type": "string", "description": "City name"},
                                              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                                              "required": ["city"]}}},
        {"type": "function", "function": {"name": "send_email", "description": "Send an email.",
                                          "parameters": {"type": "object", "properties": {
                                              "to": {"type": "string", "format": "email"},
                                              "subject": {"type": "string"}, "body": {"type": "string"}},
                                              "required": ["to", "subject", "body"]}}},
    ]
    for goal in ["What's the weather in Pune right now?",
                 "Email priya@example.com with subject 'Lunch' and say see you at 1pm.",
                 "Send an email to my manager.",
                 "Book me a flight to Tokyo.",
                 "Weather in Paris and in Berlin, please."]:
        print(goal, "->", J(agent.decide(tools, goal)))
