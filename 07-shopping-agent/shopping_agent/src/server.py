# -*- coding: utf-8 -*-
"""HTTP 服务：把 Agent 包成一个网页能用的接口。

用的是标准库的 `http.server`，没有引入 Web 框架。理由和检索层不用向量库
是同一个：这个工程要保证解压之后装两个包就能跑。多一个框架就多一份版本
约束，也多一处可能装不上的地方。

接口不多，两类：

- **普通 JSON 接口** —— 配置、记忆、会话、数据统计
- **一条 SSE 流** —— 对话。用流式是因为一次对话要跑好几秒，中间还有
  工具调用，让浏览器一直白等会让人以为卡死了

模型的密钥只存在服务端。发给浏览器的永远是掩码版本。
"""
from __future__ import annotations

import json
import mimetypes
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .brain import probe
from .config import PRESETS, ModelConfig, get_config, save_config
from .dataset import get_catalog
from .memory import get_memory
from .runtime import get_runtime
from .skills import list_skills
from .trace import TRACE_DIR

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

_mimetypes_added = False


def _serve_static(handler: BaseHTTPRequestHandler, rel: str) -> None:
    global _mimetypes_added
    if not _mimetypes_added:
        mimetypes.add_type("application/javascript", ".js")
        mimetypes.add_type("text/css", ".css")
        _mimetypes_added = True

    target = (WEB / rel.lstrip("/")).resolve()
    try:
        target.relative_to(WEB.resolve())      # 目录穿越防护
    except ValueError:
        handler.send_error(403, "forbidden")
        return
    if not target.exists() or target.is_dir():
        handler.send_error(404, "not found")
        return
    body = target.read_bytes()
    ctype, _ = mimetypes.guess_type(str(target))
    handler._send(200, body, ctype or "application/octet-stream")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "shopping-agent"

    # ── 脚手架 ──────────────────────────────────────────
    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8"
                                                  if ctype.startswith(("text", "application/json"))
                                                  else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def log_message(self, fmt: str, *args) -> None:
        if "/api/chat" in (self.path or ""):
            return
        print(f"  {self.address_string()}  {fmt % args}")

    # ── 路由 ────────────────────────────────────────────
    def do_GET(self) -> None:                                    # noqa: N802
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)

        if path in ("/", "/index.html"):
            return _serve_static(self, "index.html")
        if path.startswith("/web/"):
            return _serve_static(self, path[5:])

        rt = get_runtime()
        if path == "/api/bootstrap":
            return self._json({
                "config": get_config().masked(),
                "presets": PRESETS,
                "stats": rt.stats(),
                "users": [{"user_id": u_.user_id, "nickname": u_.nickname,
                           "city": u_.city, "member_level": u_.member_level,
                           "notes": u_.notes}
                          for u_ in get_catalog().users],
                "skills": list_skills(),
                "memory_stats": {u_.user_id: get_memory().snapshot(u_.user_id)["stats"]
                                 for u_ in get_catalog().users},
            })
        if path == "/api/models":
            cfg = self._probe_cfg(qs)
            p = probe(cfg)
            return self._json(p)
        if path == "/api/memory":
            uid = (qs.get("user_id") or ["U1001"])[0]
            return self._json(get_memory().snapshot(uid))
        if path == "/api/session":
            sid = (qs.get("session_id") or [""])[0]
            uid = (qs.get("user_id") or ["U1001"])[0]
            if not sid:
                return self._json({"history": [], "snapshot": {"active": False}})
            return self._json({"history": rt.history(sid, uid),
                               "snapshot": rt.snapshot(sid, uid)})
        if path == "/api/trace":
            sid = (qs.get("session_id") or [""])[0]
            p = TRACE_DIR / f"{sid}.jsonl"
            if not p.exists():
                return self._json({"lines": []})
            text = p.read_text(encoding="utf-8").splitlines()
            return self._json({"lines": [json.loads(x) for x in text if x.strip()]})
        return self.send_error(404, "not found")

    def do_POST(self) -> None:                                   # noqa: N802
        path = urlparse(self.path).path
        body = self._body()
        rt = get_runtime()

        if path == "/api/config":
            cfg = save_config(body)
            rt.invalidate()                     # 换了端点，已建的会话要重建
            return self._json({"ok": True, "config": cfg.masked()})

        if path == "/api/config/test":
            cfg = self._probe_cfg({}, body)
            return self._json(probe(cfg))

        if path == "/api/chat":
            sid, uid, text = body.get("session_id"), body.get("user_id"), \
                (body.get("text") or "").strip()
            if not sid or not uid or not text:
                return self._json({"error": "缺少 session_id / user_id / text"}, 400)
            return self._stream(rt.start(sid, uid, text))

        if path == "/api/confirm":
            sid, uid = body.get("session_id"), body.get("user_id")
            if not sid or not uid:
                return self._json({"error": "缺少 session_id / user_id"}, 400)
            return self._stream(rt.resume(sid, uid, bool(body.get("approved"))))

        if path == "/api/session/reset":
            rt.reset(body.get("session_id") or "", body.get("user_id") or "")
            return self._json({"ok": True})

        if path == "/api/memory/forget":
            uid = body.get("user_id") or "U1001"
            ok = get_memory().forget(uid, body.get("key") or "")
            return self._json({"ok": ok})

        if path == "/api/memory/drop-episode":
            ok = get_memory().drop_episode(body.get("ep_id") or "")
            return self._json({"ok": ok})

        if path == "/api/memory/clear":
            uid = body.get("user_id") or "U1001"
            get_memory().clear(uid, body.get("layer") or "all")
            return self._json({"ok": True})

        if path == "/api/memory/decay":
            n = get_memory().decay_scan(body.get("user_id") or "U1001")
            return self._json({"ok": True, "muted": n})

        return self.send_error(404, "not found")

    # ── SSE ─────────────────────────────────────────────
    def _stream(self, q: queue.Queue) -> None:
        """把队列里的事件转成 SSE 推给浏览器。

        非流式的话，一次对话要等十几秒才有第一个字节。中间又静悄悄的，
        用户会以为坏了。流式让「正在做什么」始终可见 ——
        这在 Agent 场景里不只是体验问题，排查问题也靠它。
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            while True:
                try:
                    ev = q.get(timeout=300)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                if ev.get("kind") == "__end__":
                    # 收尾帧带个名字，让只认 data: 的客户端也能一眼认出它不是业务事件
                    self.wfile.write(
                        b'event: end\ndata: {"kind": "__end__"}\n\n')
                    self.wfile.flush()
                    break
                payload = json.dumps(ev, ensure_ascii=False)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ── 辅助 ────────────────────────────────────────────
    def _probe_cfg(self, qs: dict, body: dict | None = None) -> ModelConfig:
        """拼一份待测试的配置：请求里的临时值覆盖已保存的值。

        「测试连接」按钮要在保存之前就能用，所以这里允许把表单里的地址
        和密钥一起发过来试，不落盘。
        """
        cur = get_config().model_dump()
        merged = dict(cur)
        for src in (body or {}, {k: v[0] for k, v in qs.items()}):
            for k in ModelConfig.model_fields:
                if src.get(k) not in (None, ""):
                    merged[k] = src[k]
        return ModelConfig(**merged)

    def do_OPTIONS(self) -> None:                                # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    cfg = get_config()
    print("\n" + "=" * 58)
    print("  购物助手 Agent  已启动")
    print("=" * 58)
    print(f"  打开这个地址：{url}")
    print(f"  模型模式：{'真实模型 · ' + (cfg.model or '未指定') if cfg.is_ready() else '内置规则（还没配模型，页面右上角可以配）'}")
    print(f"  数据目录：{ROOT / 'data'}")
    print("  Ctrl+C 停止")
    print("=" * 58 + "\n")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  已停止。")
    finally:
        httpd.server_close()
