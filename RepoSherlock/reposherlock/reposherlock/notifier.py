"""DNotifier integration (the only module that imports the DNotifier SDK).

What it uses (all from the official Python SDK docs, `pip install dnotifier`):
  * DNotifier(app_id, secret, transport="ws", user_id, on_connected, on_message, on_disconnected)
  * await client.connect(), await client.send(sender_id, receiver_ids, data, save_history)
  * client.get_plan_limits(), await client.disconnect()
It does NOT use DNotifier's hosted AI (send_ai); inference stays local in Ollama.

How delivery is verified
  `send()` over WebSocket returns a message id as soon as the frame is written, which is
  NOT proof of delivery. So every event is addressed to a second "ack" listener client
  (also connected through DNotifier) and counts as `confirmed` only when that listener
  receives it back with the same event id. Anything else is reported honestly:
    confirmed         -> received back through DNotifier
    sent_unconfirmed  -> handed to DNotifier, but no echo was seen in time
    failed            -> DNotifier raised an error
    skipped           -> DNotifier not used (disabled / no credentials / not connected)
  Non-confirmed events are also written to a clearly labelled LOCAL-ONLY file.

Identities (all inside your one DNotifier app):
  agent   DNOTIFIER_AGENT_USER_ID   (default reposherlock-agent)    sends events
  ack     DNOTIFIER_ACK_USER_ID     (default reposherlock-ack)      in-process delivery check
  monitor DNOTIFIER_MONITOR_USER_ID (default reposherlock-monitor)  `python main.py monitor`
          (set it to an empty value to send to the ack listener only)
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

LOCAL_FALLBACK_FILE = "reposherlock_events.local.jsonl"
EVENTS = ("investigation_started", "repository_scanned", "finding_discovered",
          "report_completed", "investigation_failed")


@dataclass
class DeliveryResult:
    event: str
    event_id: str
    status: str                      # confirmed | sent_unconfirmed | failed | skipped
    detail: str = ""
    latency_ms: int | None = None

    @property
    def delivered(self) -> bool:
        return self.status == "confirmed"


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _shrink(value: Any, depth: int = 0) -> Any:
    """Keep payloads small (the plan has a per-message size cap) and never ship source code."""
    if isinstance(value, str):
        return value[:300]
    if isinstance(value, dict):
        return {str(k)[:40]: _shrink(v, depth + 1) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        return [_shrink(v, depth + 1) for v in list(value)[:10]]
    return value


class Notifier:
    def __init__(self, enabled: bool = True, ack_timeout: float = 8.0, connect_timeout: float = 15.0,
                 run_id: str | None = None, local_log: str | Path = LOCAL_FALLBACK_FILE) -> None:
        self.enabled = enabled
        self.ack_timeout = ack_timeout
        self.connect_timeout = connect_timeout
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.local_log = Path(local_log)
        self.state = "disabled" if not enabled else "idle"   # disabled | idle | ready | unavailable
        self.state_detail = "DNotifier disabled for this run" if not enabled else ""
        self.notes: list[str] = []
        self.plan_limits: dict | None = None
        self.results: list[DeliveryResult] = []

        self._app_id = _env("DNOTIFIER_APP_ID")
        self._secret = _env("DNOTIFIER_SECRET")
        self._agent = _env("DNOTIFIER_AGENT_USER_ID", "reposherlock-agent")
        self._ack = _env("DNOTIFIER_ACK_USER_ID", "reposherlock-ack")
        self._monitor = os.getenv("DNOTIFIER_MONITOR_USER_ID", "reposherlock-monitor").strip()
        self._save_history = _env("DNOTIFIER_SAVE_HISTORY", "0") in ("1", "true", "yes")

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._sender = None
        self._listener = None
        self._pending: dict[str, asyncio.Future] = {}

    # ------------------------------------------------------------------ helpers
    def _clean(self, exc: BaseException) -> str:
        msg = f"{exc.__class__.__name__}: {exc}"
        if self._secret:
            msg = msg.replace(self._secret, "***")
        return msg[:240]

    def _call(self, coro, timeout: float):
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def _log_local(self, result: DeliveryResult, payload: dict) -> None:
        line = {"LOCAL_ONLY_FALLBACK": True, "dnotifier_status": result.status,
                "detail": result.detail, "payload": payload}
        try:
            with self.local_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> tuple[bool, str]:
        """Connect both clients. Returns (ready, human-readable status)."""
        if not self.enabled:
            return False, self.state_detail
        if not self._app_id or not self._secret:
            self.state, self.state_detail = "unavailable", \
                "DNOTIFIER_APP_ID / DNOTIFIER_SECRET are not set (copy .env.example to .env)"
            return False, self.state_detail
        try:
            import dnotifier  # noqa: F401
        except ImportError:
            self.state, self.state_detail = "unavailable", "The dnotifier package is missing: pip install dnotifier"
            return False, self.state_detail

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="dnotifier-loop")
        self._thread.start()
        try:
            self._call(self._connect_async(), self.connect_timeout * 2 + 5)
        except Exception as exc:  # includes TimeoutError and SDK RuntimeError("Authentication failed")
            self.state, self.state_detail = "unavailable", f"Could not connect to DNotifier ({self._clean(exc)})"
            self._shutdown_loop()
            return False, self.state_detail
        self.state, self.state_detail = "ready", "connected to DNotifier"
        return True, self.state_detail

    async def _make_client(self, user_id: str, on_message: Callable):
        from dnotifier import DNotifier
        connected = asyncio.Event()
        client = DNotifier(
            app_id=self._app_id, secret=self._secret, transport="ws", user_id=user_id,
            on_connected=connected.set,
            on_message=on_message,
            on_disconnected=lambda code=None, reason=None: None,
        )
        await client.connect()                       # returns after the handshake frame is sent...
        await asyncio.wait_for(connected.wait(), self.connect_timeout)   # ...so wait for the server's ack
        return client

    async def _connect_async(self) -> None:
        self._sender = await self._make_client(self._agent, lambda msg: None)
        try:
            limits = self._sender.get_plan_limits()
            self.plan_limits = dataclasses.asdict(limits) if limits else None
        except Exception:
            self.plan_limits = None
        self._listener = await self._make_client(self._ack, self._on_ack_message)
        max_users = (self.plan_limits or {}).get("max_users")
        if isinstance(max_users, int) and max_users < 3:
            self.notes.append(f"Your plan allows {max_users} user(s); RepoSherlock uses up to 3 identities "
                              f"(agent, ack, monitor). The `monitor` terminal may be rejected.")

    def _on_ack_message(self, msg) -> None:
        """Runs on the DNotifier loop thread whenever the ack listener receives something."""
        try:
            body = msg.payload.to_json()
        except Exception:
            return
        ids: list[str] = []
        if isinstance(body, dict):
            if body.get("event_id"):
                ids.append(str(body["event_id"]))
            text = body.get("text")
            if isinstance(text, str):
                ids += re.findall(r"\bid=([0-9a-f]{8,32})", text)
        for eid in ids:
            fut = self._pending.get(eid)
            if fut is not None and not fut.done():
                fut.set_result(time.monotonic())

    def close(self) -> None:
        if self._loop is None:
            return
        for client in (self._sender, self._listener):
            if client is not None:
                try:
                    self._call(client.disconnect(), 5)
                except Exception:
                    pass
        self._shutdown_loop()

    def _shutdown_loop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=3)
            self._loop = None

    # ------------------------------------------------------------------ events
    def emit(self, event: str, summary: str, data: dict | None = None) -> DeliveryResult:
        event_id = uuid.uuid4().hex[:12]
        payload = {
            # "type": "text" + "text" follow the documented text-message shape, so the event is
            # readable in any DNotifier client; the extra fields carry the structured event.
            "type": "text",
            "text": f"[{event}] {summary[:200]} (id={event_id})",
            "event": event, "event_id": event_id, "run_id": self.run_id,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data": _shrink(data or {}),
        }
        if self.state != "ready":
            result = DeliveryResult(event, event_id, "skipped", self.state_detail or "DNotifier not connected")
        else:
            try:
                result = self._call(self._emit_async(event, event_id, payload), self.ack_timeout + 10)
            except Exception as exc:
                result = DeliveryResult(event, event_id, "failed", self._clean(exc))
        if not result.delivered and self.state != "disabled":
            self._log_local(result, payload)
        self.results.append(result)
        return result

    async def _emit_async(self, event: str, event_id: str, payload: dict) -> DeliveryResult:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[event_id] = fut          # register BEFORE sending so a fast echo is never missed
        receivers = [self._ack] + ([self._monitor] if self._monitor else [])
        t0 = time.monotonic()
        try:
            try:
                await self._sender.send(sender_id=self._agent, receiver_ids=receivers,
                                        data=payload, save_history=self._save_history)
            except Exception as exc:
                return DeliveryResult(event, event_id, "failed", self._clean(exc))
            try:
                t1 = await asyncio.wait_for(fut, self.ack_timeout)
                return DeliveryResult(event, event_id, "confirmed", "received back through DNotifier",
                                      int((t1 - t0) * 1000))
            except asyncio.TimeoutError:
                return DeliveryResult(event, event_id, "sent_unconfirmed",
                                      f"handed to DNotifier but no echo within {self.ack_timeout:g}s")
        finally:
            self._pending.pop(event_id, None)

    def summary_line(self) -> str:
        ok = sum(r.delivered for r in self.results)
        return f"{ok}/{len(self.results)} event(s) confirmed through DNotifier"


# ---------------------------------------------------------------------- live monitor (2nd terminal)
async def monitor_events(on_event: Callable[[dict, str], None], seconds: float | None = None,
                         on_ready: Callable[[], None] | None = None) -> None:
    """Connect as the monitor user and call on_event(payload, sender) for each RepoSherlock event."""
    from dnotifier import DNotifier
    app_id, secret = _env("DNOTIFIER_APP_ID"), _env("DNOTIFIER_SECRET")
    if not app_id or not secret:
        raise RuntimeError("DNOTIFIER_APP_ID / DNOTIFIER_SECRET are not set (copy .env.example to .env)")
    user = _env("DNOTIFIER_MONITOR_USER_ID", "reposherlock-monitor") or "reposherlock-monitor"
    ready = asyncio.Event()

    def handle(msg) -> None:
        try:
            body = msg.payload.to_json()
        except Exception:
            return
        if isinstance(body, dict):
            on_event(body, str(msg.metadata.sender))

    client = DNotifier(app_id=app_id, secret=secret, transport="ws", user_id=user,
                       on_connected=ready.set, on_message=handle,
                       on_disconnected=lambda code=None, reason=None: None)
    try:
        await client.connect()
        await asyncio.wait_for(ready.wait(), 15)
        if on_ready:
            on_ready()
        await asyncio.sleep(seconds if seconds else 10 ** 6)
    except RuntimeError as exc:
        raise RuntimeError(str(exc).replace(secret, "***")) from None
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass
