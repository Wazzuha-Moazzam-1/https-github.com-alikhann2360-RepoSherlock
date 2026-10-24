"""Tests the delivery-confirmation logic with a FAKE dnotifier SDK (no network).
They prove RepoSherlock only says 'confirmed' when the echo really arrives."""
import asyncio
import dataclasses
import json
import sys
import types

import pytest

import notifier as dn


@dataclasses.dataclass
class FakeLimits:
    max_users: int | None = 100
    messages_hard_limit: int | None = 1000


class FakePayload:
    def __init__(self, data): self.data = data
    def to_json(self): return self.data


class FakeMsg:
    def __init__(self, data, sender):
        self.payload = FakePayload(data)
        self.metadata = types.SimpleNamespace(sender=sender, id="m1", timestamp=0, type="text")


REGISTRY: dict = {}
BEHAVIOUR = {"deliver": True, "fail_send": False, "fail_connect": False}


class FakeDNotifier:
    def __init__(self, app_id, secret, transport, user_id, on_connected=None, on_message=None,
                 on_disconnected=None, **kw):
        self.user_id, self.on_connected, self.on_message = user_id, on_connected, on_message
        self.secret, self.is_connected = secret, False

    async def connect(self):
        if BEHAVIOUR["fail_connect"]:
            raise RuntimeError(f"Authentication failed for secret {self.secret}")
        REGISTRY[self.user_id] = self
        self.is_connected = True
        asyncio.get_running_loop().call_soon(self.on_connected)
        return True

    def get_plan_limits(self): return FakeLimits()

    async def send(self, sender_id, data, receiver_id=None, receiver_ids=None, save_history=True):
        if BEHAVIOUR["fail_send"]:
            raise RuntimeError("Not connected")
        json.dumps(data)                                  # payload must be JSON-encodable
        if BEHAVIOUR["deliver"]:
            for r in (receiver_ids or [receiver_id]):
                if r in REGISTRY:
                    asyncio.get_running_loop().call_soon(REGISTRY[r].on_message, FakeMsg(data, sender_id))
        return "message-id-123"                           # like the real SDK: an id, NOT a delivery proof

    async def disconnect(self): self.is_connected = False


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch, tmp_path):
    REGISTRY.clear()
    BEHAVIOUR.update(deliver=True, fail_send=False, fail_connect=False)
    mod = types.ModuleType("dnotifier")
    mod.DNotifier = FakeDNotifier
    monkeypatch.setitem(sys.modules, "dnotifier", mod)
    monkeypatch.setenv("DNOTIFIER_APP_ID", "app-123456")
    monkeypatch.setenv("DNOTIFIER_SECRET", "super-secret-value")
    monkeypatch.chdir(tmp_path)


def make(**kw):
    return dn.Notifier(ack_timeout=0.5, connect_timeout=2, **kw)


def test_confirmed_only_when_echo_received():
    n = make()
    ready, _ = n.start()
    assert ready
    res = n.emit("investigation_started", "hello", {"repo": "x"})
    assert res.status == "confirmed" and res.delivered and res.latency_ms is not None
    assert not (n.local_log).exists()                    # nothing needed the fallback
    n.close()


def test_send_returning_an_id_is_not_treated_as_delivery():
    BEHAVIOUR["deliver"] = False                          # SDK accepts the frame, nothing comes back
    n = make(); n.start()
    res = n.emit("report_completed", "done")
    assert res.status == "sent_unconfirmed" and not res.delivered
    line = json.loads(n.local_log.read_text().splitlines()[0])
    assert line["LOCAL_ONLY_FALLBACK"] is True and line["dnotifier_status"] == "sent_unconfirmed"
    n.close()


def test_send_error_is_reported_as_failed():
    BEHAVIOUR["fail_send"] = True
    n = make(); n.start()
    res = n.emit("finding_discovered", "x")
    assert res.status == "failed" and "Not connected" in res.detail
    n.close()


def test_missing_credentials_means_skipped_never_confirmed(monkeypatch):
    monkeypatch.delenv("DNOTIFIER_APP_ID"); monkeypatch.delenv("DNOTIFIER_SECRET")
    n = make()
    ready, detail = n.start()
    assert not ready and "DNOTIFIER_APP_ID" in detail
    res = n.emit("investigation_started", "x")
    assert res.status == "skipped" and not res.delivered
    assert n.local_log.exists()


def test_connect_failure_hides_the_secret():
    BEHAVIOUR["fail_connect"] = True
    n = make()
    ready, detail = n.start()
    assert not ready and "super-secret-value" not in detail and "***" in detail
    assert n.emit("investigation_started", "x").status == "skipped"


def test_disabled_run_sends_nothing_and_writes_nothing():
    n = dn.Notifier(enabled=False)
    res = n.emit("investigation_started", "x")
    assert res.status == "skipped" and not n.local_log.exists()


def test_payload_is_small_and_has_documented_text_shape():
    captured = {}
    orig = FakeDNotifier.send

    async def spy(self, sender_id, data, **kw):
        captured.update(data); return await orig(self, sender_id, data, **kw)

    FakeDNotifier.send = spy
    try:
        n = make(); n.start()
        n.emit("finding_discovered", "s" * 1000, {"code": "x" * 5000, "items": list(range(100))})
        n.close()
    finally:
        FakeDNotifier.send = orig
    assert captured["type"] == "text" and captured["text"].startswith("[finding_discovered]")
    assert len(json.dumps(captured)) < 4000 and len(captured["data"]["items"]) == 10


def test_plan_user_limit_warning():
    class Tiny(FakeDNotifier):
        def get_plan_limits(self): return FakeLimits(max_users=2)

    sys.modules["dnotifier"].DNotifier = Tiny
    n = make(); n.start()
    assert any("3 identities" in note for note in n.notes)
    n.close()
