from __future__ import annotations

import asyncio
import importlib

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.service_api import (
    AutomaticDelivery,
    DurableIngress,
    GatewayEvent,
    GatewayEventKind,
    GatewayServiceContext,
    GatewayServiceError,
    GatewayServiceRegistration,
    MediaDescriptor,
    stable_ingress_event_id,
)
from gateway.service_runtime import GatewayServiceRuntime
from gateway.session import SessionSource, build_session_key
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


class _AckingService:
    def __init__(self, seen):
        self.seen = seen
        self.started = False
        self.stopped = False

    async def start(self, context):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def handle_event(self, event, context):
        self.seen.append(event)
        await context.ack(event.cursor)


class _MissingAckService(_AckingService):
    async def handle_event(self, event, context):
        self.seen.append(event)


def _ingress(event_id="ingress_one"):
    return DurableIngress(
        event_id=event_id,
        platform="whatsapp",
        platform_message_id="wamid.1",
        session_id="session-1",
        session_key="agent:main:whatsapp:dm:customer",
        chat_id="customer",
        chat_type="dm",
        sender_id="customer",
        sender_name="Customer",
        message_type="text",
        text="hello",
        occurred_at=1786158000.0,
    )


def test_durable_ingress_wire_dto_preserves_timestamp_and_opaque_attachment_refs():
    payload = DurableIngress(
        event_id="ingress_media",
        platform="whatsapp",
        platform_message_id="wamid.media",
        session_id="session-1",
        session_key="agent:main:whatsapp:dm:customer",
        chat_id="customer",
        chat_type="dm",
        sender_id="customer",
        sender_name="Customer",
        message_type="voice",
        text="",
        occurred_at=1786158000.0,
        media=(
            MediaDescriptor(
                attachment_ref="ingress_media:attachment:1",
                kind="voice",
                mime_type="audio/ogg",
            ),
        ),
    )

    event = GatewayEvent(7, GatewayEventKind.DURABLE_INGRESS, payload)
    wire = event.to_wire_dict()["payload"]

    assert wire["occurred_at"] == 1786158000.0
    assert wire["media"] == [
        {
            "attachment_ref": "ingress_media:attachment:1",
            "kind": "voice",
            "mime_type": "audio/ogg",
            "file_name": "",
            "size_bytes": None,
            "sha256": "",
        }
    ]


@pytest.mark.asyncio
async def test_service_event_requires_ack_and_replays_after_restart(tmp_path):
    journal = tmp_path / "service-events.sqlite3"
    first_seen = []
    first = _MissingAckService(first_seen)
    runtime = GatewayServiceRuntime(
        [GatewayServiceRegistration("core", lambda: first, critical=True)],
        journal,
    )
    await runtime.start()

    payload = _ingress()
    with pytest.raises(GatewayServiceError, match="without ACK"):
        await runtime.publish(
            payload.event_id, GatewayEventKind.DURABLE_INGRESS, payload
        )
    assert [event.cursor for event in first_seen] == [1]
    await runtime.stop()

    replayed = []
    second = _AckingService(replayed)
    restarted = GatewayServiceRuntime(
        [GatewayServiceRegistration("core", lambda: second, critical=True)],
        journal,
    )
    await restarted.start()
    assert [(event.cursor, event.payload.text) for event in replayed] == [(1, "hello")]

    # Tombstone idempotency: the same platform message does not call Core twice.
    await restarted.publish(
        payload.event_id, GatewayEventKind.DURABLE_INGRESS, payload
    )
    assert len(replayed) == 1
    await restarted.stop()
    assert second.started and second.stopped


@pytest.mark.asyncio
async def test_pre_automatic_delivery_is_fail_closed(tmp_path):
    class RejectingService(_AckingService):
        async def pre_automatic_delivery(self, delivery):
            return False

    service = RejectingService([])
    runtime = GatewayServiceRuntime(
        [GatewayServiceRegistration("policy", lambda: service, critical=True)],
        tmp_path / "events.sqlite3",
    )
    await runtime.start()
    delivery = AutomaticDelivery(
        delivery_id="delivery-1",
        platform="whatsapp",
        session_key="session-key",
        chat_id="customer",
        reply_to_message_id="wamid.1",
        text="answer",
    )
    assert await runtime.pre_automatic_delivery(delivery) is False
    await runtime.stop()


@pytest.mark.asyncio
async def test_concurrent_publishes_are_delivered_in_cursor_order(tmp_path):
    class SerialService(_AckingService):
        def __init__(self):
            super().__init__([])
            self.active_handlers = 0
            self.max_active_handlers = 0

        async def handle_event(self, event, context):
            self.active_handlers += 1
            self.max_active_handlers = max(
                self.max_active_handlers, self.active_handlers
            )
            if event.payload.event_id == "ingress_one":
                await asyncio.sleep(0.02)
            self.seen.append(event.payload.event_id)
            await context.ack(event.cursor)
            self.active_handlers -= 1

    service = SerialService()
    runtime = GatewayServiceRuntime(
        [GatewayServiceRegistration("core", lambda: service, critical=True)],
        tmp_path / "events.sqlite3",
    )
    await runtime.start()
    first = _ingress("ingress_one")
    second = _ingress("ingress_two")
    await asyncio.gather(
        runtime.publish(first.event_id, GatewayEventKind.DURABLE_INGRESS, first),
        runtime.publish(second.event_id, GatewayEventKind.DURABLE_INGRESS, second),
    )
    assert service.seen == ["ingress_one", "ingress_two"]
    assert service.max_active_handlers == 1
    await runtime.stop()


@pytest.mark.asyncio
async def test_next_publish_replays_failed_cursor_before_new_event(tmp_path):
    class FailFirstAttemptService(_AckingService):
        def __init__(self):
            super().__init__([])
            self.failed_once = False

        async def handle_event(self, event, context):
            self.seen.append(event.payload.event_id)
            if not self.failed_once:
                self.failed_once = True
                raise RuntimeError("transient downstream failure")
            await context.ack(event.cursor)

    service = FailFirstAttemptService()
    runtime = GatewayServiceRuntime(
        [GatewayServiceRegistration("core", lambda: service, critical=True)],
        tmp_path / "events.sqlite3",
    )
    await runtime.start()
    first = _ingress("ingress_one")
    second = _ingress("ingress_two")
    with pytest.raises(GatewayServiceError, match="failed at cursor 1"):
        await runtime.publish(
            first.event_id, GatewayEventKind.DURABLE_INGRESS, first
        )
    await runtime.publish(
        second.event_id, GatewayEventKind.DURABLE_INGRESS, second
    )
    assert service.seen == ["ingress_one", "ingress_one", "ingress_two"]
    await runtime.stop()


@pytest.mark.asyncio
async def test_failed_optional_service_does_not_block_active_service(tmp_path):
    class FailingOptionalService(_AckingService):
        async def start(self, context):
            raise RuntimeError("optional service unavailable")

    seen = []
    runtime = GatewayServiceRuntime(
        [
            GatewayServiceRegistration(
                "optional", lambda: FailingOptionalService([]), critical=False
            ),
            GatewayServiceRegistration(
                "core", lambda: _AckingService(seen), critical=True
            ),
        ],
        tmp_path / "events.sqlite3",
    )
    await runtime.start()
    payload = _ingress()
    await runtime.publish(
        payload.event_id, GatewayEventKind.DURABLE_INGRESS, payload
    )
    assert [event.payload.event_id for event in seen] == [payload.event_id]
    await runtime.stop()


def test_plugin_registers_factory_without_private_gateway_capabilities():
    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="external-adapter", source="entrypoint"), manager
    )
    context.register_gateway_service("external-core", lambda: object(), critical=True)

    registrations = manager.gateway_service_registrations()
    assert [(item.name, item.critical) for item in registrations] == [
        ("external-core", True)
    ]

    safe_context = GatewayServiceContext("external-core", lambda cursor: None)
    assert not hasattr(safe_context, "gateway_runner")
    assert not hasattr(safe_context, "adapters")
    assert not hasattr(safe_context, "config")
    assert not hasattr(safe_context, "state_db")
    assert not hasattr(safe_context, "token")


def test_pip_entrypoint_discovers_gateway_service_with_temp_hermes_home(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - sample-gateway-service\n",
        encoding="utf-8",
    )
    plugin_root = tmp_path / "site-packages"
    plugin_root.mkdir()
    (plugin_root / "sample_gateway_service.py").write_text(
        "def register(ctx):\n"
        "    ctx.register_gateway_service(\n"
        "        'sample-core', lambda: object(), critical=True\n"
        "    )\n",
        encoding="utf-8",
    )
    dist_info = plugin_root / "sample_gateway_service-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: sample-gateway-service\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[hermes_agent.plugins]\n"
        "sample-gateway-service = sample_gateway_service\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.syspath_prepend(str(plugin_root))
    importlib.invalidate_caches()

    manager = PluginManager()
    manager.discover_and_load()

    registrations = manager.gateway_service_registrations()
    assert [(item.name, item.critical) for item in registrations] == [
        ("sample-core", True)
    ]


def test_stable_ingress_id_prefers_platform_message_id():
    first = stable_ingress_event_id(
        platform="whatsapp",
        platform_message_id="wamid.1",
        session_key="one",
        text="first rendering",
        occurred_at=1.0,
    )
    second = stable_ingress_event_id(
        platform="whatsapp",
        platform_message_id="wamid.1",
        session_key="two",
        text="different rendering",
        occurred_at=2.0,
    )
    assert first == second


class _DeliveryAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, token="test-token"), Platform.WHATSAPP
        )
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id="wamid.out")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


async def _hold_typing(_chat_id, interval=2.0, metadata=None, stop_event=None):
    if stop_event is not None:
        await stop_event.wait()


def _message_event():
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="customer",
            user_id="customer",
            chat_type="dm",
        ),
        message_id="wamid.in",
    )


@pytest.mark.asyncio
async def test_automatic_delivery_uses_one_gate_and_durable_states():
    adapter = _DeliveryAdapter()
    adapter._keep_typing = _hold_typing
    order = []

    async def handler(_event):
        return "automatic answer"

    async def gate(delivery):
        order.append(("gate", delivery.delivery_id))
        return True

    async def state(event):
        order.append((event.state.value, event.delivery_id))

    adapter.set_message_handler(handler)
    adapter.set_gateway_service_handlers(
        pre_automatic_delivery=gate,
        delivery_state=state,
    )
    event = _message_event()
    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent == ["automatic answer"]
    assert [item[0] for item in order] == ["queued", "gate", "sent"]
    assert len({item[1] for item in order}) == 1


@pytest.mark.asyncio
async def test_automatic_delivery_gate_blocks_every_final_send():
    adapter = _DeliveryAdapter()
    adapter._keep_typing = _hold_typing
    states = []

    async def handler(_event):
        return "must not be sent"

    async def reject(_delivery):
        return False

    async def state(event):
        states.append(event.state.value)

    adapter.set_message_handler(handler)
    adapter.set_gateway_service_handlers(
        pre_automatic_delivery=reject,
        delivery_state=state,
    )
    event = _message_event()
    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent == []
    assert states == ["queued", "failed"]
