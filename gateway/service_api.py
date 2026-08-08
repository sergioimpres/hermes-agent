"""Public, capability-limited API for long-lived gateway plugin services.

The objects in this module are deliberately data-only.  A service never receives
the ``GatewayRunner``, a platform adapter, user configuration, database handles,
or credentials.  Installed plugins may import this module to type their service
implementation without depending on gateway internals.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping, Optional, Protocol, runtime_checkable


class GatewayEventKind(str, Enum):
    DURABLE_INGRESS = "durable_ingress"
    DELIVERY_STATE = "delivery_state"


class DeliveryState(str, Enum):
    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"


@dataclass(frozen=True)
class MediaDescriptor:
    """Non-sensitive description of an attachment.

    Local paths and fetch URLs are intentionally excluded.  A future media
    capability can expose opaque handles without widening this DTO.
    """

    kind: str
    mime_type: str = ""
    file_name: str = ""
    size_bytes: Optional[int] = None
    sha256: str = ""


@dataclass(frozen=True)
class DurableIngress:
    event_id: str
    platform: str
    platform_message_id: str
    session_id: str
    session_key: str
    chat_id: str
    chat_type: str
    sender_id: str
    sender_name: str
    message_type: str
    text: str
    occurred_at: Optional[float] = None
    media: tuple[MediaDescriptor, ...] = ()


@dataclass(frozen=True)
class AutomaticDelivery:
    delivery_id: str
    platform: str
    session_key: str
    chat_id: str
    reply_to_message_id: str
    text: str
    media_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeliveryStateEvent:
    event_id: str
    delivery_id: str
    state: DeliveryState
    platform: str
    chat_id: str
    platform_message_ids: tuple[str, ...] = ()
    error_code: str = ""


GatewayPayload = DurableIngress | DeliveryStateEvent


@dataclass(frozen=True)
class GatewayEvent:
    """An ordered, durable event delivered to a gateway service."""

    cursor: int
    kind: GatewayEventKind
    payload: GatewayPayload

    def to_wire_dict(self) -> dict[str, Any]:
        payload = _jsonable(asdict(self.payload))
        return {
            "cursor": self.cursor,
            "kind": self.kind.value,
            "payload": payload,
        }


AckCallback = Callable[[int], Awaitable[None]]


class GatewayServiceContext:
    """The complete host capability granted to one service instance."""

    __slots__ = ("_ack", "service_name")

    def __init__(self, service_name: str, ack: AckCallback):
        self.service_name = service_name
        self._ack = ack

    async def ack(self, cursor: int) -> None:
        """Acknowledge every event through ``cursor`` after durable handling."""

        await self._ack(cursor)


@runtime_checkable
class GatewayService(Protocol):
    async def start(self, context: GatewayServiceContext) -> None: ...

    async def stop(self) -> None: ...

    async def handle_event(
        self,
        event: GatewayEvent,
        context: GatewayServiceContext,
    ) -> None: ...

    async def pre_automatic_delivery(self, delivery: AutomaticDelivery) -> bool: ...


GatewayServiceFactory = Callable[[], GatewayService]


@dataclass(frozen=True)
class GatewayServiceRegistration:
    name: str
    factory: GatewayServiceFactory
    critical: bool = False


class GatewayServiceError(RuntimeError):
    """Base failure for the fail-closed gateway service boundary."""


class GatewayServiceNotAcknowledged(GatewayServiceError):
    pass


class GatewayServiceRejectedDelivery(GatewayServiceError):
    pass


def stable_ingress_event_id(
    *,
    platform: str,
    platform_message_id: str,
    session_key: str,
    text: str,
    occurred_at: Optional[float],
) -> str:
    """Return a replay-stable id without exposing message contents in the id."""

    identity = (
        f"{platform}\0{platform_message_id}"
        if platform_message_id
        else f"{platform}\0{session_key}\0{occurred_at!r}\0{text}"
    )
    return "ingress_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def stable_delivery_id(
    *,
    platform: str,
    session_key: str,
    inbound_message_id: str,
    text: str,
) -> str:
    identity = f"{platform}\0{session_key}\0{inbound_message_id}\0{text}"
    return "delivery_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def delivery_state_event_id(delivery_id: str, state: DeliveryState) -> str:
    return f"{delivery_id}:{state.value}"


def payload_to_json(payload: GatewayPayload) -> str:
    return json.dumps(_jsonable(asdict(payload)), ensure_ascii=False, separators=(",", ":"))


def payload_from_json(kind: GatewayEventKind, raw: str) -> GatewayPayload:
    data = json.loads(raw)
    if kind is GatewayEventKind.DURABLE_INGRESS:
        data["media"] = tuple(MediaDescriptor(**item) for item in data.get("media", ()))
        return DurableIngress(**data)
    data["state"] = DeliveryState(data["state"])
    data["platform_message_ids"] = tuple(data.get("platform_message_ids", ()))
    return DeliveryStateEvent(**data)


def safe_metadata_value(value: Any) -> str:
    """Normalize optional scalar metadata without forwarding arbitrary objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return "" if value is None else str(value)
    return ""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
