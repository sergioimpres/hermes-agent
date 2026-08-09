"""Async lifecycle and durable ordered outbox for gateway plugin services."""

from __future__ import annotations

import asyncio
import inspect
import logging
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional

from gateway.service_api import (
    AutomaticDelivery,
    GatewayEvent,
    GatewayEventKind,
    GatewayPayload,
    GatewayServiceContext,
    GatewayServiceError,
    GatewayServiceNotAcknowledged,
    GatewayServiceRegistration,
    payload_from_json,
    payload_to_json,
)


logger = logging.getLogger(__name__)


class GatewayServiceJournal:
    """Append-only event identities plus compactable payloads and ACK cursors."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS gateway_service_events (
                cursor INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                payload_json TEXT,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS gateway_service_acks (
                service_name TEXT PRIMARY KEY,
                cursor INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self._db.commit()

    def ensure_service(self, name: str) -> None:
        """Register a service without replaying events from before installation."""

        row = self._db.execute(
            "SELECT 1 FROM gateway_service_acks WHERE service_name = ?", (name,)
        ).fetchone()
        if row is not None:
            return
        max_cursor = int(
            self._db.execute(
                "SELECT COALESCE(MAX(cursor), 0) FROM gateway_service_events"
            ).fetchone()[0]
        )
        self._db.execute(
            "INSERT INTO gateway_service_acks(service_name, cursor) VALUES (?, ?)",
            (name, max_cursor),
        )
        self._db.commit()

    def retain_services(self, names: Iterable[str]) -> None:
        """Drop obsolete ACK cursors so removed services cannot pin payloads."""

        retained = tuple(dict.fromkeys(str(name) for name in names))
        if retained:
            placeholders = ",".join("?" for _ in retained)
            self._db.execute(
                f"DELETE FROM gateway_service_acks WHERE service_name NOT IN ({placeholders})",
                retained,
            )
        else:
            self._db.execute("DELETE FROM gateway_service_acks")
        self._compact_acknowledged_payloads()
        self._db.commit()

    def forget_service(self, name: str) -> None:
        """Remove an unavailable optional service from the compaction floor."""

        self._db.execute(
            "DELETE FROM gateway_service_acks WHERE service_name = ?", (name,)
        )
        self._compact_acknowledged_payloads()
        self._db.commit()

    def append(
        self,
        event_id: str,
        kind: GatewayEventKind,
        payload: GatewayPayload,
    ) -> GatewayEvent:
        raw = payload_to_json(payload)
        self._db.execute(
            """
            INSERT INTO gateway_service_events(event_id, kind, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(event_id) DO NOTHING
            """,
            (event_id, kind.value, raw, time.time()),
        )
        self._db.commit()
        row = self._db.execute(
            "SELECT cursor, kind, payload_json FROM gateway_service_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            raise GatewayServiceError(f"event journal lost {event_id}")
        cursor, stored_kind, stored_payload = row
        if stored_payload is None:
            # A duplicate already handled by every service is an idempotent no-op.
            return GatewayEvent(int(cursor), GatewayEventKind(stored_kind), payload)
        return GatewayEvent(
            int(cursor),
            GatewayEventKind(stored_kind),
            payload_from_json(GatewayEventKind(stored_kind), stored_payload),
        )

    def acknowledged_cursor(self, service_name: str) -> int:
        row = self._db.execute(
            "SELECT cursor FROM gateway_service_acks WHERE service_name = ?",
            (service_name,),
        ).fetchone()
        return int(row[0]) if row else 0

    def acknowledge(self, service_name: str, cursor: int) -> None:
        current = self.acknowledged_cursor(service_name)
        if cursor < current:
            return
        max_cursor = int(
            self._db.execute(
                "SELECT COALESCE(MAX(cursor), 0) FROM gateway_service_events"
            ).fetchone()[0]
        )
        if cursor > max_cursor:
            raise GatewayServiceError(
                f"service {service_name!r} acknowledged unknown cursor {cursor}"
            )
        self._db.execute(
            "UPDATE gateway_service_acks SET cursor = ? WHERE service_name = ?",
            (cursor, service_name),
        )
        self._compact_acknowledged_payloads()
        self._db.commit()

    def pending(self, service_name: str) -> list[GatewayEvent]:
        cursor = self.acknowledged_cursor(service_name)
        rows = self._db.execute(
            """
            SELECT cursor, kind, payload_json
            FROM gateway_service_events
            WHERE cursor > ? AND payload_json IS NOT NULL
            ORDER BY cursor ASC
            """,
            (cursor,),
        ).fetchall()
        return [
            GatewayEvent(
                int(row_cursor),
                GatewayEventKind(kind),
                payload_from_json(GatewayEventKind(kind), payload_json),
            )
            for row_cursor, kind, payload_json in rows
        ]

    def _compact_acknowledged_payloads(self) -> None:
        row = self._db.execute(
            "SELECT MIN(cursor) FROM gateway_service_acks"
        ).fetchone()
        if not row or row[0] is None:
            # With no registered consumers, no payload has a replay owner.
            # Keep event-id tombstones for idempotency but drop message bodies.
            self._db.execute(
                "UPDATE gateway_service_events SET payload_json = NULL"
            )
            return
        self._db.execute(
            "UPDATE gateway_service_events SET payload_json = NULL WHERE cursor <= ?",
            (int(row[0]),),
        )

    def close(self) -> None:
        self._db.close()


class GatewayServiceRuntime:
    """Own service instances, their safe contexts, and ordered event delivery."""

    def __init__(
        self,
        registrations: Iterable[GatewayServiceRegistration],
        journal_path: Path,
    ):
        registrations = tuple(registrations)
        names = [registration.name for registration in registrations]
        if len(names) != len(set(names)):
            raise GatewayServiceError("gateway service names must be unique")
        self._registrations = registrations
        self._journal = GatewayServiceJournal(journal_path)
        self._services: dict[str, object] = {}
        self._contexts: dict[str, GatewayServiceContext] = {}
        self._publish_lock = asyncio.Lock()
        self._started = False

    @property
    def active(self) -> bool:
        return self._started and bool(self._services)

    async def start(self) -> None:
        if self._started:
            return
        started: list[str] = []
        try:
            # Create every ACK row before the first service can replay and
            # compact payloads. Otherwise service A could compact an event
            # before a newly installed service B has an ACK cursor to hold the
            # compaction floor back.
            for registration in self._registrations:
                self._journal.ensure_service(registration.name)
            self._journal.retain_services(
                registration.name for registration in self._registrations
            )
            for registration in self._registrations:
                context = GatewayServiceContext(
                    registration.name,
                    lambda cursor, name=registration.name: self._ack(name, cursor),
                )
                service = registration.factory()
                self._contexts[registration.name] = context
                self._services[registration.name] = service
                try:
                    start = getattr(service, "start", None)
                    if callable(start):
                        await _maybe_await(start(context))
                    started.append(registration.name)
                    await self._replay_service(registration.name)
                except Exception:
                    stop = getattr(service, "stop", None)
                    if callable(stop):
                        try:
                            await _maybe_await(stop())
                        except Exception:
                            logger.warning(
                                "gateway service %s failed during startup cleanup",
                                registration.name,
                                exc_info=True,
                            )
                    self._services.pop(registration.name, None)
                    self._contexts.pop(registration.name, None)
                    if registration.critical:
                        raise
                    self._journal.forget_service(registration.name)
                    logger.warning(
                        "optional gateway service %s failed to start",
                        registration.name,
                        exc_info=True,
                    )
        except Exception:
            await self._stop_names(reversed(started))
            self._services.clear()
            self._contexts.clear()
            raise
        self._started = True

    async def stop(self) -> None:
        if self._services:
            await self._stop_names(reversed(tuple(self._services)))
        self._services.clear()
        self._contexts.clear()
        self._started = False
        self._journal.close()

    async def publish(
        self,
        event_id: str,
        kind: GatewayEventKind,
        payload: GatewayPayload,
    ) -> GatewayEvent:
        if not self._services:
            return GatewayEvent(0, kind, payload)
        # A service handles one cursor at a time. Without this lock, two
        # concurrent platform ingress tasks can append cursors 1 and 2, then
        # deliver cursor 2 while cursor 1 is still awaiting its remote ACK.
        async with self._publish_lock:
            event = self._journal.append(event_id, kind, payload)
            for name in self._services:
                # Drain every earlier unacknowledged cursor before the newly
                # appended event. A transient failure must not let the next
                # ingress leapfrog the event that failed.
                await self._replay_service(name)
            return event

    async def pre_automatic_delivery(self, delivery: AutomaticDelivery) -> bool:
        for name, service in self._services.items():
            callback = getattr(service, "pre_automatic_delivery", None)
            if not callable(callback):
                continue
            try:
                allowed = await _maybe_await(callback(delivery))
            except Exception as exc:
                raise GatewayServiceError(
                    f"gateway service {name!r} pre-delivery check failed"
                ) from exc
            if allowed is not True:
                return False
        return True

    async def _replay_service(self, name: str) -> None:
        for event in self._journal.pending(name):
            await self._deliver(name, event)

    async def _deliver(self, name: str, event: GatewayEvent) -> None:
        service = self._services[name]
        context = self._contexts[name]
        callback = getattr(service, "handle_event", None)
        if not callable(callback):
            await context.ack(event.cursor)
            return
        try:
            await _maybe_await(callback(event, context))
        except Exception as exc:
            raise GatewayServiceError(
                f"gateway service {name!r} failed at cursor {event.cursor}"
            ) from exc
        if self._journal.acknowledged_cursor(name) < event.cursor:
            raise GatewayServiceNotAcknowledged(
                f"gateway service {name!r} returned without ACK for cursor {event.cursor}"
            )

    async def _ack(self, name: str, cursor: int) -> None:
        self._journal.acknowledge(name, int(cursor))

    async def _stop_names(self, names: Iterable[str]) -> None:
        for name in names:
            service = self._services.get(name)
            stop = getattr(service, "stop", None)
            if not callable(stop):
                continue
            try:
                await _maybe_await(stop())
            except Exception:
                logger.warning("gateway service %s failed during stop", name, exc_info=True)


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value
