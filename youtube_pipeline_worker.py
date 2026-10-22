#!/usr/bin/env python3
"""Production-only daily scheduler for the YouTube Shorts pipeline."""

from __future__ import annotations

import logging
import os
import signal
import argparse
import threading
from datetime import date, datetime, time, timedelta, timezone

import psycopg

from cron_pipeline import (
    TIMEZONE,
    connect_database,
    load_target_connection,
    parse_args,
    required_env,
    run_pipeline,
)


LOGGER = logging.getLogger("youtube_pipeline_worker")
SCHEDULE_TIME = time(16, 0)
RETRY_DELAY = timedelta(minutes=15)
LOCK_KEY = (752083, 190401)
STOP = threading.Event()


def local_publish_window(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time(18, 0), tzinfo=TIMEZONE)
    end = datetime.combine(day + timedelta(days=1), time(0, 0), tzinfo=TIMEZONE)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def seconds_until_next_schedule(now: datetime) -> float:
    local_now = now.astimezone(TIMEZONE)
    target = datetime.combine(local_now.date(), SCHEDULE_TIME, tzinfo=TIMEZONE)
    if local_now >= target:
        target += timedelta(days=1)
    return max(1.0, (target - local_now).total_seconds())


def has_slots_for_local_day(channel_id: str, day: date) -> bool:
    start, end = local_publish_window(day)
    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*) AS slot_count
                FROM youtube_pipeline_slots
                WHERE channel_id = %s
                  AND publish_at >= %s
                  AND publish_at < %s
                """,
                (channel_id, start, end),
            )
            result = cursor.fetchone()
    return bool(result and result["slot_count"])


def acquire_lock() -> psycopg.Connection | None:
    connection = psycopg.connect(required_env("DATABASE_URL"), autocommit=True)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_try_advisory_lock(%s, %s)",
                LOCK_KEY,
            )
            locked = cursor.fetchone()[0]
    except Exception:
        connection.close()
        raise
    if not locked:
        connection.close()
        return None
    return connection


def run_daily_pipeline(day: date) -> bool:
    channel = load_target_connection()
    if has_slots_for_local_day(channel["channel_id"], day):
        LOGGER.warning(
            "Pipeline slots already exist for %s; skipping to prevent duplicate uploads.",
            day.isoformat(),
        )
        return True

    LOGGER.info(
        "Starting the daily pipeline for %s at %s.",
        day.isoformat(),
        TIMEZONE,
    )
    try:
        manifest = run_pipeline(parse_args(["--execute"]))
        LOGGER.info("Daily pipeline completed. Manifest: %s", manifest)
        return True
    except Exception:
        LOGGER.exception("Daily pipeline failed.")
        if has_slots_for_local_day(channel["channel_id"], day):
            LOGGER.error(
                "Some slots were recorded for %s; the worker will not retry them "
                "automatically to avoid duplicate uploads.",
                day.isoformat(),
            )
            return True
        return False


def run_scheduler() -> None:
    completed_day: date | None = None
    retry_after: datetime | None = None

    while not STOP.is_set():
        now = datetime.now(TIMEZONE)
        today = now.date()
        if retry_after and retry_after.date() != today:
            retry_after = None
        if completed_day and completed_day != today:
            completed_day = None

        if now.hour == SCHEDULE_TIME.hour and completed_day != today:
            if retry_after and now < retry_after:
                STOP.wait(60)
                continue
            try:
                completed = run_daily_pipeline(today)
            except Exception:
                LOGGER.exception("The daily scheduler attempt could not start.")
                completed = False
            if completed:
                completed_day = today
                retry_after = None
            else:
                retry_after = now + RETRY_DELAY
                LOGGER.warning(
                    "No publishing slots were reserved; will retry at or after %s.",
                    retry_after.isoformat(),
                )
            STOP.wait(60)
            continue

        if now.hour == SCHEDULE_TIME.hour:
            STOP.wait(60)
        else:
            STOP.wait(min(300, seconds_until_next_schedule(now)))


def _request_shutdown(signum: int, _frame: object) -> None:
    LOGGER.info("Received signal %s; finishing the current operation before exit.", signum)
    STOP.set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run today's pipeline once and exit; used by the authenticated API trigger.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if os.getenv("NODE_ENV") != "production":
        LOGGER.error("Refusing to start the YouTube worker outside production.")
        return 2
    if not args.run_once and os.getenv("YOUTUBE_PIPELINE_WORKER_ENABLED") != "true":
        LOGGER.error("The production YouTube worker is not enabled.")
        return 2
    if os.getenv("YOUTUBE_PIPELINE_UPLOADS_ENABLED") != "true":
        LOGGER.error("Live uploads are not enabled for the YouTube worker.")
        return 2

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    if args.run_once:
        try:
            lock_connection = acquire_lock()
        except Exception:
            LOGGER.exception("Could not acquire the pipeline lock for the API trigger.")
            return 1
        if not lock_connection:
            LOGGER.warning("Another pipeline run already holds the scheduler lock.")
            return 3
        try:
            completed = run_daily_pipeline(datetime.now(TIMEZONE).date())
            return 0 if completed else 1
        except Exception:
            LOGGER.exception("The API-triggered pipeline run failed.")
            return 1
        finally:
            lock_connection.close()
            LOGGER.info("Released the API-triggered pipeline lock.")

    lock_connection: psycopg.Connection | None = None
    try:
        while not STOP.is_set():
            try:
                lock_connection = acquire_lock()
                if lock_connection:
                    LOGGER.info("Acquired the single-worker database lock.")
                    break
                LOGGER.info("Another worker owns the scheduler lock; retrying shortly.")
            except Exception:
                LOGGER.exception("Could not acquire the scheduler lock.")
            STOP.wait(60)
        if STOP.is_set():
            return 0
        run_scheduler()
    finally:
        if lock_connection:
            lock_connection.close()
            LOGGER.info("Released the single-worker database lock.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
