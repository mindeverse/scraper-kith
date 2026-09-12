"""Supabase client — batch upsert, diff helpers, stale cleanup."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from supabase import create_client

from config import cfg

logger = logging.getLogger(__name__)

STALE_TRACKER_FILE = Path("logs") / "stale_tracker.json"
FAILED_PRODUCTS_LOG = Path("logs") / "failed_products.log"


class SupabaseClient:
    def __init__(self):
        if not cfg.SUPABASE_URL or not cfg.SUPABASE_KEY:
            raise ValueError("Set SUPABASE_URL and SUPABASE_KEY env vars")
        self.client = create_client(cfg.SUPABASE_URL, cfg.SUPABASE_KEY)

    def fetch_existing_products(self, source: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        try:
            response = (
                self.client.table("products")
                .select("*")
                .eq("source", source)
                .execute()
            )
            for row in response.data or []:
                result[row["product_url"]] = row
            logger.info("Fetched %d existing products from DB", len(result))
        except Exception as e:
            logger.error("Failed to fetch existing products: %s", e)
        return result

    def upsert_products(self, products: list[dict[str, Any]], batch_size: int = 5) -> tuple[int, int]:
        total = len(products)
        batches = [products[i : i + batch_size] for i in range(0, total, batch_size)]
        ok = 0
        fail = 0
        for i, batch in enumerate(batches):
            logger.info("Upserting batch %d/%d (%d products)", i + 1, len(batches), len(batch))
            batch_ok = False
            for attempt in range(3):
                try:
                    self.client.table("products").upsert(
                        batch, on_conflict="source,product_url"
                    ).execute()
                    ok += len(batch)
                    batch_ok = True
                    break
                except Exception as e:
                    logger.warning("Batch upsert attempt %d/3 failed: %s", attempt + 1, e)
                    time.sleep(2 ** (attempt + 1))
            if batch_ok:
                continue
            # Fall back to single-row upserts (avoids statement timeout on vector batches)
            logger.warning("Falling back to single-row upsert for batch %d", i + 1)
            for row in batch:
                row_ok = False
                for attempt in range(3):
                    try:
                        self.client.table("products").upsert(
                            row, on_conflict="source,product_url"
                        ).execute()
                        ok += 1
                        row_ok = True
                        break
                    except Exception as e:
                        logger.warning("Single upsert attempt %d/3 failed: %s", attempt + 1, e)
                        time.sleep(2 ** (attempt + 1))
                if not row_ok:
                    fail += 1
                    self._log_failed_products([row])
        return ok, fail

    def delete_product(self, product_id: str) -> None:
        self.client.table("products").delete().eq("id", product_id).execute()

    def _log_failed_products(self, products: list[dict[str, Any]]) -> None:
        try:
            FAILED_PRODUCTS_LOG.parent.mkdir(exist_ok=True)
            with open(FAILED_PRODUCTS_LOG, "a") as f:
                for p in products:
                    f.write(
                        f"{datetime.now(timezone.utc).isoformat()} | "
                        f"{p.get('source', '')} | {p.get('product_url', '')}\n"
                    )
        except Exception as e:
            logger.error("Failed to log failed products: %s", e)


def load_stale_tracker() -> dict[str, int]:
    if STALE_TRACKER_FILE.exists():
        try:
            return json.loads(STALE_TRACKER_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load stale tracker: %s", e)
    return {}


def save_stale_tracker(tracker: dict[str, int]) -> None:
    try:
        STALE_TRACKER_FILE.parent.mkdir(exist_ok=True)
        STALE_TRACKER_FILE.write_text(json.dumps(tracker, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error("Failed to save stale tracker: %s", e)


def handle_stale_products(
    supa: SupabaseClient,
    source: str,
    seen_urls: set[str],
    existing_products: dict[str, dict[str, Any]],
    stale_tracker: dict[str, int],
    threshold: int = 2,
) -> tuple[int, dict[str, int]]:
    deleted = 0
    updated_tracker: dict[str, int] = {}
    for product_url, row in existing_products.items():
        if product_url in seen_urls:
            updated_tracker[product_url] = 0
        else:
            current_misses = stale_tracker.get(product_url, 0) + 1
            updated_tracker[product_url] = current_misses
            if current_misses >= threshold:
                try:
                    supa.delete_product(row["id"])
                    logger.info("Deleted stale product: %s", product_url[:80])
                    deleted += 1
                    updated_tracker.pop(product_url, None)
                except Exception as e:
                    logger.error("Failed to delete stale %s: %s", product_url[:60], e)
    return deleted, updated_tracker