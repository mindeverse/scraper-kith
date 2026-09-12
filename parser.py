"""Vwoollo Shopify parser — /collections/{handle}/products.json with EUR cookie."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any, Optional
from urllib.parse import urljoin

import requests

from config import cfg

logger = logging.getLogger(__name__)

BACK_KEYWORDS = ("back", "rear", "_b.", "_back", "-back", "backview", "back_view")


def _headers() -> dict[str, str]:
    return {
        "User-Agent": cfg.USER_AGENT,
        "Accept": "application/json,text/html,*/*",
        # Force EUR presentment on Shopify storefront JSON
        "Cookie": f"cart_currency={cfg.CURRENCY}; localization=EU",
    }


def _stable_id(product_url: str) -> str:
    digest = hashlib.sha256(f"{cfg.SOURCE}:{product_url}".encode()).hexdigest()[:24]
    return f"vwoollo_{digest}"


def _money(amount: Optional[float | int | str], currency: str | None = None) -> Optional[str]:
    if amount is None:
        return None
    currency = currency or cfg.CURRENCY
    try:
        val = float(amount)
    except (TypeError, ValueError):
        return None
    return f"{val:.2f}{currency}"


def _parse_price_value(raw: Any) -> Optional[float]:
    if raw is None or raw == "":
        return None
    try:
        if isinstance(raw, str):
            return float(raw)
        raw_f = float(raw)
        # Heuristic: integers >= 100 from .js are cents
        if isinstance(raw, int) and raw_f >= 100:
            return raw_f / 100.0
        return raw_f
    except (TypeError, ValueError):
        return None


def _detect_back_image(images: list[dict[str, Any]], front_src: str) -> Optional[str]:
    for img in images:
        src = img.get("src") or ""
        alt = (img.get("alt") or "").lower()
        blob = f"{src} {alt}".lower()
        if src and src != front_src and any(k in blob for k in BACK_KEYWORDS):
            return src
    return None


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("//"):
        return f"https:{url}"
    if url.startswith("/"):
        return urljoin(cfg.BASE_URL + "/", url)
    return url


def _category_handle(category_url: str) -> str:
    return category_url.rstrip("/").split("/")[-1]


def _infer_gender(handle: str, category: Optional[str], tags: list[str]) -> Optional[str]:
    blob = " ".join(
        [
            handle or "",
            category or "",
            " ".join(tags),
        ]
    ).lower()
    # Prefer women* before men* (substring "men" appears inside "women")
    if re.search(r"\bwomens?\b|\bwoman\b|women", blob):
        return "Women"
    if handle.startswith("womens-") or (category or "").lower().startswith("womens"):
        return "Women"
    if handle.startswith("mens-") or (category or "").lower().startswith("mens"):
        return "Men"
    if re.search(r"\bmens?\b|\bman\b", blob):
        return "Men"
    return getattr(cfg, "GENDER_DEFAULT", None)


def fetch_collection_products(category_url: str) -> list[dict[str, Any]]:
    """Paginate Shopify products.json until an empty page."""
    handle = _category_handle(category_url)
    display = cfg.CATEGORY_DISPLAY.get(handle, handle.replace("-", " ").title())
    limit = getattr(cfg, "PRODUCTS_JSON_LIMIT", 50)
    products: list[dict[str, Any]] = []
    page = 1

    while True:
        url = (
            f"{cfg.BASE_URL}/collections/{handle}/products.json"
            f"?limit={limit}&page={page}&currency={cfg.CURRENCY}"
        )
        time.sleep(cfg.RATE_LIMIT_DELAY)
        try:
            resp = requests.get(url, headers=_headers(), timeout=cfg.REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Failed products.json %s page=%s: %s", handle, page, e)
            break

        batch = data.get("products") or []
        if not batch:
            logger.info("Category %s page=%d empty — stop pagination", display, page)
            break

        for raw in batch:
            parsed = parse_shopify_product(raw, collection_handle=handle)
            if parsed:
                products.append(parsed)

        if len(batch) < limit:
            break
        page += 1

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for p in products:
        if p["product_url"] not in seen:
            seen.add(p["product_url"])
            unique.append(p)
    logger.info("Category %s: %d products", display, len(unique))
    return unique


def parse_shopify_product(
    raw: dict[str, Any],
    category: Optional[str] = "",
    collection_handle: str = "",
) -> Optional[dict[str, Any]]:
    handle = raw.get("handle") or ""
    if not handle:
        return None
    product_url = f"{cfg.BASE_URL}/products/{handle}"
    title = (raw.get("title") or "").strip() or "Unknown"
    images = raw.get("images") or []
    front = ""
    if images:
        front = _normalize_url(images[0].get("src") or "")
    if not front and raw.get("image"):
        front = _normalize_url((raw.get("image") or {}).get("src") or "")
    if not front:
        logger.warning("Skip %s: no image", product_url)
        return None

    variants = raw.get("variants") or []
    prices: list[float] = []
    compare_prices: list[float] = []
    sizes: list[str] = []
    colors: list[str] = []
    available_any = False
    for v in variants:
        if v.get("available"):
            available_any = True
        pv = _parse_price_value(v.get("price"))
        if pv is not None:
            prices.append(pv)
        cv = _parse_price_value(v.get("compare_at_price"))
        if cv is not None:
            compare_prices.append(cv)
        opt1 = (v.get("option1") or "").strip()
        opt2 = (v.get("option2") or "").strip()
        if opt1 and opt1 not in sizes:
            sizes.append(opt1)
        if opt2 and opt2 not in colors:
            colors.append(opt2)

    # Prefer original (compare_at) as price when on sale; amounts already major units in EUR
    sale_price = None
    price = None
    if compare_prices and prices and min(compare_prices) > min(prices):
        price = _money(min(compare_prices))
        sale_price = _money(min(prices))
    elif prices:
        price = _money(min(prices))

    body = raw.get("body_html") or ""
    description = re.sub(r"<[^>]+>", " ", body)
    description = re.sub(r"\s+", " ", description).strip() or None

    back_image_url = _detect_back_image(images, front)
    additional = []
    for img in images[1:]:
        src = _normalize_url(img.get("src") or "")
        if src and src != front:
            additional.append(src)
    if back_image_url and back_image_url not in additional:
        additional.append(back_image_url)
    additional_images = " , ".join(additional) if additional else None

    tags_raw = raw.get("tags") or []
    if isinstance(tags_raw, str):
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    else:
        tags = list(tags_raw)

    product_type = (raw.get("product_type") or "").strip() or None

    gender = _infer_gender(collection_handle, category, tags)

    metadata = {
        "handle": handle,
        "vendor": raw.get("vendor"),
        "product_type": product_type,
        "sku": (variants[0].get("sku") if variants else None),
        "colors": colors,
        "sizes": sizes,
        "availability": available_any,
        "tags": tags,
        "currency": cfg.CURRENCY,
        "scrape_source": "products.json",
        "collection_handle": collection_handle or None,
    }

    # Exclude product_type Insurance and Gift Card
    excluded_types = {"insurance", "gift card"}
    if product_type and product_type.lower() in excluded_types:
        logger.info("Exclude product_type %s for %s", product_type, product_url)
        return None

    return {
        "id": _stable_id(product_url),
        "source": cfg.SOURCE,
        "product_url": product_url,
        "affiliate_url": None,
        "image_url": front,
        "compressed_image_url": None,
        "back_image_url": back_image_url,
        "brand": cfg.BRAND_COLUMN,
        "title": title,
        "description": description,
        "category": category,
        "gender": gender,
        "price": price,
        "sale": sale_price,
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "size": ", ".join(sizes) if sizes else None,
        "second_hand": cfg.SECOND_HAND,
        "country": "IE",  # Vwoollo is Ireland-based
        "tags": tags or None,
        "additional_images": additional_images,
        "other": None,
    }


def fetch_all_products_json() -> list[dict[str, Any]]:
    """Paginate store-wide /products.json for full catalog coverage (~9.7k SKUs)."""
    limit = getattr(cfg, "PRODUCTS_JSON_LIMIT", 50)
    products: list[dict[str, Any]] = []
    page = 1
    while True:
        url = f"{cfg.BASE_URL}/products.json?limit={limit}&page={page}"
        time.sleep(cfg.RATE_LIMIT_DELAY)
        try:
            resp = requests.get(url, headers=_headers(), timeout=cfg.REQUEST_TIMEOUT)
            resp.raise_for_status()
            batch = (resp.json() or {}).get("products") or []
        except Exception as e:
            logger.error("products.json page %s failed: %s", page, e)
            break
        if not batch:
            break
        for raw in batch:
            parsed = parse_shopify_product(raw, category="All", collection_handle="all")
            if parsed:
                products.append(parsed)
        if len(batch) < limit:
            break
        page += 1
    logger.info("products.json full catalog: %d products", len(products))
    return products


def scrape_all_categories() -> list[dict[str, Any]]:
    all_products: list[dict[str, Any]] = []
    seen: set[str] = set()
    # Prefer full catalog crawl first for coverage
    for p in fetch_all_products_json():
        seen.add(p["product_url"])
        all_products.append(p)
    for url in cfg.CATEGORY_URLS:
        handle = _category_handle(url)
        for p in fetch_collection_products(url):
            if p["product_url"] in seen:
                # Merge duplicates; prefer explicit category/leaf data
                existing = next(x for x in all_products if x["product_url"] == p["product_url"])
                cats = {c.strip() for c in (existing.get("category") or "").split(",") if c.strip()}
                if p.get("category"):
                    cats.add(p["category"])
                existing["category"] = ", ".join(sorted(cats)) if cats else existing.get("category")
                # Prefer explicit gender from leaf collections
                if p.get("gender") and not existing.get("gender"):
                    existing["gender"] = p["gender"]
                continue
            seen.add(p["product_url"])
            all_products.append(p)
    logger.info("Total unique products: %d", len(all_products))
    return all_products