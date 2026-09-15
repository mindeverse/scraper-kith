"""Image and text embeddings using SigLIP (768-dim) - local model."""
import gc
import io
import json
import logging
import math
import os
import pickle
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

import requests
import torch
from PIL import Image

try:
    from transformers import SiglipImageProcessorPil as SiglipImageProcessor
except ImportError:
    from transformers import SiglipImageProcessor
from transformers import SiglipModel, SiglipTokenizer

logger = logging.getLogger(__name__)

logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

MODEL_NAME = "google/siglip-base-patch16-384"
EMBEDDING_DIM = 768
DOWNLOAD_WORKERS = 10
INFERENCE_BATCH_SIZE = 64
CHECKPOINT_DIR = "logs"
BATCH_DOWNLOAD = 200

_model = None
_image_processor = None
_tokenizer = None
_device = None


def _get_device():
    global _device
    if _device is None:
        _device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return _device


def _load_model():
    global _model, _image_processor, _tokenizer
    if _model is None:
        num_cpus = os.cpu_count() or 2
        torch.set_num_threads(num_cpus)
        logger.info("Loading SigLIP model %s (threads=%d)...", MODEL_NAME, num_cpus)
        _image_processor = SiglipImageProcessor.from_pretrained(MODEL_NAME)
        _tokenizer = SiglipTokenizer.from_pretrained(MODEL_NAME)
        _model = SiglipModel.from_pretrained(MODEL_NAME)
        _model.to(_get_device())
        _model.eval()
    return _model, _image_processor, _tokenizer


def _download_image(image_url: str) -> Optional[Image.Image]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    }
    try:
        resp = requests.get(image_url, timeout=15, headers=headers)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logger.warning("Failed to download image %s: %s", image_url, e)
        return None


def _embed_images_batch(images: list[Image.Image]) -> list[Optional[list[float]]]:
    model, image_processor, _ = _load_model()
    device = _get_device()
    results: list[Optional[list[float]]] = [None] * len(images)
    try:
        inputs = image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.get_image_features(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb_tensor = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            emb_tensor = outputs.last_hidden_state[:, 0, :]
        else:
            emb_tensor = outputs
        for i in range(emb_tensor.shape[0]):
            results[i] = emb_tensor[i].cpu().float().numpy().flatten().tolist()
    except Exception as e:
        logger.warning("Batch embed failed (size %d): %s", len(images), e)
        for i, img in enumerate(images):
            try:
                results[i] = _embed_single(img)
            except Exception:
                pass
    return results


def _embed_single(image: Image.Image) -> Optional[list[float]]:
    model, image_processor, _ = _load_model()
    device = _get_device()
    try:
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.get_image_features(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb_tensor = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            emb_tensor = outputs.last_hidden_state[:, 0, :]
        else:
            emb_tensor = outputs
        return emb_tensor.cpu().float().numpy().flatten().tolist()
    except Exception as e:
        logger.warning("Failed to embed image: %s", e)
        return None


def get_text_embedding(text: str) -> Optional[list[float]]:
    if not text or not str(text).strip():
        return None
    model, _, tokenizer = _load_model()
    device = _get_device()
    try:
        inputs = tokenizer(
            text=[str(text).strip()],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.get_text_features(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb_tensor = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            emb_tensor = outputs.last_hidden_state[:, 0, :]
        else:
            emb_tensor = outputs
        return emb_tensor.cpu().float().numpy().flatten().tolist()
    except Exception as e:
        logger.warning("Failed to embed text: %s", e)
        return None


def get_text_embeddings_batch(texts: list[str], batch_size: int = 32) -> list[Optional[list[float]]]:
    """Embed multiple texts in batches for much faster throughput."""
    if not texts:
        return []
    model, _, tokenizer = _load_model()
    device = _get_device()
    all_results: list[Optional[list[float]]] = [None] * len(texts)

    valid_indices = [i for i, t in enumerate(texts) if t and str(t).strip()]
    if not valid_indices:
        return all_results

    for batch_start in range(0, len(valid_indices), batch_size):
        batch_indices = valid_indices[batch_start: batch_start + batch_size]
        batch_texts = [str(texts[i]).strip() for i in batch_indices]

        try:
            inputs = tokenizer(
                text=batch_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model.get_text_features(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb_tensor = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                emb_tensor = outputs.last_hidden_state[:, 0, :]
            else:
                emb_tensor = outputs
            for j, idx in enumerate(batch_indices):
                all_results[idx] = emb_tensor[j].cpu().float().numpy().flatten().tolist()
        except Exception as e:
            logger.warning("Batch text embed failed (size %d): %s", len(batch_texts), e)
            for idx in batch_indices:
                try:
                    all_results[idx] = get_text_embedding(texts[idx])
                except Exception:
                    pass

    return all_results


def _build_info_text(product: dict[str, Any]) -> str:
    parts = []
    parts.append(f"Brand: {product.get('brand', 'Vwoollo')}")
    parts.append(f"Product: {product.get('title', '')}")
    if product.get("category"):
        parts.append(f"Category: {product['category']}")
    if product.get("gender"):
        parts.append(f"Gender: {product['gender']}")
    if product.get("price"):
        parts.append(f"Price: {product['price']}")
    if product.get("sale"):
        parts.append(f"Sale price: {product['sale']}")
    if product.get("description"):
        parts.append(f"Description: {product['description']}")
    try:
        metadata = json.loads(product.get("metadata") or "{}")
        if metadata.get("colors"):
            parts.append(f"Color: {', '.join(metadata['colors'])}")
        if metadata.get("sizes"):
            parts.append(f"Sizes: {', '.join(metadata['sizes'])}")
    except (json.JSONDecodeError, TypeError):
        pass
    return " | ".join(parts)


def _download_batch(urls: list[str]) -> dict[str, Optional[Image.Image]]:
    results: dict[str, Optional[Image.Image]] = {}
    completed = 0
    total = len(urls)
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        future_to_url = {executor.submit(_download_image, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception as e:
                logger.warning("Download failed for %s: %s", url, e)
                results[url] = None
            completed += 1
            if completed % 50 == 0:
                logger.info("    Download progress: %d/%d", completed, total)
    return results


# ── Checkpoint helpers ──────────────────────────────────────────────

_checkpoint_source: str = ""
_checkpoint_data: dict[str, dict[str, Any]] = {}


def _checkpoint_path(source: str) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"embed_checkpoint_{source}.pkl")


def load_checkpoint(source: str) -> dict[str, dict[str, Any]]:
    path = _checkpoint_path(source)
    if not os.path.exists(path):
        json_path = os.path.join(CHECKPOINT_DIR, f"embed_checkpoint_{source}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                logger.info("Migrated legacy JSON checkpoint: %d products", len(data))
                _save_checkpoint(source, data)
                os.remove(json_path)
                return data
            except Exception as e:
                logger.warning("Failed to load legacy checkpoint: %s", e)
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info("Loaded checkpoint: %d products already embedded", len(data))
        return data
    except Exception as e:
        logger.warning("Failed to load checkpoint: %s", e)
        return {}


def _save_checkpoint(source: str, data: dict[str, dict[str, Any]]):
    path = _checkpoint_path(source)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def clear_checkpoint(source: str):
    path = _checkpoint_path(source)
    if os.path.exists(path):
        os.remove(path)
        logger.info("Cleared embedding checkpoint")


def _log_memory(label: str):
    try:
        import resource
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if raw > 1_000_000:
            mb = raw / (1024 * 1024)
        else:
            mb = raw / 1024
        logger.info("  [MEM %s] RSS: %.0f MB", label, mb)
    except Exception:
        pass


def _setup_signal_handlers(source: str):
    global _checkpoint_source, _checkpoint_data
    _checkpoint_source = source

    def handler(signum, frame):
        logger.warning("Received signal %d, saving checkpoint before exit...", signum)
        if _checkpoint_data:
            try:
                _save_checkpoint(_checkpoint_source, _checkpoint_data)
                logger.warning("Checkpoint saved (%d products)", len(_checkpoint_data))
            except Exception as e:
                logger.error("Failed to save checkpoint: %s", e)
        sys.exit(143)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


# ── Main embedding pipeline ────────────────────────────────────────

def embed_products(
    products: list[dict[str, Any]],
    existing_embeddings: dict[str, dict[str, Any]] | None = None,
    source: str = "scraper-kith",
    on_batch_done: Callable[[list[dict[str, Any]], int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if existing_embeddings is None:
        existing_embeddings = {}

    stats = {
        "front_embeddings": 0,
        "back_embeddings": 0,
        "text_embeddings": 0,
        "skipped": 0,
    }

    checkpoint = load_checkpoint(source)
    for url, ckpt_data in checkpoint.items():
        if url not in existing_embeddings:
            existing_embeddings[url] = ckpt_data

    _load_model()
    _log_memory("after model load")
    _setup_signal_handlers(source)

    all_needed: list[tuple[int, str, str]] = []
    for i, product in enumerate(products):
        product_url = product.get("product_url", "")
        existing = existing_embeddings.get(product_url, {})

        image_url = product.get("image_url", "")
        existing_image_url = existing.get("image_url", "")
        if image_url and (
            not existing
            or image_url != existing_image_url
            or not existing.get("image_embedding")
        ):
            all_needed.append((i, image_url, "front"))

        back_url = product.get("back_image_url")
        existing_back_url = existing.get("back_image_url")
        if back_url and (
            not existing
            or back_url != existing_back_url
            or not existing.get("back_image_embedding")
        ):
            all_needed.append((i, back_url, "back"))

    logger.info("Need to embed %d images total", len(all_needed))
    _log_memory("before embedding loop")

    for i, product in enumerate(products):
        product_url = product.get("product_url", "")
        existing = existing_embeddings.get(product_url, {})
        if existing:
            if existing.get("image_embedding") and not product.get("image_embedding"):
                product["image_embedding"] = existing["image_embedding"]
            if existing.get("back_image_embedding") and not product.get("back_image_embedding"):
                product["back_image_embedding"] = existing["back_image_embedding"]
            if existing.get("info_embedding") and not product.get("info_embedding"):
                product["info_embedding"] = existing["info_embedding"]

    global _checkpoint_data
    checkpoint_data: dict[str, dict[str, Any]] = dict(checkpoint)
    _checkpoint_data = checkpoint_data
    completed_urls: set[str] = set(checkpoint.keys())

    if not all_needed:
        logger.info("All images already embedded (from checkpoint/existing), skipping image embedding")
    else:
        total_batches = math.ceil(len(all_needed) / BATCH_DOWNLOAD)
        downloaded_urls: set[str] = set()

        download_batches: list[list[tuple[int, str, str]]] = []
        for batch_start in range(0, len(all_needed), BATCH_DOWNLOAD):
            download_batches.append(all_needed[batch_start : batch_start + BATCH_DOWNLOAD])

        pending_download: dict[str, Optional[Image.Image]] = {}
        download_executor = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
        next_download_future = None

        def _start_download(batch: list[tuple[int, str, str]]):
            urls = list(set(url for _, url, _ in batch if url not in downloaded_urls))
            if not urls:
                return None
            return download_executor.submit(_download_batch, urls)

        if download_batches:
            next_download_future = _start_download(download_batches[0])

        for batch_num, batch in enumerate(download_batches, 1):
            if next_download_future is not None:
                logger.info("  Batch %d/%d: waiting for downloads...", batch_num, total_batches)
                try:
                    pending_download = next_download_future.result()
                except Exception as e:
                    logger.error("Download batch failed: %s", e)
                    pending_download = {}

            if batch_num < len(download_batches):
                next_download_future = _start_download(download_batches[batch_num])
            else:
                next_download_future = None

            images_to_embed: list[tuple[int, str, str, Image.Image]] = []
            for idx, url, view_type in batch:
                product = products[idx]
                product_url = product.get("product_url", "")
                existing = existing_embeddings.get(product_url, {})

                if view_type == "front":
                    image_url = product.get("image_url", "")
                    existing_image_url = existing.get("image_url", "")
                    if image_url and (
                        not existing
                        or image_url != existing_image_url
                        or not existing.get("image_embedding")
                    ):
                        image = pending_download.get(image_url)
                        if image:
                            images_to_embed.append((idx, image_url, "front", image))
                        else:
                            stats["skipped"] += 1
                elif view_type == "back":
                    back_url = product.get("back_image_url")
                    existing_back_url = existing.get("back_image_url")
                    if back_url and (
                        not existing
                        or back_url != existing_back_url
                        or not existing.get("back_image_embedding")
                    ):
                        image = pending_download.get(back_url)
                        if image:
                            images_to_embed.append((idx, back_url, "back", image))
                        elif not back_url:
                            product["back_image_embedding"] = None

            for infer_start in range(0, len(images_to_embed), INFERENCE_BATCH_SIZE):
                infer_batch = images_to_embed[infer_start : infer_start + INFERENCE_BATCH_SIZE]
                pil_images = [img for _, _, _, img in infer_batch]
                embeddings = _embed_images_batch(pil_images)

                for (idx, url, view_type, _), embedding in zip(infer_batch, embeddings):
                    product = products[idx]
                    purl = product.get("product_url", "")
                    if embedding:
                        if view_type == "front":
                            product["image_embedding"] = embedding
                            # embedding_version = 2 is never upserted to DB (live products table has no such column)
                            stats["front_embeddings"] += 1
                        else:
                            product["back_image_embedding"] = embedding
                            stats["back_embeddings"] += 1
                        if purl not in checkpoint_data:
                            checkpoint_data[purl] = {}
                        if view_type == "front":
                            checkpoint_data[purl]["image_embedding"] = embedding
                            checkpoint_data[purl]["image_url"] = product.get("image_url", "")
                        else:
                            checkpoint_data[purl]["back_image_embedding"] = embedding
                            checkpoint_data[purl]["back_image_url"] = product.get("back_image_url", "")
                    else:
                        stats["skipped"] += 1

            downloaded_urls.update(url for _, url, _ in batch)
            pending_download.clear()
            gc.collect()

            _save_checkpoint(source, checkpoint_data)
            completed_urls.update(checkpoint_data.keys())

            _log_memory("batch %d/%d" % (batch_num, total_batches))
            logger.info(
                "  Batch %d/%d done. front=%d, back=%d (checkpoint saved, %d URLs tracked)",
                batch_num,
                total_batches,
                stats["front_embeddings"],
                stats["back_embeddings"],
                len(completed_urls),
            )

            if on_batch_done:
                batch_product_indices = set(idx for idx, _, _ in batch)
                batch_products = [products[idx] for idx in sorted(batch_product_indices)]
                try:
                    on_batch_done(batch_products, batch_num)
                except Exception as e:
                    logger.error("Batch upsert callback failed: %s", e)

        download_executor.shutdown(wait=False)

    # Text embeddings — batched for throughput
    checkpoint_data = load_checkpoint(source)
    _checkpoint_data = checkpoint_data

    logger.info("  Generating text embeddings for %d products...", len(products))

    # First pass: build info texts and identify which need new embeddings
    product_texts: list[tuple[int, str, str, dict[str, Any]]] = []
    for i, product in enumerate(products):
        product_url = product.get("product_url", "")
        existing = existing_embeddings.get(product_url, {})

        image_url = product.get("image_url", "")
        existing_image_url = existing.get("image_url", "")
        if not (image_url and (not existing or image_url != existing_image_url)):
            if existing and existing.get("image_embedding"):
                product["image_embedding"] = existing["image_embedding"]

        back_url = product.get("back_image_url")
        existing_back_url = existing.get("back_image_url")
        if not back_url and not (not existing or back_url != existing_back_url):
            if existing and existing.get("back_image_embedding"):
                product["back_image_embedding"] = existing["back_image_embedding"]

        info_text = _build_info_text(product)
        existing_info_text = ""
        if existing:
            existing_info_text = _build_info_text(existing)

        if not existing or info_text != existing_info_text:
            product_texts.append((i, product_url, info_text, existing))
        else:
            if existing and existing.get("info_embedding"):
                product["info_embedding"] = existing["info_embedding"]

    if product_texts:
        # Batch embed all texts at once
        batch_texts = [t for _, _, t, _ in product_texts]
        batch_embeddings = get_text_embeddings_batch(batch_texts, batch_size=cfg.TEXT_EMBED_BATCH_SIZE)

        for (i, product_url, _, existing), text_emb in zip(product_texts, batch_embeddings):
            product = products[i]
            if text_emb:
                product["info_embedding"] = text_emb
                stats["text_embeddings"] += 1
                if product_url not in checkpoint_data:
                    checkpoint_data[product_url] = {}
                checkpoint_data[product_url]["info_embedding"] = text_emb

        _save_checkpoint(source, checkpoint_data)
        logger.info("  Text embedding complete: %d/%d embedded", stats["text_embeddings"], len(product_texts))
    else:
        logger.info("  All text embeddings up to date, skipping")

    _save_checkpoint(source, checkpoint_data)
    _checkpoint_data.clear()

    return products, stats