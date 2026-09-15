"""Kith scraper configuration."""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    BRAND_NAME: str = "Kith"
    SOURCE: str = "scraper-kith"
    BRAND_COLUMN: str = "Kith"
    SECOND_HAND: bool = False
    LANDING_PAGE: str = "https://kith.com/"
    BASE_URL: str = "https://kith.com"
    CURRENCY: str = "USD"
    PRODUCTS_JSON_LIMIT: int = 250

    CATEGORY_URLS: list[str] = field(default_factory=lambda: [
        "https://kith.com/collections/kith",
        "https://kith.com/collections/jackets",
        "https://kith.com/collections/footwear",
        "https://kith.com/collections/crewnecks",
        "https://kith.com/collections/bags",
        "https://kith.com/collections/kith-shirts",
        "https://kith.com/collections/kith-shorts",
        "https://kith.com/collections/kith-graphic-tees",
        "https://kith.com/collections/kith-classic-tees",
        "https://kith.com/collections/cargo-pants",
        "https://kith.com/collections/eyewear"
    ])

    CATEGORY_DISPLAY: dict[str, str] = field(default_factory=lambda: {
        "kith": "Kith",
        "jackets": "Jackets",
        "footwear": "Footwear",
        "crewnecks": "Crewnecks",
        "bags": "Bags",
        "kith-shirts": "Shirts",
        "kith-shorts": "Shorts",
        "kith-graphic-tees": "Graphic Tees",
        "kith-classic-tees": "Classic Tees",
        "cargo-pants": "Cargo Pants",
        "eyewear": "Eyewear"
    })

    SUPABASE_URL: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    SUPABASE_KEY: str = field(default_factory=lambda: os.getenv("SUPABASE_KEY", ""))

    EMBEDDING_MODEL: str = "google/siglip-base-patch16-384"
    EMBEDDING_DIM: int = 768
    EMBEDDING_VERSION: int = 2
    RATE_LIMIT_DELAY: float = 0.2
    BATCH_SIZE: int = 50
    STALE_MISS_THRESHOLD: int = 2
    REQUEST_TIMEOUT: int = 30
    SCRAPE_WORKERS: int = 8
    DOWNLOAD_WORKERS: int = 30
    TEXT_EMBED_BATCH_SIZE: int = 32
    USER_AGENT: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
    GENDER_DEFAULT: str = "Unisex"


cfg = Config()
