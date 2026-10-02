"""环境变量与模型配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_FLASH = "deepseek-flash"
DEEPSEEK_PRO = "deepseek-v4-pro"
JEV_MODEL = "jev-latest"

# 每百万 token 的美元价格，用于 benchmark 估算成本。
# Jev 只按输入计费；DeepSeek Flash 取高峰期、未命中缓存的价格（保守估计），可用环境变量覆盖。
JEV_PRICE_IN = 0.042
# Workers AI 上的 Clef 系列，同样只按输入计费
CLEF_PRICE_IN = 0.24
CLEF_FLASH_PRICE_IN = 0.09
DEEPSEEK_FLASH_PRICE_IN = float(os.getenv("DEEPSEEK_FLASH_PRICE_IN", "0.30"))
DEEPSEEK_FLASH_PRICE_OUT = float(os.getenv("DEEPSEEK_FLASH_PRICE_OUT", "1.20"))


@dataclass(frozen=True)
class Keys:
    jev: str
    deepseek: str


def _env(*names: str) -> str:
    for name in names:
        if value := os.getenv(name):
            return value.strip()
    raise RuntimeError(f"缺少环境变量: {' / '.join(names)}（请在 .env 中配置）")


def keys() -> Keys:
    return Keys(
        jev=_env("jev_api_key", "JEV_API_KEY", "TYPESAFE_API_KEY"),
        deepseek=_env("deepseek_api_key", "DEEPSEEK_API_KEY"),
    )


def cloudflare_account_id() -> str:
    return _env("CLOUDFLARE_ACCOUNT_ID")


def db_path() -> Path:
    return Path(os.getenv("JEVMEM_DB", ROOT / "data" / "memory.db"))
