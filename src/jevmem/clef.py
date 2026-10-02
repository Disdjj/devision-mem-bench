"""Cloudflare Workers AI 上的 Clef / Clef-flash 客户端。

Clef 与 Jev 的 System One API 兼容（同样的 state / questions / answers），
但走 Workers AI 的 REST 路径，且响应包在 {"result": ...} 里。
这里提供与 typesafe_sdk 客户端相同的 `system_one` 接口，返回同一个 SystemOneResponse，
从而直接复用 JevDecider 的全部判断逻辑。
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess

import httpx
from typesafe_sdk import SystemOneResponse

from . import config

# Clef 单次请求最多 64 道题（Jev 没有这个限制）
CLEF_MAX_QUESTIONS = 64
RETRY_STATUSES = {429, 500, 502, 503, 504, 529}


def _wrangler_token() -> str:
    """没有配置 CLOUDFLARE_API_TOKEN 时，借用 wrangler 的 OAuth token（会自动刷新）。"""
    out = subprocess.run(["wrangler", "auth", "token"], capture_output=True, text=True, check=True).stdout
    return out.strip().splitlines()[-1].strip()


class CloudflareSystemOne:
    def __init__(self, model: str = "clef", account_id: str | None = None, token: str | None = None, retries: int = 4):
        self.model = model
        self.account_id = account_id or config.cloudflare_account_id()
        self._fixed_token = token or os.getenv("CLOUDFLARE_API_TOKEN")
        self.token = self._fixed_token or _wrangler_token()
        self.retries = retries
        self.url = (
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/@cf/cloudflare/{model}"
        )
        self.http = httpx.AsyncClient(timeout=60)

    async def system_one(self, state, questions) -> SystemOneResponse:
        body = {
            "model": self.model,
            "state": state,
            "questions": {k: q.model_dump(mode="json", exclude_none=True, by_alias=True) for k, q in questions.items()},
        }
        refreshed = False
        for attempt in range(self.retries + 1):
            resp = await self.http.post(self.url, json=body, headers={"Authorization": f"Bearer {self.token}"})
            if resp.status_code == 401 and not refreshed and not self._fixed_token:
                # OAuth token 过期，刷新一次后重试
                self.token, refreshed = _wrangler_token(), True
                continue
            if resp.status_code in RETRY_STATUSES and attempt < self.retries:
                await asyncio.sleep(min(8.0, 0.5 * 2**attempt))
                continue
            resp.raise_for_status()
            payload = resp.json()
            if not payload.get("success", True):
                raise RuntimeError(f"Workers AI 返回错误: {payload.get('errors')}")
            # Score 的 legend/probabilities 键是字符串，JSON 模式校验才会转成 SDK 期望的 int
            return SystemOneResponse.model_validate_json(json.dumps(payload["result"]))
        raise RuntimeError("Workers AI 重试次数用尽")

    async def aclose(self) -> None:
        await self.http.aclose()
