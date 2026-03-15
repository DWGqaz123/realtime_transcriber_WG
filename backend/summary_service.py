# backend/summary_service.py

"""
智能摘要服务
使用 OpenAI API 生成 Markdown 格式的知识点摘要
"""
from typing import Optional
import logging
import asyncio
import time
import httpx
from config import SummaryConfig

log = logging.getLogger("transcriber.summary")

class SummaryService:
    """摘要生成服务"""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or SummaryConfig.get_api_key()
        self.api_url = SummaryConfig.API_URL or "https://api.openai.com/v1/chat/completions"
        self.model = SummaryConfig.MODEL
        self.temperature = SummaryConfig.TEMPERATURE
        self.max_tokens = SummaryConfig.MAX_TOKENS
        self._client = httpx.AsyncClient(timeout=30.0)

    
    async def generate_summary(
        self,
        buffer_text: str,
        context: str = "",
    ) -> Optional[str]:
        """
        生成摘要
        
        Args:
            buffer_text: 待摘要的文本内容
            context: 上一段的末尾文本（提供上下文）
        
        Returns:
            Markdown 格式的摘要，失败返回 None
        """
        if not self._api_key:
            return None
        
        try:
            # 构建 prompt
            prompt = self._build_prompt(buffer_text, context)
            
            
            # 调用 API（带重试）
            max_retries = 3
            response = None
            for attempt in range(max_retries):
                try:
                    t0 = time.perf_counter()
                    response = await self._client.post(
                        self.api_url,
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a professional note-taking assistant, skilled at extracting key knowledge points from lectures."
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            "max_tokens": self.max_tokens,
                            "temperature": self.temperature
                        }
                    )
                    log.info("API latency: %.2fs (attempt %d)", time.perf_counter() - t0, attempt + 1)
                    break
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        log.warning("OpenAI API attempt %d failed: %s, retrying in %ds", attempt + 1, e, wait_time)
                        await asyncio.sleep(wait_time)
                    else:
                        log.error("OpenAI API failed after %d attempts: %s", max_retries, e)
                        return None
            
            # 检查响应
            if response is None or response.status_code != 200:
                if response is not None:
                    log.warning("OpenAI API returned status %d", response.status_code)
                return None
            
            # 解析响应
            result = response.json()

            # OpenAI 响应格式
            choices = result.get("choices", [])

            if not choices:
                return None

            # 提取文本
            message = choices[0].get("message", {})
            summary = message.get("content", "")
            
            
            return summary
            
        except Exception as e:
            log.error("Summary generation failed: %s", e, exc_info=True)
            return None
    
    def _build_prompt(self, buffer_text: str, context: str) -> str:
        focus = "Extract the main concepts, important conclusions and key data"

        if context:
            return (
                f"Please generate a concise summary of knowledge points based on the following content.\n\n"
                f"[Context] (For understanding coherence only. Do not summarize this part):\n{context}\n\n"
                f"[Current Content] (Generate a summary of this part):\n{buffer_text}\n\n"
                f"Requirements:\n"
                f"1. Use [Context] to understand background and coherence\n"
                f"2. Only generate 3-5 key knowledge points for [Current Content]\n"
                f"3. Do not repeat concepts already mentioned in [Context]\n"
                f"4. {focus}\n"
                f"5. Use Markdown format, each knowledge point starts with `- `\n"
                f"6. Each knowledge point is 1-2 sentences, concise and clear\n"
                f"7. Do not add a title, directly list the knowledge points"
            )
        else:
            return (
                f"Please generate a concise summary of knowledge points based on the following content.\n\n"
                f"[Content]:\n{buffer_text}\n\n"
                f"Requirements:\n"
                f"1. Extract 3-5 key knowledge points\n"
                f"2. {focus}\n"
                f"3. Use Markdown format, each knowledge point starts with `- `\n"
                f"4. Each knowledge point is 1-2 sentences, concise and clear\n"
                f"5. Do not add a title, directly list the knowledge points"
            )
