# backend/summary_service.py

"""
智能摘要服务
使用 OpenAI API 生成 Markdown 格式的知识点摘要
"""
from typing import Optional
import logging
import asyncio
import httpx
from config import SummaryConfig

log = logging.getLogger("transcriber.summary")

class SummaryService:
    """摘要生成服务"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化摘要服务
        
        Args:
            api_key: OpenAI API key (可选，会从配置或环境变量读取)
        """
        
        self._api_key = api_key or SummaryConfig.get_api_key()
        self.api_url = SummaryConfig.API_URL or "https://api.openai.com/v1/chat/completions"
        self.model = SummaryConfig.MODEL
        self.temperature = SummaryConfig.TEMPERATURE
        self.max_tokens = SummaryConfig.MAX_TOKENS
        
    
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
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.post(
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
                    break  # 成功则跳出重试循环
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
        """构建摘要 prompt
            Args:
                buffer_text: 待摘要的文本（当前快照）
                context: 上一段的末尾句子（上下文）
            Returns:
                str: 完整的 Prompt
        """
        
        focus = "Extract the main concepts, important conclusions and key data"
        
        if context:
        # 有上下文
            prompt = f"""Please generate a concise summary of knowledge points based on the following content.

[Context](For understanding coherence only. Do not generate a summary of this part):
        {context}

        【Current Content】（Please generate a summary of this part）：
        {buffer_text}

        Requirements:
        1. Based on the [Context] understand background and coherence
        2. Only generate 3-5 key knowledge points for the [Current Content]
        3. Do not repeat concepts already mentioned in [Context]
        4. {focus}
        5. Use Markdown format, each knowledge point starts with `- `
        6. Each knowledge point is 1-2 sentences, concise and clear
        7. Highlight core concepts and key information
        8. Do not add a title, directly list the knowledge points"""
        else:
            # 无上下文（第一次摘要）
            prompt = f"""Please generate a concise summary of knowledge points based on the following content.
                    【Content】：
                    {buffer_text}

                    Requirements:
                    1. Extract 3-5 key knowledge points
                    2. {focus}
                    3. Use Markdown format, each knowledge point starts with `- `
                    4. Each knowledge point is 1-2 sentences, concise and clear
                    5. Highlight core concepts and key information
                    6. Do not add a title, directly list the knowledge points"""

        return prompt
