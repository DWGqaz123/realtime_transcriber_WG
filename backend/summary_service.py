# backend/summary_service.py

"""
智能摘要服务
使用 OpenAI API 生成 Markdown 格式的知识点摘要
"""

import os
import json
from typing import Optional
import httpx
from datetime import datetime
import httpx
from config import SummaryConfig
class SummaryService:
    """摘要生成服务"""
    
    def __init__(self, api_key: Optional[str] = None):
        
        # 🔧 使用配置类
        self.api_key = api_key or SummaryConfig.get_api_key()
        self.api_url = SummaryConfig.API_URL or "https://api.openai.com/v1/chat/completions"
        self.model = SummaryConfig.MODEL
        self.temperature = SummaryConfig.TEMPERATURE
        self.max_tokens = SummaryConfig.MAX_TOKENS
        if not self.api_key:
            print("⚠️ OPENAI_API_KEY not set, summary generation will fail")
        
    
    async def generate_summary(
        self,
        buffer_text: str,
        context: str = "",
        mode: str = "lecture"
    ) -> Optional[str]:
        """
        生成摘要
        
        Args:
            buffer_text: 待摘要的文本内容
            context: 上一段的末尾文本（提供上下文）
            mode: 录音模式 (lecture/discussion)
        
        Returns:
            Markdown 格式的摘要，失败返回 None
        """
        if not self.api_key:
            print("❌ Cannot generate summary: API key not configured")
            return None
        
        try:
            # 构建 prompt
            prompt = self._build_prompt(buffer_text, context, mode)
            
            print(f"\n{'='*60}")
            print(f"🤖 Generating summary with OpenAI API")
            print(f"   Model: {self.model}")
            print(f"   Input length: {len(buffer_text)} chars")
            print(f"{'='*60}\n")
            
            # 调用 API
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "你是一个专业的笔记助手，擅长从讲座和讨论中提取关键知识点。"
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "max_tokens": 1024,
                        "temperature": 0.3  # 较低的温度确保输出稳定
                    }
                )
            
            # 检查响应
            if response.status_code != 200:
                print(f"❌ OpenAI API error: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
            
            # 解析响应
            result = response.json()

            # OpenAI 响应格式
            choices = result.get("choices", [])

            if not choices:
                print("❌ Empty response from OpenAI API")
                return None

            # 提取文本
            message = choices[0].get("message", {})
            summary = message.get("content", "")
            
            print(f"✅ Summary generated successfully")
            print(f"   Output length: {len(summary)} chars")
            print(f"   Preview: {summary[:100]}...\n")
            
            return summary
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _build_prompt(self, buffer_text: str, context: str, mode: str) -> str:
        """构建摘要 prompt
            Args:
                buffer_text: 待摘要的文本（当前快照）
                context: 上一段的末尾句子（上下文）
                mode: 模式（lecture/discussion）
            Returns:
                str: 完整的 Prompt
        """
        # 模式特定的指令
        if mode == "lecture":
            focus = "提取主要概念、重要结论、关键数据"
        else:  # discussion
            focus = "总结核心观点、不同立场、达成的共识"
        
        if context:
        # 有上下文
            prompt = f"""请基于以下内容生成简洁的知识点摘要。

        【上下文】（仅供理解连贯性，不要对此部分生成摘要）：
        {context}

        【当前内容】（请对此部分生成摘要）：
        {buffer_text}

        要求：
        1. 根据【上下文】理解背景和连贯性
        2. 只对【当前内容】生成 3-5 个关键知识点
        3. 不要重复【上下文】中已提到的概念
        4. {focus}
        5. 使用 Markdown 格式，每个知识点用 `- ` 开头
        6. 每个知识点 1-2 句话，简洁明了
        7. 突出核心概念和关键信息
        8. 不要添加标题，直接列出知识点"""
        else:
            # 无上下文（第一次摘要）
            prompt = f"""请基于以下内容生成简洁的知识点摘要。
                    【内容】：
                    {buffer_text}

                    要求：
                    1. 提取 3-5 个关键知识点
                    2. {focus}
                    3. 使用 Markdown 格式，每个知识点用 `- ` 开头
                    4. 每个知识点 1-2 句话，简洁明了
                    5. 突出核心概念和关键信息
                    6. 不要添加标题，直接列出知识点"""
        
        return prompt


# 全局实例（可选）
_summary_service: Optional[SummaryService] = None


def get_summary_service() -> SummaryService:
    """获取全局摘要服务实例"""
    global _summary_service
    
    if _summary_service is None:
        _summary_service = SummaryService()
    
    return _summary_service