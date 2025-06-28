import math
from typing import Optional
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import os
from datetime import datetime

# For Knowledge Base Tool
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings # Or use DeepSeekEmbeddings if available
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
load_dotenv()

# --- 1. Initialize NVIDIA LLM ---
# 使用 ChatNVIDIA 初始化 LLM，并指定模型和 API 密钥
llm = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    api_key=os.getenv("NVIDIA_API_KEY_LLAMA33"),
    temperature=0.6,
    top_p=0.7,
    max_tokens=4096,
)

# --- 2. Define Tools ---

# API Tool: Weather
def get_current_weather(location: str) -> str:
    """
    获取指定地点的当前天气预报。
    这是一个模拟实现。在实际应用中，它会调用一个真实的天气 API。
    """
    location_lower = location.lower()
    current_date = datetime.now().strftime("%Y年%m月%d日")

    if "tokyo" in location_lower or "东京" in location_lower:
        return f"{current_date} {location} 的天气预报：多云，最高气温 28°C，最低气温 20°C。30% 的降水概率。"
    elif "beijing" in location_lower or "北京" in location_lower:
        return f"{current_date} {location} 的天气预报：晴朗，最高气温 25°C，最低气温 15°C。10% 的降水概率。"
    elif "shanghai" in location_lower or "上海" in location_lower:
        return f"{current_date} {location} 的天气预报：多云，最高气温 28°C，最低气温 20°C。30% 的降水概率。"
    elif "osaka" in location_lower or "大阪" in location_lower:
        return f"{current_date} {location} 的天气预报：有雨，最高气温 26°C，最低气温 19°C。60% 的降水概率。"
    else:
        return f"抱歉，目前没有 {location} 的天气数据。请尝试查询北京、上海、东京或大阪。"
