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

# Local Function Tool: Travel Budget Calculator
def calculate_travel_budget(expenses: str) -> str:
    """
    根据逗号分隔的费用列表计算旅行总预算。
    示例输入："机票2500, 酒店3000, 餐饮1200, 交通500"
    """
    try:
        parts = expenses.split(',')
        total_cost = 0.0
        details = []
        for part in parts:
            part = part.strip()
            # 从字符串中提取数字，采用简单的方法
            # 对于复杂情况可能需要更健壮的解析
            numbers = [float(s) for s in part.split() if s.replace('.', '', 1).isdigit()]
            if numbers:
                cost = numbers[0]
                total_cost += cost
                # 尝试提取费用名称，如果分割后有内容
                item_name = part.split()[0] if part.split() and not part.split()[0].replace('.', '', 1).isdigit() else '项目'
                details.append(f"{item_name}: {cost:.2f}元") # 格式化为两位小数
            else:
                details.append(f"无法解析 '{part}'")
        
        if not details:
            return "错误：没有找到有效的费用项目进行计算。"

        return f"您的旅行总预算为 {total_cost:.2f} 元。详细明细：{'; '.join(details)}。"
    except Exception as e:
        return f"计算预算时出错：{str(e)}。请提供逗号分隔的费用格式，例如 '机票2500, 酒店3000'。"

# Knowledge Base Tool: Tourism Info Retriever
def setup_tourism_knowledge_base():
    """设置用于旅游信息的 FAISS 向量存储。"""
    try:
        # 确保您的 'tourism_info.txt' 文件存在且编码为 UTF-8
        loader = TextLoader("tourism_info.txt", encoding="utf-8")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        
        # 使用 HuggingFaceEmbeddings 作为演示的常用选择。
        # 您可能需要安装 'sentence-transformers' 包。
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        db = FAISS.from_documents(docs, embeddings)
        return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    except FileNotFoundError:
        print("错误：未找到 'tourism_info.txt' 文件。请按照 README 中的说明创建它。")
        return None
    except Exception as e:
        print(f"设置旅游知识库时出错：{e}")
        return None

# 初始化知识库
qa_chain = setup_tourism_knowledge_base()
if qa_chain is None:
    print("旅游知识库初始化失败。TourismInfoRetriever 工具将不可用。")


tools = [
    Tool(
        name="WeatherTool",
        func=get_current_weather,
        description="获取指定地点的实时天气预报。输入应为城市名称，例如'北京'、'东京'、'上海'、'大阪'。",
    ),
    Tool(
        name="TravelBudgetCalculator",
        func=calculate_travel_budget,
        description="计算旅行总预算。输入应为逗号分隔的费用列表，例如'机票2500, 酒店3000, 餐饮1200, 交通500'。",
    ),
]

# 如果成功初始化，则添加 TourismInfoRetriever 工具
if qa_chain:
    tools.append(
        Tool(
            name="TourismInfoRetriever",
            func=qa_chain.run, # qa_chain.run 期望字符串输入并返回字符串输出
            description="查询热门旅游目的地的景点信息和概况。输入应为地点名称或相关关键词，例如'北京景点'、'上海旅游'、'东京有什么好玩的'。",
        )
    )

# --- 3. Initialize Memory ---
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True # 返回消息对象列表
)

# --- 4. Initialize Agent ---
# 初始化 Agent 执行器
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory, # 传入记忆模块
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True, # 设置为 True 可以看到 Agent 的思考过程
    handle_parsing_errors=True # 帮助处理 LLM 可能输出的格式错误的动作
)

# --- 5. Run the Agent ---
if __name__ == "__main__":
    print("你好！我是你的旅行规划助手。有什么可以帮助你的吗？")
    while True:
        query = input("\n请输入你的问题（或输入'exit'退出）：")
        if query.lower() == 'exit':
            print("再见！祝你旅途愉快！")
            break
        try:
            # invoke 方法的输入通常是一个字典，键为 "input"
            result = agent_executor.invoke({"input": query})
            # 确保打印的是 output 字段
            print(f"助手：{result['output']}")
        except Exception as e:
            print(f"发生错误：{e}")
            print("请尝试重新措辞你的问题，或者输入'exit'退出。")
