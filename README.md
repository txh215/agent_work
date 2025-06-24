-----

# 协作型 Agent 系统项目报告

本项目是一个基于 **LangChain 框架**构建的协作型 Agent 系统，旨在通过对话形式协助用户完成旅行规划任务。该 Agent 能够理解用户意图、引用知识库内容（如热门景点信息）、调用外部 API（如天气查询），并能够通过记忆模块保持对话上下文，实现多轮交互。

-----

## 1\. Agent 功能说明

### 所选 Agent 功能与背景说明

我们选择构建一个**智能旅行规划助手 Agent**。

在旅行规划过程中，用户常常面临信息碎片化、决策耗时等问题，例如查询目的地天气、了解热门景点、甚至计算预算等。传统的旅行规划方式可能需要用户在多个网站和应用之间反复切换。我们的 Agent 旨在提供一个一站式、智能化的解决方案，通过对话形式协助用户完成以下任务：

  * **查询目的地天气**：获取指定城市的天气预报，帮助用户合理安排行程和准备衣物。
  * **获取景点信息**：提供热门旅游目的地的概况和推荐景点，辅助用户进行景点选择。
  * **计算旅行预算**：根据用户提供的花费明细，进行简单的费用加总或估算，协助用户管理旅行开销。
  * **保持多轮对话**：通过记忆功能，理解用户在不同轮次对话中的意图，提供更连贯和个性化的服务。

这个 Agent 的目标是简化旅行规划流程，提升用户体验，让用户能够更轻松、高效地规划和享受他们的旅程。

-----

## 2\. 系统实现细节

### 使用的技术栈简介

  * **LangChain**: 核心框架，用于构建 LLM 驱动的应用程序。它提供了 Agent、Tool、Memory、Retrievers 等模块，极大地简化了复杂 LLM 应用的开发。
  * **ChatNVIDIA LLM**: 作为本项目中的大语言模型（LLM），负责理解用户意图、生成回复以及决定何时调用哪个工具。
  * **Python**: 主要开发语言。
  * **`python-dotenv`**: 用于管理环境变量，安全地加载 API 密钥。
  * **`faiss-cpu` / `ChromaDB` (可选)**: 用于构建向量数据库，支持知识库工具的检索。
  * **`sentence-transformers` (可选)**: 用于生成文本嵌入，如果采用基于 Embedding 的检索问答系统。

### Agent 使用的 Tools 简介

本项目集成了以下三个核心工具：

#### 知识库工具：景点信息检索 (TourismInfoRetriever)

  * **功能**: 基于预先提供的文档，检索特定旅游目的地的概况和热门景点信息。
  * **实现方式**:
      * 我们维护了一个简单的文本文件（例如 `tourism_info.txt`），其中包含一些热门城市的景点介绍。
      * 利用 LangChain 的 **`TextLoader`** 加载文档，**`CharacterTextSplitter`** 将文档分割成小块。
      * 使用 **`HuggingFaceEmbeddings`** (或 DeepSeek 提供的嵌入模型) 生成文本块的嵌入向量。
      * 构建 **`FAISS`** (或 Chroma) 向量数据库，用于存储和检索这些嵌入向量。
      * 创建一个 LangChain **`RetrievalQA`** 链，允许 Agent 通过语义搜索从知识库中获取相关信息。
  * **示例应用**: 用户询问“北京有哪些好玩的景点？”或“上海的历史文化”，Agent 会从知识库中检索并提供相关信息。

#### API 工具：天气查询 (WeatherTool)

  * **功能**: 调用外部公开 API 来获取指定城市实时天气信息。
  * **实现方式**: 封装了一个名为 `get_current_weather` 的 Python 函数。虽然在示例代码中是桩函数，但实际部署时，我们会集成一个真实的天气 API（例如 OpenWeatherMap API）。该函数接受一个 `location` 参数，并返回该地点的天气预报。
  * **示例**: 用户询问“明天大阪的天气怎么样？”，Agent 会调用此工具并返回天气信息。

#### 本地函数工具：旅行预算计算 (TravelBudgetCalculator)

  * **功能**: 实现一个自定义 Python 函数，帮助用户计算旅行预算，例如对多项花费进行加总。
  * **实现方式**: 封装了一个名为 `calculate_travel_budget` 的 Python 函数。此函数可以接收一个包含数字和可选描述的字符串，解析出数字并进行求和。例如，用户可以输入“机票1500，酒店2000，餐饮500”，函数将返回总和。
  * **示例**: 用户询问“我的机票花了2000，酒店1800，当地交通300，总共多少钱？”，Agent 会调用此工具并计算总费用。

### 记忆模块

  * **功能**: 通过 `ConversationBufferMemory` 模块保持对话上下文，确保 Agent 能够记住之前的对话内容，从而实现多轮交互。
  * **实现方式**: `ConversationBufferMemory` 会将整个对话历史存储在内存中，并在每次调用 Agent 时将其作为 `chat_history` 传递给 LLM。这使得 LLM 能够理解上下文并作出更连贯的响应。例如，用户先问了北京天气，再问“那上海呢？”，Agent 能够理解“那上海呢”是关于上海天气的询问。

-----

## 3\. 运行测试方法说明

### 模型集成方式说明，需让使用者根据此步骤能成功运行测试

要运行和测试此 Agent 系统，请遵循以下步骤：

1.  **克隆项目仓库**:

    ```bash
    git clone <您的GitHub仓库URL>
    cd <您的项目文件夹>
    ```

2.  **创建并激活虚拟环境 (推荐)**:

    ```bash
    python -m venv venv
    # macOS/Linux
    source venv/bin/activate
    # Windows
    .\venv\Scripts\activate
    ```

3.  **安装依赖**:
    创建一个 `requirements.txt` 文件，内容如下：

    ```
    langchain
    langchain-nvidia-ai-endpoints # 注意这里是 langchain-nvidia-ai-endpoints
    python-dotenv
    faiss-cpu # 或者 chromadb，根据你的知识库实现选择
    sentence-transformers # 如果使用HuggingFaceEmbeddings
    ```

    然后执行安装命令：

    ```bash
    pip install -r requirements.txt
    ```

4.  **准备知识库文件**:
    在项目根目录下创建一个名为 `tourism_info.txt` 的文件，并填充一些旅游景点信息。例如：

    ```
    # tourism_info.txt

    北京：
    故宫：中国明清两代的皇家宫殿，被誉为世界五大宫之首。
    长城：世界七大奇迹之一，是中华民族的象征。
    颐和园：中国现存最大的皇家园林。

    上海：
    外滩：上海的标志性建筑群和历史文化街区。
    东方明珠：上海的标志性建筑，观光和广播电视塔。
    豫园：上海著名的江南古典园林。

    东京：
    浅草寺：东京最古老的寺庙，有标志性的雷门。
    涩谷交叉路口：世界上最繁忙的十字路口之一。
    新宿御苑：结合了日本传统、法式和英式风格的大型公园。
    ```

5.  **配置 ChatNVIDIA Key**:
    在项目根目录下创建一个名为 `.env` 的文件，并将您的 ChatNVIDIA Key 添加到其中：

    ```
    NVIDIA_API_KEY_LLAMA33="sk-YOUR_NVIDIA_API_KEY_LLAMA33_HERE"
    ```

    请将 `"sk-YOUR_NVIDIA_API_KEY_LLAMA33_HERE"` 替换为您的实际 ChatNVIDIA API Key。

6.  **运行 Agent 脚本**:
    创建一个主脚本文件（例如 `main.py`），将提供的代码粘贴进去。
    然后运行：

    ```bash
    python main.py
    ```

7.  **开始对话**:
    运行脚本后，您将在命令行看到提示 `Enter your question (or 'exit' to quit):`。此时您可以输入问题与 Agent 进行交互。

-----

## 4\. 聊天记录示例

### 聊天记录示例

以下是与 Agent 的一段聊天记录示例：

```
Enter your question (or 'exit' to quit): 北京今天天气怎么样？
> Entering new AgentExecutor chain...
Thought: The user is asking about the weather in Beijing. I should use the 'WeatherTool' to get this information.
Action: WeatherTool
Action Input: Beijing
Observation: Weather forecast for Beijing: Sunny with a high of 25°C and low of 15°C. 10% chance of precipitation.
Thought: I have successfully retrieved the weather information for Beijing. I should now respond to the user with this information.
Final Answer: 北京今天天气晴朗，最高温度25°C，最低温度15°C，有10%的降水概率。
Result: 北京今天天气晴朗，最高温度25°C，最低温度15°C，有10%的降水概率。

Enter your question (or 'exit' to quit): 北京有什么好玩的景点？
> Entering new AgentExecutor chain...
Thought: The user is asking for tourist attractions in Beijing. This information should be available in the knowledge base. I should use the 'TourismInfoRetriever' tool.
Action: TourismInfoRetriever
Action Input: 北京景点
Observation: 故宫：中国明清两代的皇家宫殿，被誉为世界五大宫之首。长城：世界七大奇迹之一，是中华民族的象征。颐和园：中国现存最大的皇家园林。
Thought: I have retrieved information about Beijing's attractions. I should now present this information to the user.
Final Answer: 北京有很多著名的景点，例如：故宫（中国明清两代的皇家宫殿，世界五大宫之首）、长城（世界七大奇迹之一，中华民族的象征）和颐和园（中国现存最大的皇家园林）。
Result: 北京有很多著名的景点，例如：故宫（中国明清两代的皇家宫殿，世界五大宫之首）、长城（世界七大奇迹之一，中华民族的象征）和颐和园（中国现存最大的皇家园林）。

Enter your question (or 'exit' to quit): 如果我机票2500，酒店3000，餐饮1200，当地交通500，总共需要多少钱？
> Entering new AgentExecutor chain...
Thought: The user is asking to sum up travel expenses. This is a calculation task, so I should use the 'TravelBudgetCalculator' tool.
Action: TravelBudgetCalculator
Action Input: 2500, 3000, 1200, 500
Observation: 您的旅行总预算为 7200.0 元。
Thought: I have calculated the total travel budget. I should now inform the user of the result.
Final Answer: 您的旅行总预算为 7200 元。
Result: 您的旅行总预算为 7200 元。

Enter your question (or 'exit' to quit): 那上海的天气和景点呢？
> Entering new AgentExecutor chain...
Thought: The user is now asking about Shanghai's weather and attractions, referencing the previous conversation. I need to use both the 'WeatherTool' and 'TourismInfoRetriever' for Shanghai.
Action: WeatherTool
Action Input: Shanghai
Observation: Weather forecast for Shanghai: Cloudy with a high of 28°C and low of 20°C. 30% chance of precipitation.
Thought: I have the weather for Shanghai. Now I need to get the attractions for Shanghai.
Action: TourismInfoRetriever
Action Input: 上海景点
Observation: 外滩：上海的标志性建筑群和历史文化街区。东方明珠：上海的标志性建筑，观光和广播电视塔。豫园：上海著名的江南古典园林。
Thought: I have both the weather and attraction information for Shanghai. I should now combine them and respond to the user.
Final Answer: 上海今天天气多云，最高温度28°C，最低温度20°C，有30%的降水概率。至于景点，上海有著名的外滩、东方明珠和豫园。
Result: 上海今天天气多云，最高温度28°C，最低温度20°C，有30%的降水概率。至于景点，上海有著名的外滩、东方明珠和豫园。

Enter your question (or 'exit' to quit): exit
```

### 分析此 Agent 如何能帮助用户

此 Agent 通过以下方式帮助用户：

  * **提升效率**: 用户无需打开多个应用或网页，在一个对话界面即可完成天气查询、景点信息检索和预算计算等多项任务。
  * **语境理解与多轮交互**: 借助记忆模块，Agent 能够理解多轮对话中的上下文，例如用户在询问完北京后，直接问“那上海呢？”，Agent 能够推断出用户依然在询问天气和景点，从而提供更自然、连贯的交流体验。
  * **任务自动化**: Agent 能够根据用户意图自动选择并调用合适的工具（如天气 API 或计算器函数），将复杂的查询或计算过程自动化，用户只需表达需求即可获得结果。
  * **信息整合**: Agent 能将来自不同工具的信息进行整合，例如同时回答上海的天气和景点，提供更全面的回复。
  * **降低学习成本**: 用户无需学习如何使用各种复杂的应用，只需通过日常对话即可完成任务，极大降低了使用门槛。

-----

## 5\. 合作与反思

### 成员1：谭秀辉

  * **负责内容**:
      * 负责 Agent 功能说明和系统实现细节的撰写。
      * 设计并实现了 `TravelBudgetCalculator` 本地函数工具的逻辑。
      * 负责 LangChain Agent 的初始化配置，包括 LLM 和 Memory 的集成。
      * 完成基本的运行测试脚本。
      * 编写了部分聊天记录示例和对应分析。
  * **学到的内容**:
      * 深入理解 LangChain 中 Agent、Tool 和 Memory 模块的工作原理及其相互协作方式。
      * 学习了如何根据用户意图设计合适的自定义工具，并将其集成到 LangChain Agent 中。
      * 掌握了使用 ChatNVIDIA LLM 作为 Agent 驱动模型的集成方法。
      * 了解了 `.env` 文件进行敏感信息管理的重要性。
  * **遇到的困难**:
      * 在最初设计 `Calculator` 工具时，如何使其能够灵活地解析用户输入的自然语言表达式，并准确提取数字和操作符是一个挑战。通过细化解析逻辑，并考虑不同操作的关键词，逐步完善了该工具。
      * 理解 Agent 的 `Thought/Action/Observation` 链式思考过程，并根据 `verbose=True` 的输出调试 Agent 的决策行为，花费了一些时间。

### 成员2：王旌旗

  * **负责内容**:
      * 负责报告的整体结构和排版，确保符合 Markdown 格式要求。
      * 设计并实现了 `TourismInfoRetriever` 知识库工具，包括文档加载、文本分割、嵌入生成和向量数据库检索（例如使用 FAISS）。
      * 负责 `WeatherTool` 的 API 接口设计（尽管是桩函数，但考虑了真实 API 的数据结构）。
      * 负责运行测试方法的详细说明，确保其他使用者能够顺利运行。
      * 补充了聊天记录示例，并对 Agent 如何帮助用户进行了详细分析。
  * **学到的内容**:
      * 掌握了基于 LangChain 构建 RAG（检索增强生成）系统的基本流程，包括文档处理、嵌入模型选择和向量数据库的使用。
      * 学会了如何将外部 API 包装成 LangChain 的 Tool，并让 Agent 能够智能地调用。
      * 加深了对 LangChain `initialize_agent` 函数中不同 `AgentType` 的理解，并根据任务需求进行选择。
      * 熟悉了 GitHub 提交和协作流程，理解了代码提交均衡性的重要性。
  * **遇到的困难**:
      * 在构建知识库工具时，选择合适的文本分割策略和嵌入模型，以及调试检索结果的相关性是关键。特别是当知识库内容复杂时，如何优化检索效率和准确性需要更多尝试。
      * 初期在 `Tools` 的 `description` 编写上，如何清晰地描述工具的功能，以便 LLM 能够准确理解并调用，是一个需要反复斟酌的细节。
