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

