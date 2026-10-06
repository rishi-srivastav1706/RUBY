from __future__ import annotations
import tempfile

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any, Dict, Optional
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os
import logging
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings

import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
import aiosqlite
import requests
import requests
import asyncio
import threading



load_dotenv()
logger = logging.getLogger(__name__)

hf_token = os.getenv("HF_API_TOKEN")

endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="text-generation",
    provider="nscale",
    huggingfacehub_api_token=hf_token
)
llm = ChatHuggingFace(llm=endpoint)

embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=hf_token,
)

# ***************************** PDF retriever store *************************
# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embedding)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass



# **************** DEDICATED ASYNC LOOP FOR BACKEND TASKS ****************
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target= _ASYNC_LOOP.run_forever, daemon = True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# ****************** TOOLS ***************
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock price for a ticker symbol using Alpha Vantage."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Set ALPHAVANTAGE_API_KEY to use the stock price tool.")
    response = requests.get(
        "https://www.alphavantage.co/query",
        params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()

MCP_SERVERS = {
    "arith": {
        "transport": "stdio",
        "command": "python3",
        "args": ["/Users/nitish/Desktop/mcp-math-server/main.py"],
    },
    "expense": {
        "transport": "streamable_http",
        "url": "https://splendid-gold-dingo.fastmcp.app/mcp",
    },
}
client = MultiServerMCPClient(MCP_SERVERS)

def load_mcp_tools() -> list[BaseTool]:
    loaded_tools: list[BaseTool] = []
    for server_name in MCP_SERVERS:
        try:
            loaded_tools.extend(
                run_async(client.get_tools(server_name=server_name))
            )
        except Exception:
            logger.exception("Failed to load MCP tools from server '%s'", server_name)
    return loaded_tools

mcp_tools = load_mcp_tools()

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools: list[BaseTool] = [search_tool, get_stock_price, *mcp_tools, rag_tool]
llm_with_tools = llm.bind_tools(tools) if tools else llm

# ******** STATES ***********
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ******************* NODES ************************

async def chat_node(state: ChatState, config = None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )


    messages =[system_message, *state["messages"]]
    response = await llm_with_tools.ainvoke(messages, config = config)
    return {"messages": [response]}

tool_node = ToolNode(tools) if tools else None

# **************** CHECKPOINTERS ************************

async def _init_checkpointer():
    conn = await aiosqlite.connect(database="Ruby.db")
    return AsyncSqliteSaver(conn)
#checkpointer
checkpointer = run_async(_init_checkpointer())

# ****************** GRAPH *******************

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")


chatbot = graph.compile(checkpointer=checkpointer)



# ********************** HELPER  ******************************
async def _alist_threads():
    all_threads = set()

    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

def retrieve_all_threads():
    return run_async(_alist_threads())

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})