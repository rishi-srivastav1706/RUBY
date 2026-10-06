from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os
import logging
from langchain_huggingface import HuggingFaceEndpoint
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
    repo_id="Qwen/Qwen3.8-27B:deepinfra",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)
llm = ChatHuggingFace(llm=endpoint)


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

tools: list[BaseTool] = [search_tool, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm

# ******** STATES ***********
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ******************* NODES ************************

async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
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
