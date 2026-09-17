from corecoder import Agent, LLM
from corecoder.mcp import load_mcp_tools
import os

def test_mcp_agent():
    llm = LLM(
        model="qwen3.7-plus",
        api_key=os.getenv("ALI_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 从 ~/.corecoder/mcp.json 加载 MCP 工具
    mcp_tools = load_mcp_tools()

    print("\n加载到的 MCP 工具：")
    for tool in mcp_tools:
        print("-", tool.name)

    assert mcp_tools, "没有加载到 MCP 工具"

    agent = Agent(
        llm=llm,
        tools=mcp_tools,
    )

    result = agent.chat(
        "请使用 MCP 工具查看 E:\\CoreCoder 目录下有哪些文件"
    )

    print("\nAgent 最终回答：")
    print(result)