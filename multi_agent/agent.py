import asyncio
import os
from typing import Dict, List

import serpapi
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool

from services import model_service

# 选择要使用的模型（可以修改这个变量来切换模型）
SELECTED_MODEL = "qwen3:30b"  # 这里可以改成任意可用的模型


class SerpAPISearch:
    """SerpAPI搜索工具类 - 使用官方客户端"""

    def __init__(self, api_key: str = None):
        """
        初始化SerpAPI搜索工具

        Args:
            api_key: SerpAPI Key (从serpapi.com获取)
        """
        self.api_key = api_key
        self.client = serpapi.Client(api_key=api_key) if api_key else None

    def search_web(self, query: str, num_results: int = 5) -> Dict:
        """
        执行网页搜索

        Args:
            query: 搜索关键词
            num_results: 返回结果数量，默认5条

        Returns:
            Dict: 搜索结果
        """
        try:
            # 如果没有提供API密钥，返回演示数据
            if not self.api_key or not self.client:
                return self._get_demo_results(query, num_results)

            # 使用官方客户端进行搜索
            search_params = {
                'engine': 'google',
                'q': query,
                'num': num_results
            }

            # 发送搜索请求
            search_result = self.client.search(search_params)

            # 解析搜索结果
            results = self._parse_api_results(search_result, query)

            return {
                "status": "success",
                "query": query,
                "results": results,
                "total_results": len(results),
                "source": "serpapi"
            }

        except Exception as e:
            return {
                "status": "error",
                "query": query,
                "error": f"搜索失败: {str(e)}",
                "results": []
            }

    def _parse_api_results(self, api_response: Dict, query: str) -> List[Dict]:
        """
        解析SerpAPI返回的搜索结果

        Args:
            api_response: API响应数据
            query: 搜索关键词

        Returns:
            List[Dict]: 解析后的结果列表
        """
        results = []

        # 检查API响应结构
        if "organic_results" in api_response:
            # 解析有机搜索结果
            for item in api_response["organic_results"]:
                result = {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "Google搜索",
                    "position": item.get("position", 0)
                }
                results.append(result)
        elif "answer_box" in api_response:
            # 解析答案框结果
            answer_box = api_response["answer_box"]
            results.append({
                "title": answer_box.get("title", "答案"),
                "url": answer_box.get("link", ""),
                "snippet": answer_box.get("answer", answer_box.get("snippet", "")),
                "source": "Google答案框",
                "position": 0
            })
        else:
            # 如果没有找到标准格式，返回演示数据
            return self._get_demo_results(query, 5)["results"]

        return results

    def _get_demo_results(self, query: str, num_results: int) -> Dict:
        """
        获取演示搜索结果（当没有配置API密钥时使用）

        Args:
            query: 搜索关键词
            num_results: 结果数量

        Returns:
            Dict: 演示搜索结果
        """
        results = []

        # 基于查询内容生成更相关的演示数据
        for i in range(min(num_results, 5)):
            results.append({
                "title": f"关于 '{query}' 的搜索结果 {i+1}",
                "url": f"https://www.google.com/search?q={query.replace(' ', '+')}",
                "snippet": f"这是关于 '{query}' 的详细搜索结果摘要 {i+1}。在实际使用中，这里会显示通过SerpAPI获取的真实搜索结果内容。",
                "source": "Google搜索",
                "position": i + 1
            })

        return {
            "status": "success",
            "query": query,
            "results": results,
            "total_results": len(results),
            "source": "demo_mode",
            "note": "当前使用演示模式，请配置SerpAPI密钥以获取真实搜索结果"
        }

# 创建SerpAPI搜索工具函数，供Agent使用
def serpapi_search(query: str, num_results: int = 5) -> Dict:
    """
    SerpAPI搜索工具函数

    Args:
        query: 搜索关键词
        num_results: 返回结果数量

    Returns:
        Dict: 搜索结果
        {
        "title": ...,
         "url": ...,
         "snippet": ...,
         "source": "Google搜索",
         "position": ...
        }

    """
    # 从环境变量获取API密钥
    api_key = os.getenv("SERPAPI_KEY")

    search_tool = SerpAPISearch(api_key=api_key)

    return search_tool.search_web(query, num_results)

def create_research_agent(model_name: str):
    """创建使用指定模型的agent - 使用模型服务类"""
    # 使用模型服务类创建模型实例
    model = model_service.create_model(model_name)

    return LlmAgent(
        name="researcher_agent",
        model=model,
        description="""
        你是一名专业研究专员。你的唯一工作是使用SerpSearch（serpapi_search）工具，针对给定主题查找 2-3 条相关信息，并附上引用来源呈现研究结果。
        """,
        tools=[serpapi_search],
        output_key="research_findings"
    )

def create_summarizer_agent(model_name: str):
    """创建使用指定模型的agent - 使用模型服务类"""
    # 使用模型服务类创建模型实例
    model = model_service.create_model(model_name)

    return LlmAgent(
        name="summarizer_agent",
        model=model,
        description="""
        根据搜索结果：{research_findings}，并以要点列表形式简明总结3-5个关键点。
        """,
        output_key="final_summary"
    )

researcher_agent = create_research_agent(SELECTED_MODEL)
summarizer_agent = create_summarizer_agent(SELECTED_MODEL)

def create_greeter_agent(model_name: str):
    """编排Agent"""
    # 使用模型服务类创建模型实例
    model = model_service.create_model(model_name)

    return LlmAgent(
        name="ResearchCoordinator",
        model=model,
        description="""
        您是一名研究协调员。您的目标是通过组织工作流程来解答用户的查询。
        1. 首先，您必须调用 `researcher_agent` 工具来查找与用户提供的主题相关的有用信息。
        2. 接下来，在收到研究结果后，您必须调用 `summarizer_agent` 工具来创建简洁的摘要。
        3. 最后，将最终摘要清晰地呈现给用户作为您的回复。
        """,
        sub_agents=[researcher_agent, summarizer_agent]
    )

root_agent = create_greeter_agent(SELECTED_MODEL)

async def run_debug(question: str):
    """运行调试会话"""
    print("\n🚀 开始调试会话...")
    runner = InMemoryRunner(agent=root_agent)
    result = await runner.run_debug(question)
    print(result)

if __name__ == "__main__":
    asyncio.run(run_debug("Run the daily executive briefing on Tech, Health, and Finance"))




