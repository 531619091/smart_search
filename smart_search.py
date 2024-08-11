import uuid
import json
import logging
from typing import Dict, List, Optional, Generator
from collections import defaultdict
from openai import OpenAI
from duckduckgo_search import DDGS
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts import PLANNER_PROMPT_CN, PLANNER_PROMPT_EN, FINAL_RESPONSE_CN, FINAL_RESPONSE_EN, \
SEARCH_AGENT_PROMPT_EN, SEARCH_AGENT_PROMPT_CN,REFLECTION_AGENT_PROMPT_EN, REFLECTION_AGENT_PROMPT_CN

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_json(text):
    # 使用正则表达式找到 JSON 内容
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            # 尝试解析 JSON
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            return None
    else:
        print("未找到 JSON 内容")
        return None

class SearchAgent:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.ddgs = DDGS()
        self.logger = logging.getLogger(__name__ + ".SearchAgent")

    def process(self, messages: List[Dict[str, str]]) -> str:
        self.logger.info("SearchAgent 开始处理消息")
        
        current_date = datetime.now().strftime("%Y年%m月%d日")
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_query",
                    "description": "使用给定的查询进行网络搜索，返回相关的搜索结果。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "要搜索的查询字符串"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_url_detail",
                    "description": "获取给定URL的网页详情",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "要获取网页详情的URL"
                            }
                        },
                        "required": ["url"]
                    }
                }
            }
        ]

        system_message = {
            "role": "system", 
            "content": SEARCH_AGENT_PROMPT_CN.format(current_date=current_date)
        }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[system_message] + messages,
            tools=tools
        )

        response_message = response.choices[0].message

        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "search_query":
                    self.logger.info(f"执行搜索查询: {function_args.get('query')}")
                    results = self._search_query(function_args.get("query"), function_args.get("num_results", 10))
                    explanation = ("我是searcher，以下是我经过搜索得到的结果，包括标题、简介和网页地址。"
                                   "每个结果都有一个索引编号，可用于引用特定的搜索结果：\n\n")
                    formatted_results = json.dumps(results, ensure_ascii=False, indent=2)
                    return explanation + formatted_results

                elif function_name == "get_url_detail":
                    url = function_args["url"]
                    self.logger.info(f"获取网页详情: {url}")
                    detail = self._get_url_detail(url)
                    if detail:
                        explanation = f"我是searcher，以下是从网址 {url} 获取的详细内容：\n\n"
                        return explanation + detail
                    else:
                        return f"无法获取网页 {url} 的详细内容。"
        
        self.logger.info("SearchAgent 无法识别有效的搜索请求")
        return "我是搜索助手，需要使用搜索函数来查找信息。请提供明确的搜索查询。"

    def _search_query(self, query: str, num_results: int = 10) -> List[Dict]:
        try:
            results = list(self.ddgs.text(query, max_results=num_results))
            formatted_results = [
                {
                    "title": result.get("title", "No title"),
                    "body": result.get("body", "No description"),
                    "href": result.get("href", "No URL"),
                    "index": i + 1
                }
                for i, result in enumerate(results)
            ]
            return formatted_results
        except Exception as e:
            self.logger.error(f"搜索失败: {query}, 错误: {str(e)}")
            return []

    def _get_url_detail(self, url: str) -> Optional[str]:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return re.sub(r'\n+', '\n', soup.get_text())
        except Exception as e:
            self.logger.error(f"获取网页内容失败: {url}, 错误: {str(e)}")
            return None

class ReflectionAgent:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.logger = logging.getLogger(__name__ + ".ReflectionAgent")

    def process(self, messages: List[Dict[str, str]], original_query: str) -> str:
        self.logger.info("ReflectionAgent 开始处理消息")
        
        current_date = datetime.now().strftime("%Y年%m月%d日")
        
        system_prompt = REFLECTION_AGENT_PROMPT_CN.format(current_date=current_date, original_query=original_query)

        system_message = {"role": "system", "content": system_prompt}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[system_message] + messages,
            temperature=0.7,
            max_tokens=5000
        )

        reflection = response.choices[0].message.content
        self.logger.info(f"ReflectionAgent 响应:\n{reflection[:500]}...")  # 打印前500个字符
        return reflection

class Searcher:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.search_agent = SearchAgent(client, model)
        self.reflection_agent = ReflectionAgent(client, model)
        self.logger = logging.getLogger(__name__ + ".DialogueManager")

    def search(self, query: str, max_turns: int = 5) -> str:
        self.logger.info(f"开始对话，原始问题: {query}")
        
        search_messages = [{"role": "user", "content": f"帮我搜索以下问题的答案：{query}"}]
        
        for turn in range(max_turns):
            self.logger.info(f"\n--- 对话轮次 {turn + 1} ---")
            
            # SearchAgent 回合
            self.logger.info("SearchAgent 处理中...")
            search_response = self.search_agent.process(search_messages)
            self.logger.info(f"SearchAgent 回答:\n{search_response[:500]}...")  # 打印前500个字符
            search_messages.append({"role": "assistant", "content": search_response})
            
            # 将最后一条消息的角色改为 "user"，以便传递给 ReflectionAgent
            reflection_messages = search_messages[:-1] + [{"role": "user", "content": search_response}]
            
            # ReflectionAgent 回合
            self.logger.info("ReflectionAgent 处理中...")
            reflection = self.reflection_agent.process(reflection_messages, query)
            self.logger.info(f"ReflectionAgent 回答:\n{reflection[:500]}...")  # 打印前500个字符
            
            # 检查ReflectionAgent的响应
            if "<|end|>" in reflection:
                self.logger.info("ReflectionAgent 指示对话结束")
                return f"针对问题 '{query}' 的搜索答案如下：\n\n{reflection.replace('<|end|>', '').strip()}"
            else:
                search_messages.append({"role": "user", "content": reflection})
                self.logger.info("对话继续...")

        self.logger.warning(f"达到最大对话轮数 {max_turns}")
        final_message = {"role": "user", "content": f"请基于之前的所有对话内容，给出针对原始问题 '{query}' 的最终总结回答。"}
        search_messages.append(final_message)
        self.logger.info("生成最终总结...")
        final_reflection = self.reflection_agent.process(search_messages, query)
        final_reflection = f"针对原始问题 '{query}' 的最终总结回答如下：\n\n{final_reflection}"
        self.logger.info(f"最终总结:\n{final_reflection[:500]}...")  # 打印前500个字符
        return final_reflection.replace("<|end|>", "").strip()


class WebSearchGraph:
    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.adjacency_list: Dict[str, List[Dict]] = defaultdict(list)
        self.logger = logging.getLogger(__name__ + ".WebSearchGraph")

    def add_node(self, node_name: str, node_content: str, node_type: str = "searcher", thought: str = "", answer: str = "") -> None:
        self.nodes[node_name] = {
            "content": node_content,
            "type": node_type,
            "thought": thought,
            "answer": answer
        }
        self.logger.info(f"添加节点: {node_name}, 类型: {node_type}")

    def add_edge(self, start_node: str, end_node: str) -> None:
        self.adjacency_list[start_node].append({
            "id": str(uuid.uuid4()),
            "name": end_node,
            "state": 2
        })
        self.logger.info(f"添加边: {start_node} -> {end_node}")

    def get_node(self, node_name: str) -> Optional[Dict]:
        self.logger.info(f"获取节点信息: {node_name}")
        return self.nodes.get(node_name)

    def update_node(self, node_name: str, answer: str) -> None:
        if node_name in self.nodes:
            self.nodes[node_name]["answer"] = answer
            self.logger.info(f"更新节点答案: {node_name}")

    def get_parent_nodes(self, node_name: str) -> List[str]:
        parents = [node for node, edges in self.adjacency_list.items() if any(edge["name"] == node_name for edge in edges)]
        self.logger.info(f"获取父节点: {node_name}, 父节点: {parents}")
        return parents

    def get_root_node(self) -> Optional[str]:
        root = next((node for node, data in self.nodes.items() if data["type"] == "root"), None)
        self.logger.info(f"获取根节点: {root}")
        return root

    def to_natural_language(self) -> str:
        self.logger.info("将图状态转换为自然语言描述")
        root_node = self.get_root_node()
        if not root_node:
            return "当前没有问题。"

        def node_to_text(node_name: str, depth: int = 0) -> str:
            node = self.nodes[node_name]
            indent = "  " * depth
            text = f"{indent}节点 ID: {node_name}\n"
            text += f"{indent}问题内容: {node['content']}\n"
            if node['answer']:
                text += f"{indent}答案: {node['answer']}\n"
            else:
                text += f"{indent}状态: 尚未回答\n"
            
            if node['thought']:
                text += f"{indent}思考过程: {node['thought']}\n"
            
            children = self.adjacency_list.get(node_name, [])
            if children:
                text += f"{indent}子问题:\n"
                for child in children:
                    text += node_to_text(child['name'], depth + 1)
            return text

        description = "问题层次结构:\n\n"
        description += node_to_text(root_node)


        return description

    def get_structured_state(self) -> Dict:
        self.logger.info("获取结构化状态")
        def build_question_hierarchy(node_name: str) -> Dict:
            node = self.nodes[node_name]
            children = [
                build_question_hierarchy(child["name"])
                for child in self.adjacency_list[node_name]
            ]
            return {
                "question": node["content"],
                "type": node["type"],
                "answer": node["answer"],
                "thought": node["thought"],
                "status": "answered" if node["answer"] else "pending",
                "subquestions": children
            }

        root_node = self.get_root_node()
        if root_node:
            hierarchy = build_question_hierarchy(root_node)
        else:
            hierarchy = {}

        state = {
            "question_hierarchy": hierarchy,
            "total_nodes": len(self.nodes),
            "answered_nodes": sum(1 for node in self.nodes.values() if node["answer"]),
            "pending_nodes": sum(1 for node in self.nodes.values() if not node["answer"])
        }
        self.logger.info(f"结构化状态: {json.dumps(state, indent=2)}")
        return state

    def to_json(self) -> str:
        self.logger.info("将图状态转换为JSON")
        return json.dumps(self.get_structured_state(), indent=2, ensure_ascii=False)

class Planner:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.logger = logging.getLogger(__name__ + ".Planner")

    def plan(self, question: str, structured_state: Dict) -> Dict:
        self.logger.info(f"开始规划: {question}")
        messages = self._build_messages(question, structured_state)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=5000
            )
            
            plan_text = response.choices[0].message.content

            if "<|end|>" in plan_text:
                self.logger.info("规划结束: 收到结束信号")
                return {"end": True}
            
            plan = extract_json(plan_text)
            if plan is None:
                self.logger.error("规划解析失败: JSON解码错误")
                return {"subquestions": []}
            try:
                # plan = json.loads(plan_text)
                for subq in plan.get("subquestions", []):
                    if "thought" not in subq:
                        subq["thought"] = "未提供思考过程"
                self.logger.info(f"规划完成: 生成 {len(plan.get('subquestions', []))} 个子问题")
                return plan
            except json.JSONDecodeError:
                self.logger.error("规划解析失败: JSON解码错误")
                return {"subquestions": []}
        
        except Exception as e:
            self.logger.error(f"搜索过程中发生未预期的错误: {e}")
            return {"error": str(e)}

    def _build_messages(self, question: str, structured_state: Dict) -> List[Dict]:
        self.logger.info("构建LLM消息")
        current_state = structured_state['graph'].to_natural_language()
        user_message = f"""
主问题: {question}

当前问题解决进度：
{current_state}

请仔细分析上述问题解决进度，并按照以下步骤决定下一步行动：

1. 首先评估主问题是否已经得到充分解答：
   - 检查现有的子问题及其答案是否足够回答主问题。
   - 如果你认为主问题已经得到充分解答，请输出以下格式：
     <|end|>
     理由：[在这里简明扼要地解释为什么你认为主问题已经得到充分解答]

2. 只有在主问题尚未得到充分解答的情况下，才考虑以下行动：
   a. 提供下一步的问题分解计划，针对尚未解答或需要进一步探讨的方面。
   b. 在制定计划时，考虑已有的答案，避免重复已解答的内容。
   c. 如果需要对某个已有的答案进行补充或澄清，可以提出相关的子问题。
   d. 在提供子问题时，使用节点的唯一标识符（如 'root', 'subq_1', 'subq_2' 等）作为父节点引用，而不是使用问题内容。

3. 输出格式：
   - 如果决定结束，使用上述的 <|end|> 格式，包含简明的结束理由。
   - 如果继续分解问题，请严格按照系统消息中描述的 JSON 格式输出你的计划。

请记住，避免不必要的循环是很重要的。如果现有的信息已经足够回答主问题，就应该结束过程并提供结束的理由。
"""

        return [
            {"role": "system", "content": PLANNER_PROMPT_CN},
            {"role": "user", "content": user_message}
        ]

class SummarizeAgent:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.logger = logging.getLogger(__name__ + ".SummarizeAgent")

    def summarize(self, structured_state: Dict) -> str:
        self.logger.info("开始生成最终答案")
        messages = self._build_messages(structured_state)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=8000
            )
            
            summary = response.choices[0].message.content
            self.logger.info("最终答案生成完成")
            return summary
        
        except Exception as e:
            self.logger.error(f"生成总结答案时出错: {e}")
            return "无法生成总结答案。"

    def _build_messages(self, structured_state: Dict) -> List[Dict]:
        self.logger.info("构建总结消息")
        system_message = FINAL_RESPONSE_CN
        current_state = structured_state['graph'].to_natural_language()
        root_node = structured_state['graph'].nodes.get('root', {})
        root_question = root_node.get('content', "未找到根问题")

        user_message = f"""
我们要解决的问题是：{root_question}

以下是搜索过程中的问答对：

{current_state}

请根据以上信息，生成一个详细完备的最终回答。
"""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

class SmartSearch:
    def __init__(self, api_key: str, base_url: str, model: str, max_turn: int = 10):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.planner = Planner(self.client, model)
        self.summarize_agent = SummarizeAgent(self.client, model)
        self.searcher = Searcher(self.client, model)
        self.graph = WebSearchGraph()
        self.max_turn = max_turn
        self.logger = logging.getLogger(__name__ + ".SmartSearch")

    def search(self, question: str) -> Generator[Dict, None, None]:
        self.logger.info(f"开始搜索流程: {question}")
        self.graph.add_node("root", question, "root", thought="初始问题")
        yield {"status": "Started", "question": question}

        for turn in range(self.max_turn):
            self.logger.info(f"开始第 {turn + 1} 轮搜索")
            structured_state = {"graph": self.graph}
            
            yield {"status": "Planning", "progress": "开始规划"}
            plan = self.planner.plan(question, structured_state)
            yield {"status": "Planning", "progress": "规划完成"}
            
            if "error" in plan:
                self.logger.error(f"规划过程中出错: {plan['error']}")
                yield {"status": "Error", "error": plan['error']}
                return

            if plan.get("end", False):
                self.logger.info("搜索结束: 收到结束信号")
                final_answer = self.summarize_agent.summarize(structured_state)
                self.graph.add_node("response", final_answer, "response")
                yield {"status": "Completed", "answer": final_answer}
                return

            search_tasks = []
            for subq in plan.get("subquestions", []):
                subq_content = subq["content"]
                subq_name = f"subq_{len(self.graph.nodes)}"
                self.graph.add_node(subq_name, subq_content, thought=subq.get("thought", ""))
                # 这里需要确保 subq["parent"] 是正确的节点标识符
                parent_node = subq.get("parent", "root")  # 默认为 root 如果没有指定父节点
                self.graph.add_edge(parent_node, subq_name)
                search_tasks.append((subq_name, subq_content))

            self.logger.info(f"开始搜索 {len(search_tasks)} 个子问题")
            search_results = self._search(search_tasks)

            for subq_name, subq_content, summary in search_results:
                self.graph.update_node(subq_name, summary)
                yield {
                    "status": "Searched",
                    "subquestion": subq_content,
                    "summary": summary
                }

        # 如果循环正常结束（达到最大搜索轮数）
        self.logger.warning(f"达到最大搜索轮数 {self.max_turn}")
        structured_state = {"graph": self.graph}
        final_answer = self.summarize_agent.summarize(structured_state)
        self.graph.add_node("response", final_answer, "response")
        yield {"status": "MaxTurnReached", "answer": final_answer}

    def _search(self, search_tasks: List[tuple]) -> List[tuple]:
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:  # 你可以根据需要调整 max_workers
            future_to_task = {executor.submit(self._execute_search, subq_name, subq_content): (subq_name, subq_content) for subq_name, subq_content in search_tasks}
            for future in as_completed(future_to_task):
                subq_name, subq_content = future_to_task[future]
                try:
                    summary = future.result()
                    self.logger.info(f"完成搜索和总结: {subq_name}")
                    results.append((subq_name, subq_content, summary))
                except Exception as exc:
                    self.logger.error(f"{subq_name} 搜索失败: {exc}")
                    results.append((subq_name, subq_content, f"搜索失败: {str(exc)}"))
        return results

    def _execute_search(self, subq_name: str, subq_content: str) -> str:
        self.logger.info(f"开始搜索和总结: {subq_name}")
        return self.searcher.search(subq_content)

def main():
    api_key = ""
    base_url = ""
    model = ""

    smart_search = SmartSearch(api_key, base_url, model)

    question = "请总结2024年美国大选候选人有哪些，他们的优势劣势分别是什么"
    logger.info(f"开始处理问题: {question}")
    for step in smart_search.search(question):
        print(json.dumps(step, indent=2, ensure_ascii=False))
        logger.info(f"搜索步骤: {step['status']}")
        if step['status'] in ['Completed', 'Error', 'MaxTurnReached']:
            break
    logger.info("问题处理完成")

if __name__ == "__main__":
    main()