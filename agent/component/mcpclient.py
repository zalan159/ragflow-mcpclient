import asyncio
import os
import json
from typing import Optional, Union, Dict, List, Any
from contextlib import AsyncExitStack
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import  sse_client
from mcp.client.stdio import stdio_client
from agent.component.base import ComponentBase, ComponentParamBase
import pandas as pd
from functools import partial
import re
from api.db.services.conversation_service import structure_answer
from api.db.services.llm_service import LLMBundle
from api import settings
from api.db import LLMType
import types
import functools

class MCPServerConfig:
    """MCP服务器配置"""
    def __init__(
        self, 
        url: Optional[str] = None,
        command: Optional[str] = None,
        args: List[str] = None,
        env: Dict[str, str] = None,
        headers: Optional[Dict[str, Any]] = None,
        timeout: int = 5,
        sse_read_timeout: int = 60 * 5
    ):
        self.url = url
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.headers = headers or {}
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout
        
    def validate(self):
        # 互斥校验
        if not (self.url or self.command):
            raise ValueError("必须提供 url 或 command")
        if self.url and self.command:
            raise ValueError("url 和 command 不能同时存在")
        
        # SSE模式校验
        if self.url:
            if not (self.url.startswith('http://') or self.url.startswith('https://')):
                raise ValueError("URL 必须是 http:// 或 https:// 开头")
            if self.headers and not isinstance(self.headers, dict):
                raise TypeError("headers 必须是字典类型")
            if not isinstance(self.timeout, int):
                raise TypeError("timeout 必须是整数")
            if not isinstance(self.sse_read_timeout, int):
                raise TypeError("sse_read_timeout 必须是整数")
        
        # 标准输入输出模式校验
        if self.command:
            if not isinstance(self.command, str):
                raise TypeError("command 必须是字符串")
            if self.args and not isinstance(self.args, list):
                raise TypeError("args 必须是列表类型")
            if self.env and not isinstance(self.env, dict):
                raise TypeError("env 必须是字典类型")

class MCPParam(ComponentParamBase):
    """
    定义MCP组件的参数
    """
    def __init__(self, name: str = None):
        super().__init__()
        self.prompt = ""
        self.debug_inputs = []
        self.server_config: Optional[Union[MCPServerConfig, dict]] = None  # 允许字典类型
        self.cite = True  # 新增引用开关
        self.history_window = 5  # 新增历史截断长度
        if name:
            self.set_name(name)

    def set_server(self, server_config: Union[MCPServerConfig, dict]):
        """处理字典类型的配置"""
        if isinstance(server_config, dict):
            self.server_config = MCPServerConfig(**server_config)
        else:
            self.server_config = server_config

    def check(self):
        self.check_empty(self.prompt, "[MCP] Prompt")
        if not self.server_config:
            raise ValueError("[MCP] 需要设置服务器配置")
        # 添加类型判断和转换
        if isinstance(self.server_config, dict):
            self.server_config = MCPServerConfig(**self.server_config)
        self.server_config.validate()

    def gen_conf(self):
        return {
            "server": {
                "url": self.server_config.url,
                "command": self.server_config.command,
                "args": self.server_config.args,
                "env": self.server_config.env
            }
        }

class MCP(ComponentBase):
    component_name = "MCP"
    
    # API配置
    API_KEY = "your api key"
    BASE_URL = "https://api.deepseek.com/v1"
    MODEL_NAME = "deepseek-chat"

    def __init__(self, canvas, id, param: Union[str, MCPParam]):
        if isinstance(param, str):
            param = MCPParam(param)
        super().__init__(canvas, id, param)
        self.client = None
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.stdio = None
        self.write = None

    async def connect_to_server(self) -> ClientSession:
        """连接到服务器"""
        # 在连接前确保清理旧资源
        await self.disconnect_from_server()
        
        server_config = self._param.server_config
        print("正在连接到服务器...")

        try:
            async def _connect():
                # 使用新的exit_stack管理连接
                if server_config.url:
                    stdio_transport = await self.exit_stack.enter_async_context(
                        sse_client(
                            server_config.url,
                            headers=server_config.headers,
                            timeout=server_config.timeout,
                            sse_read_timeout=server_config.sse_read_timeout
                        )
                    )
                else:
                    server_params = StdioServerParameters(
                        command=server_config.command,
                        args=server_config.args,
                        env=server_config.env
                    )
                    stdio_transport = await self.exit_stack.enter_async_context(
                        stdio_client(server_params)
                    )

                self.stdio, self.write = stdio_transport
                self.session = await self.exit_stack.enter_async_context(
                    ClientSession(self.stdio, self.write)
                )
                await self.session.initialize()
                return self.session

            return await asyncio.wait_for(_connect(), timeout=10)

        except asyncio.TimeoutError:
            print("连接服务器超时")
            raise
        except Exception as e:
            print(f"连接服务器时发生错误: {str(e)}")
            raise

    async def disconnect_from_server(self):
        """断开与服务器的连接"""
        try:
            # 先关闭底层传输
            if self.stdio:
                await self.stdio.aclose()
            if self.write:
                await self.write.aclose()
        except Exception as e:
            print(f"关闭传输时发生错误: {str(e)}")
        finally:
            try:
                # 最后关闭 exit_stack
                if self.exit_stack:
                    await self.exit_stack.aclose()
            except RuntimeError as e:
                if "different task" not in str(e):
                    print(f"断开连接时发生错误: {str(e)}")
            finally:
                # 创建新的 AsyncExitStack 实例
                self.exit_stack = AsyncExitStack()
                # 重置其他连接相关属性
                self.session = None
                self.stdio = None
                self.write = None

    async def cleanup(self):
        """清理服务器连接"""
        try:
            await self.disconnect_from_server()
        except Exception as e:
            print(f"清理资源时发生错误: {str(e)}")
            # 不抛出异常，确保清理过程不会中断主流程

    def get_input_elements(self):
        """获取输入元素"""
        return [{"key": "user", "name": "Input your question here:"}]

    async def process_query(self, query: str) -> str:
        """处理查询并调用工具"""
        print("开始处理查询...")
        messages = [{"role": "user", "content": query}]
        
        try:
            # 连接到服务器并获取工具列表
            print("正在连接到服务器...")
            session = await self.connect_to_server()
            print("服务器连接成功，正在获取工具列表...")
            response = await session.list_tools()
            tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
            } for tool in response.tools]

            # 添加循环处理工具调用
            max_iterations = 10  # 安全阀防止无限循环
            while max_iterations > 0:
                max_iterations -= 1
                
                # 调用OpenAI API
                response = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[{"role": "system", "content": self._param.prompt}] + messages,
                    tools=tools,
                    temperature=0.3
                )

                content = response.choices[0]
                print(f"API 返回的 finish_reason: {content.finish_reason}")
                
                if content.finish_reason != "tool_calls":
                    return content.message.content

                # 处理所有工具调用
                tool_messages = []
                for tool_call in content.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"<think>准备调用工具: {tool_name}")
                    result = await session.call_tool(tool_name, tool_args)
                    
                    # 为每个工具调用添加响应
                    tool_messages.append({
                        "role": "tool",
                        "content": result.content[0].text,
                        "tool_call_id": tool_call.id,
                    })

                # 更新消息历史
                messages.append(content.message.model_dump())
                messages.extend(tool_messages)

            return "达到最大迭代次数，终止处理"

        except Exception as e:
            print(f"处理查询时发生错误: {str(e)}")
            import traceback
            print("错误堆栈:")
            print(traceback.format_exc())
            raise
        finally:
            await self.cleanup()

    async def process_query_streamly(self, query: str, system_prompt: str, history: list):
        """修改为真正的异步生成器"""
        print(f"[MCP Debug] 进入流式处理流程 | 初始消息长度: {len(history)}")
        messages = []
        full_content = ""  # 新增全量内容累加器
        
        try:
            if history:
                print(f"[MCP Debug] 合并历史消息 | 原消息数: {len(messages)} | 新增历史消息数: {len(history)}")
                # 过滤无效历史消息并转换格式
                messages = [
                    {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                        # 添加工具调用信息（如果存在）
                        "tool_calls": msg.get("tool_calls", []),
                        "name": msg.get("name", "")
                    } 
                    for msg in history if isinstance(msg, dict)
                ]
            
            # 添加当前用户消息
            messages.append({"role": "user", "content": query})
            
            print("[MCP Debug] 正在连接服务器...")
            session = await self.connect_to_server()
            print("[MCP Debug] 服务器连接成功，获取工具列表...")
            response = await session.list_tools()
            print(f"[MCP Debug] 获取到{len(response.tools)}个工具")
            tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
            } for tool in response.tools]

            # 使用传入的system_prompt（关键修改）
            system_message = {"role": "system", "content": system_prompt}
            
            max_iterations = 10
            full_content = ""  # 重置全量内容（每次工具调用后重新开始）
            while max_iterations > 0:
                max_iterations -= 1
                
                # 修改流处理逻辑（关键修复）
                stream = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[system_message, *messages],
                    tools=tools,
                    temperature=0.3,
                    stream=True
                )

                
                tool_calls = []
                current_tool_call = None
                
                # 同步流转换为异步生成器
                for chunk in stream:
                    await asyncio.sleep(0)
                    content = chunk.choices[0].delta
                    if content.content:
                        full_content += content.content  # 累加内容
                        # 改为发送全量内容（关键修改）
                        yield self._format_chunk({"content": full_content}, 0)
                    
                    # 收集工具调用信息
                    if content.tool_calls:
                        for tool_call in content.tool_calls:
                            if tool_call.id:
                                current_tool_call = {
                                    "id": tool_call.id,
                                    "function": {
                                        "name": "",
                                        "arguments": ""
                                    }
                                }
                                tool_calls.append(current_tool_call)
                            
                            if current_tool_call:
                                current_tool_call["function"]["name"] += tool_call.function.name or ""
                                current_tool_call["function"]["arguments"] += tool_call.function.arguments or ""
                
                # 处理工具调用后生成最终结果
                if tool_calls:
                    # 在调用工具前添加提示信息
                    tool_call_prompt = "<think>\n\n（正在调用工具："
                    full_content += tool_call_prompt
                    yield self._format_chunk({"content": full_content}, 0)
                    
                    assistant_msg = {
                        "role": "assistant",
                        "content": full_content,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"]
                                },
                                "type": "function"
                            } for tc in tool_calls
                        ]
                    }
                    messages.append(assistant_msg)
                    
                    tool_messages = []
                    for tool_call in tool_calls:
                        try:
                            # 添加工具调用开始提示
                            tool_start = f"{tool_call['function']['name']}("
                            full_content += tool_start
                            yield self._format_chunk({"content": full_content}, 0)
                            
                            # 添加参数提示
                            args_str = tool_call["function"]["arguments"].strip()
                            full_content += f"{args_str})..."
                            yield self._format_chunk({"content": full_content}, 0)
                            
                            # 实际调用工具
                            tool_name = tool_call["function"]["name"]
                            tool_args = json.loads(args_str)
                            result = await session.call_tool(tool_name, tool_args)
                            
                            # 添加结果提示
                            tool_result = f"\n工具返回：{result.content[0].text if result.content else '无返回内容'}\n"
                            full_content += tool_result
                            yield self._format_chunk({"content": full_content}, 0)
                            
                            tool_messages.append({
                                "role": "tool",
                                "content": result.content[0].text if result.content else "无返回内容",
                                "tool_call_id": tool_call["id"],
                            })
                            
                            # 添加调用结束提示
                            tool_end = "）</think>"
                            full_content += tool_end
                            yield self._format_chunk({"content": full_content}, 0)

                        except json.JSONDecodeError:
                            error_msg = f"\n（参数解析失败：{args_str}）</think>"
                            full_content += error_msg
                            yield self._format_chunk({"content": full_content}, 0)
                        except Exception as e:
                            error_msg = f"\n（工具执行失败：{str(e)}）</think>"
                            full_content += error_msg
                            yield self._format_chunk({"content": full_content}, 0)
                    
                    messages.extend(tool_messages)
                else:
                    break

        except Exception as e:
            print(f"[MCP Error] 流式处理异常: {str(e)}")
            import traceback
            print(f"异常堆栈:\n{traceback.format_exc()}")
            raise
        finally:
            await self.cleanup()

    def _run(self, history, **kwargs):
        """修改后的运行方法支持流式"""
        # 输入预处理（新增）
        print(f"[MCP Debug] 开始处理输入 | 历史消息: {history}")

        # 从history中提取最新用户消息（关键修改）
        query = ""
        if history and isinstance(history[-1], (list, tuple)) and history[-1][0] == "user":
            query = history[-1][1]
            print(f"[MCP Debug] 从history提取用户问题: {query}")
        else:
            query = kwargs.get("user", "")
            print(f"[MCP Debug] 从kwargs获取用户问题: {query}")

        # 参数替换到prompt（保持原有逻辑）
        original_prompt = self._param.prompt
        for n, v in kwargs.items():
            self._param.prompt = re.sub(r"\{%s\}" % re.escape(n), str(v).replace("\\", " "), self._param.prompt)
        print(f"[MCP Debug] Prompt替换结果 | 原始: {original_prompt[:50]}... | 替换后: {self._param.prompt[:50]}...")

        # 构造带引用的系统提示（保持原有逻辑）
        query, citations = self._preprocess_input(query)
        system_prompt = self._build_system_prompt(citations)
        print(f"[MCP Debug] 系统提示构建完成: {system_prompt[:100]}...")

        # 历史对话截断（关键修改）
        truncated_history = []
        if history:
            # 排除最后一条用户消息（当前query）
            truncated_history = self._truncate_history(history[:-1]) 
            print(f"[MCP Debug] 截断历史（排除最新用户消息）| 原长度: {len(history)} | 新长度: {len(truncated_history)}")
        
        # 初始化OpenAI客户端（保持原有逻辑）
        self.client = OpenAI(
            api_key=self.API_KEY,
            base_url=self.BASE_URL
        )

        # 流式处理判断（保持原有逻辑）
        downstreams = []
        if self._canvas:
            component_info = self._canvas.get_component(self._id)
            downstreams = component_info.get("downstream", []) if component_info else []
        
        if kwargs.get("stream") and downstreams:
            return functools.partial(self._sync_stream_wrapper, query, system_prompt, truncated_history)

        # 非流式处理（调整参数传递）
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(
                self.process_query(query, system_prompt, truncated_history)  # 传递截断后的历史
            )
            if self._param.cite:
                response = self._add_citations(response, citations)
            return pd.DataFrame([{"content": response}])
        finally:
            loop.close()

    # 新增同步包装方法
    def _sync_stream_wrapper(self, query, system_prompt, history):
        """优化事件循环管理"""
        print(f"[SYNC WRAPPER] 进入同步包装器")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async_gen = self.stream_output(query, system_prompt, history)
        
        try:
            while True:
                try:
                    chunk = loop.run_until_complete(async_gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            loop.close()
            print("[SYNC WRAPPER] 事件循环已关闭")

    # 新增辅助方法
    def _preprocess_input(self, query: str) -> tuple:
        """增强的预处理方法，支持检索"""
        # 解析显式引用
        citations = []
        pattern = r"@\[([^\]]+)\]\[([^\]]+)\]"
        matches = re.findall(pattern, query)
        
        # 处理显式引用
        for match in matches:
            citations.append({"name": match[0], "id": match[1]})
            query = query.replace(f"@[{match[0]}][{match[1]}]", "")
        
        # 添加隐式检索逻辑（与Generate一致）
        retrieval_res = self.get_input()
        # 增加字段存在性检查（关键修改）
        if not retrieval_res.empty and "doc_id" in retrieval_res.columns:
            citations.extend([{
                "doc_id": row.get("doc_id", ""),
                "docnm_kwd": row.get("docnm_kwd", ""),
                "content_ltks": row.get("content_ltks", ""),
                "vector": row.get("vector", "")
            } for _, row in retrieval_res.iterrows()])
        
        return query.strip(), citations  # 移除pd.DataFrame转换

    def _build_system_prompt(self, citations: list) -> str:  # 参数类型改为list
        """构建带引用的系统提示"""
        base_prompt = self._param.prompt
        if not citations:
            return base_prompt
        
        citation_text = "\n".join([
            f"文档《{cite['name']}》（ID:{cite['id']}）已提供，请优先参考"
            if 'name' in cite else  # 处理不同格式的引用
            f"文档《{cite['docnm_kwd']}》（ID:{cite['doc_id']}）已提供，请优先参考"
            for cite in citations
        ])
        
        return f"{base_prompt}\n\n当前对话可参考以下文档：\n{citation_text}"

    def _truncate_history(self, history: list) -> list:
        """历史对话截断"""
        window_size = max(0, self._param.history_window)
        return history[-window_size:]

    def _add_citations(self, response: str, citations: list) -> dict:
        """统一的结构化输出"""
        if not citations:
            return {"content": response, "reference": {"chunks": [], "doc_aggs": []}}
        
        return self.set_cite(citations, response)

    async def stream_output(self, query, system_prompt, history):
        """优化后的流式输出方法"""
        try:
            print(f"[STREAM OUTPUT] 开始流式输出处理 | query: {query[:50]}... | history长度: {len(history)}")
            full_response = ""
            chunk_count = 0
            
            # 获取原始生成器
            raw_generator = self.process_query_streamly(query, system_prompt, history)
            
            # 异步迭代并解包（关键修改）
            async for raw_chunk in raw_generator:
                # 如果chunk本身是生成器，进行深度解包
                if isinstance(raw_chunk, types.GeneratorType):
                    print(f"[STREAM OUTPUT] 检测到嵌套生成器，开始深度解包")
                    try:
                        while True:
                            sub_chunk = await raw_chunk.__anext__()
                            chunk_count += 1
                            formatted = self._format_chunk(sub_chunk, chunk_count)
                            yield formatted
                    except StopAsyncIteration:
                        continue
                else:
                    chunk_count += 1
                    formatted = self._format_chunk(raw_chunk, chunk_count)
                    yield formatted

            print(f"[STREAM OUTPUT] 流式处理完成 | 总chunk数: {chunk_count} | 总长度: {len(full_response)}")
            
        except Exception as e:
            print(f"[STREAM OUTPUT ERROR] 流式输出异常: {str(e)}")
            import traceback
            print(f"异常堆栈:\n{traceback.format_exc()}")
            raise
        finally:
            print("[STREAM OUTPUT] 开始清理资源...")
            await self.cleanup()

    def _format_chunk(self, chunk, count):
        """统一格式化输出块"""
        # 确保输出结构符合Canvas预期
        return {
            "content": chunk.get("content", ""),
            "running_status": count == 1,  # 首个chunk标记为运行状态
            "reference": chunk.get("reference", {})
        }

    @staticmethod
    def be_output(content):
        """格式化输出"""
        return pd.DataFrame([{"content": content}]) 

    def debug(self, **kwargs):
        """调试方法，用于测试组件功能"""
        if self._param.debug_inputs:
            query = self._param.debug_inputs[0].get("value", "")
        else:
            query = kwargs.get("user", "")
            
        if not query:
            return pd.DataFrame([{"content": "请输入您的问题"}])
            
        self._param.check()
            
        self.client = OpenAI(
            api_key=self.API_KEY,
            base_url=self.BASE_URL
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(self.process_query(query))
            loop.run_until_complete(self.cleanup())
            return pd.DataFrame([{"content": response}])
        finally:
            loop.close() 

    def set_cite(self, retrieval_res, answer):
        """与Generate组件保持一致的引用处理逻辑"""
        # 确保包含必要字段
        retrieval_res = retrieval_res.dropna(subset=["vector", "content_ltks"]).reset_index(drop=True)
        
        # 插入引用标注
        answer, idx = settings.retrievaler.insert_citations(
            answer,
            [ck["content_ltks"] for _, ck in retrieval_res.iterrows()],
            [ck["vector"] for _, ck in retrieval_res.iterrows()],
            LLMBundle(self._canvas.get_tenant_id(), LLMType.EMBEDDING, 
                     self._canvas.get_embedding_model()),
            tkweight=0.7,
            vtweight=0.3
        )
        
        # 构建参考文献结构
        doc_ids = set()
        recall_docs = []
        for i in idx:
            did = retrieval_res.loc[int(i), "doc_id"]
            if did not in doc_ids:
                doc_ids.add(did)
                recall_docs.append({
                    "doc_id": did,
                    "doc_name": retrieval_res.loc[int(i), "docnm_kwd"]
                })
        
        reference = {
            "chunks": [ck.to_dict() for _, ck in retrieval_res.iterrows()],
            "doc_aggs": recall_docs
        }
        
        # 结构化输出
        res = {"content": answer, "reference": reference}
        return structure_answer(None, res, "", "") 