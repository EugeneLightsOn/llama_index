import asyncio
from threading import Thread
from typing import Any, List, Optional, Tuple, Dict

from llama_index.callbacks import trace_method
from llama_index.chat_engine.types import (
    CohereAgentChatResponse,
    CohereStreamingAgentChatResponse,
    ToolOutput,
)
from llama_index.core.base_retriever import BaseRetriever
from llama_index.core.llms.types import ChatMessage, MessageRole
from llama_index.llms.llm import LLM
from llama_index.schema import NodeWithScore
from llama_index.chat_engine import ContextChatEngine
from llama_index.llms.cohere_utils import transform_nodes_to_cohere_documents_list


class CohereContextChatEngine(ContextChatEngine):
    """Cohere Context + Citations Chat Engine.

    Uses a retriever to retrieve a context, set the context in the Cohere
    documents param(https://docs.cohere.com/docs/retrieval-augmented-generation-rag),
    and then uses an LLM to generate a response.

    NOTE: this is made to be compatible with Cohere's chat model Document mode
    """

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> CohereAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))
        context_str_template, nodes = self._generate_context(message)
        prefix_messages = self._get_prefix_messages_with_context(context_str_template)

        all_messages = self._memory.get_all()
        # transform nodes to cohere documents list
        documents_list = transform_nodes_to_cohere_documents_list(nodes)
        # and then uses an LLM to generate a response
        chat_response = self._llm.chat(all_messages, documents=documents_list)
        # TODO ask TJ do we need it?
        ai_message = chat_response.message
        self._memory.put(ai_message)

        return CohereAgentChatResponse(
            response=str(chat_response.message.content),
            citations=chat_response.raw.get("citations", []),
            documents=chat_response.raw.get("documents", []),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> CohereStreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        context_str_template, nodes = self._generate_context(message)
        prefix_messages = self._get_prefix_messages_with_context(context_str_template)
        all_messages = self._memory.get_all()
        documents_list = transform_nodes_to_cohere_documents_list(nodes)
        chat_response = CohereStreamingAgentChatResponse(
            chat_stream=self._llm.stream_chat(all_messages, documents=documents_list),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )
        thread = Thread(
            target=chat_response.write_response_to_history, args=(self._memory,)
        )
        thread.start()

        return chat_response

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> CohereAgentChatResponse:
        # TODO ask TJ do we need to set chat history?
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        context_str_template, nodes = await self._agenerate_context(message)
        prefix_messages = self._get_prefix_messages_with_context(context_str_template)
        all_messages = self._memory.get_all()
        documents_list = transform_nodes_to_cohere_documents_list(nodes)

        chat_response = await self._llm.achat(all_messages, documents=documents_list)
        ai_message = chat_response.message
        self._memory.put(ai_message)

        return CohereAgentChatResponse(
            response=str(chat_response.message.content),
            citations=chat_response.raw.get("citations", []),
            documents=chat_response.raw.get("documents", []),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> CohereStreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        context_str_template, nodes = await self._agenerate_context(message)
        prefix_messages = self._get_prefix_messages_with_context(context_str_template)

        all_messages = self._memory.get_all()
        documents_list = transform_nodes_to_cohere_documents_list(nodes)
        chat_response = StreamingAgentChatResponse(
            achat_stream=await self._llm.astream_chat(
                all_messages, documents=documents_list
            ),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )
        thread = Thread(
            target=lambda x: asyncio.run(chat_response.awrite_response_to_history(x)),
            args=(self._memory,),
        )
        thread.start()

        return chat_response

    def reset(self) -> None:
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history."""
        return self._memory.get_all()
