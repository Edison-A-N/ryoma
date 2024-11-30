import textwrap
from typing import Literal
from pydantic import BaseModel
import json
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage

from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from ryoma.core.tool.html import HTMLParser
from ryoma.core.llm import create_llm
from ryoma.core.logging import logger


class URLInput(BaseModel):
    url: str


async def async_read_url(inputs: URLInput) -> str:
    """Read content from a URL"""
    html_tool = HTMLParser()
    result = await html_tool.parse_url(inputs.url)
    return result


read_url = StructuredTool.from_function(coroutine=async_read_url)


class URLSummaryWorkflow:
    def __init__(self):
        self.model = create_llm().get_model()
        self.llm = self.model.bind_tools([read_url])
        self.tool_node = ToolNode([read_url])
        self.app = self._create_workflow()

    def should_continue(self, state: MessagesState) -> Literal["tools", END]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    async def call_model(self, state: MessagesState):
        messages = state["messages"]
        response = await self.llm.ainvoke(messages)
        return {"messages": [response]}

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(state_schema=MessagesState)

        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self.should_continue)
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    async def process(self, url: str) -> dict:
        """Process URL and return content summary"""
        initial_state = {
            "messages": [
                HumanMessage(
                    content=textwrap.dedent(
                        f"""
                        Please read and summarize the content from {url}.

                        Return a JSON object with exactly two fields:
                        - "title": the page title
                        - "content": a summary of the main content

                        Return ONLY the JSON object, no other text.
                        Ensure all text is properly escaped for JSON, especially:
                        - non-ASCII characters
                        - double quotes (") should be escaped as \\\"
                        Do not include any line breaks in the content.
                        """
                    )
                ),
            ]
        }

        try:
            result = await self.app.ainvoke(initial_state)
            summary = result["messages"][-1].content

            # Try to parse JSON from the summary
            try:
                # Ensure proper JSON encoding for non-ASCII characters
                parsed_json = json.loads(summary, strict=False)
                return {"summary": parsed_json, "error": None}
            except json.JSONDecodeError as je:
                # Try to clean and re-parse the JSON if initial parsing fails
                cleaned_summary = summary.strip().replace("\n", "")
                try:
                    parsed_json = json.loads(cleaned_summary, strict=False)
                    return {"summary": parsed_json, "error": None}
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON: {str(je)}")
                    return {
                        "summary": summary,
                        "error": f"JSON parsing failed: {str(je)}",
                    }

        except Exception as e:
            logger.error(f"Failed to process URL: {str(e)}", exc_info=True)
            return {"summary": None, "error": f"Workflow failed: {str(e)}"}
