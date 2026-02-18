"""
Learner Agent

Helps user in providing learning material that can be used to study a certain concept
"""

from typing import Any, Dict, Optional

import structlog
from pydantic import BaseModel

from app.agents.agent_types import LEARNER_AGENT_NAME
from app.agents.base_agent import BaseAgent
from app.agents.llm_models import LLMModels
from app.agents.state import ExamHelperState
from app.models.response_models import ExamHelperResponse
from langchain.agents import create_agent

from app.tools.firecrawl_tool import get_learner_tools

logger = structlog.get_logger(__name__)


LEARNER_AGENT_PROMPT = """

Todo

CONVERSATION CONTEXT:
{context}

"""

def _extract_text_from_message(message) -> str:
    """
    Convert structured message into a clean string.

    Handles:
    - Gemini content blocks (list of dicts)
    - plain string segments
    - mixed content safely
    """
    content = message.content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
            else:
                parts.append(str(block))

        return "\n".join(p for p in parts if p.strip())

    return content

class LearnerAgent(BaseAgent):
    """Agent for handling queries related to providing easy to grasp learning material"""

    def __init__(
        self,
        agent_name: str = LEARNER_AGENT_NAME,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        model_name: str = LLMModels.GEMINI_2_5_FLASH,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            api_key=api_key,
            temperature=temperature,
            model_name=model_name,
        )

    def get_result_key(self) -> str:
        return "learner_agent_result"

    def get_prompt(self, state: Optional[ExamHelperState] = None) -> str:
        from app.agents.state import get_conversation_context

        context = get_conversation_context(state) if state else ""
        return LEARNER_AGENT_PROMPT.format(context=context)

    def get_response_format(self) -> type[BaseModel]:
        return ExamHelperResponse

    async def process_query(
        self,
        query: str,
        state: Optional[ExamHelperState] = None,
    ) -> Dict[str, Any]:
        """Process a query and provide related learning material"""
        try:
            from langchain_core.messages import HumanMessage

            prompt = self.get_prompt(state)

            agent = create_agent(
                model=self.model,
                tools=get_learner_tools(),
                system_prompt=prompt,
            )

            result = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(content=query)
                    ]
                }
            )

            final_output = _extract_text_from_message(result["messages"][-1])

            return {
                "success": True,
                self.get_result_key(): final_output,
                "error": [],
            }

        except Exception as e:
            logger.error("Learner agent processing failed", error=str(e))
            return {
                "success": False,
                self.get_result_key(): None,
                "error": [str(e)],
            }