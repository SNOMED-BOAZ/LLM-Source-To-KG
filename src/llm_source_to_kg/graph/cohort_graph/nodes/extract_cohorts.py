import asyncio
from llm_source_to_kg.graph.cohort_graph.state import CohortGraphState
from llm_source_to_kg.utils.llm_util import get_llm
from llm_source_to_kg.utils.logger import get_logger
from llm_source_to_kg.schema.llm import LLMMessage, LLMConfig
from json_repair import repair_json

async def extract_cohorts(state: CohortGraphState) -> CohortGraphState:
    """
    코호트 추출 노드
    """
    doc_logger = get_logger(name=state["source_reference_number"])

    llm = get_llm(llm_type="gemini", model="gemini-2.0-flash")

    llm_config = LLMConfig(
        temperature=0.2,
        top_p=0.95,
        max_output_tokens=8192
    )
    prompt = open("../prompts/extract_cohort_prompt.txt", "r").read()

    messages = [
        LLMMessage(role="system", content=prompt),
        LLMMessage(role="user", content=state["document"])
    ]

    response = await llm.chat_llm(messages, llm_config)

    cohort_result = repair_json(response.content)

    doc_logger.info(f"{state['source_reference_number']} 코호트 추출 응답: {cohort_result}")

    state["cohort_result"] = cohort_result
    return state


# 테스트용 코드
if __name__ == "__main__":

    state = CohortGraphState(
        document=open("../../../../../datasets/guideline/contents/NG238.json", "r").read(),
        source_reference_number="NG238",
        cohort_result=[]
    )
    asyncio.run(extract_cohorts(state))