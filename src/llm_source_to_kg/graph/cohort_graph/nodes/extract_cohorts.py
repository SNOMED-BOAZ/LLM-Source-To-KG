import asyncio
import json
from llm_source_to_kg.graph.cohort_graph.state import CohortGraphState
from llm_source_to_kg.utils.llm_util import get_llm
from llm_source_to_kg.utils.logger import get_logger
from llm_source_to_kg.schema.llm import LLMMessage, LLMConfig
from json_repair import repair_json
from llm_source_to_kg.utils.s3 import get_file_content_from_s3
from llm_source_to_kg.config import config
import os

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
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "extract_cohort_prompt.txt")
    prompt = open(prompt_path, "r").read()

    messages = [
        LLMMessage(role="system", content=prompt),
        LLMMessage(role="user", content=state["source_contents"])
    ]

    response = await llm.chat_llm(messages, llm_config)

    # JSON 문자열 수정 후 Python 객체로 파싱
    repaired_json_str = repair_json(response.content)
    try:
        cohort_result = json.loads(repaired_json_str)
    except json.JSONDecodeError as e:
        doc_logger.error(f"JSON 파싱 오류: {e}")
        cohort_result = {"main_cohorts": []}

    doc_logger.info(f"{state['source_reference_number']} 코호트 추출 완료")

    state["cohorts_json"] = cohort_result
    return state


# 테스트용 코드
if __name__ == "__main__":

    state = CohortGraphState(
        document= get_file_content_from_s3(config.AWS_S3_BUCKET, f"nice/NG238.json"),
        source_reference_number="NG238",
        cohorts_json=[]
    )
    asyncio.run(extract_cohorts(state))
