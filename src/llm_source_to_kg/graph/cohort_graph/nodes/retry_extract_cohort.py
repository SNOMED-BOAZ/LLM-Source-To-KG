import asyncio
import json
from llm_source_to_kg.graph.cohort_graph.state import CohortGraphState
from llm_source_to_kg.utils.llm_util import get_llm
from llm_source_to_kg.utils.logger import get_logger
from llm_source_to_kg.schema.llm import LLMMessage, LLMConfig
from llm_source_to_kg.schema.validate.cohorts import validate_cohort_schema
from json_repair import repair_json
import os


# TODO
"""
validate_cohort.py에서 추가된 state["validation_feedback"] 사용해서 프롬프트 고도화
state["retries"]는 외부에서 관리되므로 여기서 처리할 필요 X

validate_cohort.py와 검증 후 처리와 state에 반영해야 할 것은 동일
"""

async def retry_extract_cohort(state: CohortGraphState) -> CohortGraphState:
    """
    유효하지 않은 코호트에 대해 재시도하는 함수
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        CohortGraphState: 재시도 결과가 포함된 상태
    """
    logger = get_logger(name=state["source_reference_number"])
    logger.info("Retrying cohort extraction...")
    
    # 재시도 횟수 증가
    state["retries"] += 1
    
    # 최대 재시도 횟수 체크 (3회)
    if state["retries"] > 3:
        logger.warning(f"Maximum retry attempts reached for {state['source_reference_number']}")
        state["is_valid"] = False
        return state
    
    try:
        # LLM 인스턴스 생성
        llm = get_llm(llm_type="gemini", model="gemini-2.0-flash")
        
        llm_config = LLMConfig(
            temperature=0.1,  # 재시도 시 더 보수적인 온도 설정
            top_p=0.9,
            max_output_tokens=8192
        )
        
        # 프롬프트 로드
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "extract_cohort_prompt.txt")
        base_prompt = open(prompt_path, "r").read()
        
        # 재시도용 추가 지침
        retry_instruction = """
        
        IMPORTANT: This is a retry attempt. Please pay extra attention to:
        1. Ensure all required fields are properly formatted
        2. Double-check the JSON structure
        3. Verify that all cohort descriptions are clear and complete
        4. Make sure inclusion/exclusion criteria are specific and actionable
        """
        
        enhanced_prompt = base_prompt + retry_instruction
        
        messages = [
            LLMMessage(role="system", content=enhanced_prompt),
            LLMMessage(role="user", content=state["source_contents"])
        ]
        
        response = await llm.chat_llm(messages, llm_config)
        
        # JSON 문자열 수정 후 Python 객체로 파싱
        repaired_json_str = repair_json(response.content)
        try:
            cohort_result = json.loads(repaired_json_str)
            
            # 스키마 검증 수행
            is_valid, schema_errors = validate_cohort_schema(cohort_result)
            
            if not is_valid:
                logger.warning(f"Schema validation failed during retry: {schema_errors}")
                cohort_result = {"main_cohorts": []}
            else:
                logger.info("Schema validation successful during retry")
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error during retry: {e}")
            cohort_result = {"main_cohorts": []}
        
        # 결과 업데이트
        state["cohorts_json"] = cohort_result
        state["answer"] = response.content
        
        logger.info(f"Retry attempt {state['retries']} completed for {state['source_reference_number']}")
        
    except Exception as e:
        logger.error(f"Error during retry extraction: {str(e)}")
        state["cohorts_json"] = {"main_cohorts": []}
        state["is_valid"] = False
    
    return state


# 동기 래퍼 함수 (그래프에서 호출용)
def retry_extract_cohort_sync(state: CohortGraphState) -> CohortGraphState:
    """
    retry_extract_cohort의 동기 래퍼 함수
    """
    return asyncio.run(retry_extract_cohort(state))