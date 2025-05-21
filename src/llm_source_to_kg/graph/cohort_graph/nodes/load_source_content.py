from llm_source_to_kg.utils.logger import get_logger
from llm_source_to_kg.utils.s3 import get_file_content_from_s3
from llm_source_to_kg.config import config
from llm_source_to_kg.graph.cohort_graph.state import CohortGraphState
import json

def load_source_content(state: CohortGraphState) -> CohortGraphState:
    """
    소스 콘텐츠를 로드하고 state를 업데이트합니다.
    
    Args:
        state: 현재 상태
        source_id: 소스 ID (예: "NG238")
        
    Returns:
        업데이트된 state
    """
    source_id = state["source_reference_number"]
    
    logger = get_logger()
    logger.info(f"Loading source content: {source_id}")
    
    # 소스 콘텐츠 로드
    source_content_json = get_file_content_from_s3(config.AWS_S3_BUCKET, f"nice/{source_id}.json")

    
    # state 업데이트
    state["source_reference_number"] = source_id
    state["source_contents"] = source_content_json
    
    logger.info(f"Source content successfully loaded for: {source_id}")
    return state


# 테스트용 코드
if __name__ == "__main__":
    # 테스트용 초기 state 생성
    test_state: CohortGraphState = {
        "context": "",
        "question": "",
        "answer": "",
        "is_valid": False,
        "retries": 0,
        "source_reference_number": "NG238",
        "source_contents": "",
        "cohort_result": []
    }
    
    # 함수 테스트
    updated_state = load_source_content(test_state)
    print(updated_state['source_contents'])
    print(f"Source Reference Number: {updated_state['source_reference_number']}")
    print(f"Source Contents Length: {len(updated_state['source_contents'])}")