"""
코호트 그래프에서 사용되는 유틸리티 함수
"""
from typing import Literal
from llm_source_to_kg.graph.cohort_graph.state import CohortGraphState

# 최대 재시도 횟수
MAX_RETRIES = 3


def route_after_validation(state: CohortGraphState) -> Literal["retry_extract_cohort", "return_final_cohorts"]:
    """
    검증 결과에 따라 다음 노드를 결정하는 라우팅 함수
    
    검증 로직:
    1. state["is_valid"]가 True이면 return_final_cohorts로 진행
    2. state["is_valid"]가 False이면:
       - state["retry_cohorts"] 길이가 0보다 크면 retry_extract_cohort 진행
       - state["retry_cohorts"] 길이가 0이면서 state["cohorts_json"] 길이가 0보다 크면 return_final_cohorts로 진행
       - state["retries"]가 3 초과이면 재시도 종료하고 return_final_cohorts 진행
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        다음에 실행할 노드 이름
    """
    # 검증이 성공한 경우
    if state.get("is_valid", False):
        return "return_final_cohorts"
    
    # 검증이 실패한 경우
    # 재시도 횟수가 최대치를 초과한 경우 종료
    if state.get("retries", 0) > MAX_RETRIES:
        return "return_final_cohorts"
    
    # 재시도할 코호트가 있는 경우
    retry_cohorts = state.get("retry_cohorts", [])
    if len(retry_cohorts) > 0:
        return "retry_extract_cohort"
    
    # 재시도할 코호트는 없지만 유효한 코호트가 있는 경우
    cohorts_json = state.get("cohorts_json", [])
    if len(cohorts_json) > 0:
        return "return_final_cohorts"
    
    # 기본적으로 최종 결과 반환 (빈 결과라도)
    return "return_final_cohorts" 