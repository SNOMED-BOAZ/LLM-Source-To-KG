from llm_source_to_kg.graph.cohort_graph.state import CohortGraphState
from llm_source_to_kg.utils.logger import get_logger
from llm_source_to_kg.utils.llm_util import get_llm

def retry_extract_cohort(state: CohortGraphState) -> CohortGraphState:
    """
    유효하지 않은 코호트에 대해 재시도하는 함수
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        CohortGraphState: 재시도 결과가 포함된 상태
    """
    logger = get_logger()
    logger.info("Retrying extraction for invalid cohorts...")
    
    # 검증 결과에서 재시도가 필요한 코호트 찾기
    invalid_cohorts = [
        result for result in state.validation_results
        if not result['is_valid'] and result['can_retry']
    ]
    
    if not invalid_cohorts:
        logger.info("No cohorts need retry")
        return state
    
    # LLM 인스턴스 생성
    llm = get_llm("gemini")
    
    # 각 유효하지 않은 코호트에 대해 재시도
    for result in invalid_cohorts:
        cohort_id = result['cohort_id']
        logger.info(f"Retrying extraction for cohort: {cohort_id}")
        
        try:
            # LLM을 사용하여 코호트 정보 재추출
            prompt = f"""
            다음 코호트 정보를 다시 추출해주세요:
            ID: {cohort_id}
            원본 내용: {state.source_contents}
            
            이전 오류:
            {', '.join(result['errors'])}
            """
            
            response = llm.generate(prompt)
            
            # 응답을 파싱하여 코호트 정보 업데이트
            # TODO: 실제 구현에서는 LLM 응답을 적절히 파싱하여 코호트 정보 업데이트
            # 현재는 예시로 간단히 처리
            
            # 상태 업데이트
            state.cohorts = [
                cohort for cohort in state.cohorts
                if cohort.id != cohort_id
            ]
            
            # 새로운 코호트 정보 추가 (실제 구현에서는 파싱된 정보 사용)
            # state.cohorts.append(new_cohort)
            
            logger.info(f"Successfully retried extraction for cohort: {cohort_id}")
            
        except Exception as e:
            logger.error(f"Failed to retry extraction for cohort {cohort_id}: {str(e)}")
            result['can_retry'] = False
    
    logger.info("Retry extraction completed")
    return state