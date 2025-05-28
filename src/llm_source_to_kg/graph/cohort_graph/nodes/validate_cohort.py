from llm_source_to_kg.graph.cohort_graph.state import CohortGraphState
from llm_source_to_kg.utils.logger import get_logger

def validate_cohort(state: CohortGraphState) -> CohortGraphState:
    """
    코호트의 유효성을 검증하는 함수
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        CohortGraphState: 검증 결과가 포함된 상태
    """
    logger = get_logger()
    logger.info("Validating cohorts...")
    
    # 검증할 코호트 목록 가져오기
    cohorts = state.cohorts
    
    # 각 코호트에 대한 검증 수행
    validation_results = []
    for cohort in cohorts:
        result = {
            'cohort_id': cohort.id,
            'is_valid': True,
            'errors': [],
            'can_retry': True
        }
        
        # 필수 필드 검증
        required_fields = ['id', 'name', 'criteria', 'population']
        for field in required_fields:
            if not hasattr(cohort, field) or getattr(cohort, field) is None:
                result['is_valid'] = False
                result['errors'].append(f"Missing required field: {field}")
        
        # 데이터 타입 검증
        if not isinstance(cohort.criteria, dict):
            result['is_valid'] = False
            result['errors'].append("Criteria must be a dictionary")
            
        # 값의 범위/형식 검증
        if cohort.population < 0:
            result['is_valid'] = False
            result['errors'].append("Population cannot be negative")
            
        # 코호트 간 관계 검증
        if hasattr(cohort, 'parent_cohort_id'):
            if cohort.parent_cohort_id not in [c.id for c in cohorts]:
                result['is_valid'] = False
                result['errors'].append(f"Parent cohort {cohort.parent_cohort_id} not found")
        
        validation_results.append(result)
    
    # 검증 결과 상태 업데이트
    state.validation_results = validation_results
    
    # 재시도 가능 여부 확인
    state.can_retry = any(
        result['is_valid'] is False and result['can_retry'] 
        for result in validation_results
    )
    
    logger.info(f"Validation completed. Found {len(validation_results)} results.")
    return state