from llm_source_to_kg.graph.cohort_graph.state import CohortGraphState
from llm_source_to_kg.utils.logger import get_logger

def validate_cohort(state: CohortGraphState) -> CohortGraphState:
    """
    코호트 검증 노드 (구현 예정)
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        CohortGraphState: 상태 그대로 반환
    """
    logger = get_logger(name=state["source_reference_number"])
    logger.info("Cohort validation node - implementation pending")
    
    # TODO: LLM 기반 코호트 검증 로직 구현 필요
    """
    스키마에 대한 검증은 extract_cohorts에서 이미 완료됨
    코호트 검증 로직 추가 필요
    코호트 형식
    ```json
        [
          {
            "subject": "Main cohort title",
            "details": "Expanded clinical theme description",
            "sub_cohorts": [
              {
                "description": {
                  "subject": "Sub-cohort title",
                  "details": "Detailed sub-group use case"
                },
                "inclusion_criteria": ["..."],  # Optional
                "exclusion_criteria": ["..."],  # Optional
                "source_sentences": ["Original sentence from the guideline...", "..."]  # Optional
              }
            ]
          }, 
          ...
        ]

    ```
    배열의 각 인덱스 별 항목이 하나의 cohort로서 검증의 단위가 된다고 보면 됨. sub_cohort는 main에 속한 소주제 같은 느낌
    반복문으로 각각의 cohort 별 검증 수행

    우선, 본격적인 검증 시작 전에 state["cohorts_json"]을 다른 변수에 저장해 두고 []로 비우기.

    =>  검증에서 전부 통과되면 state["is_valid"] = True

        검증에서 실패하면 state["is_valid"] = False, state["retries"] += 1
        검증에서 실패한 이유를 state["validation_feedback"]에 저장
        검증에서 실패한 main_cohort를 state["retry_cohort"]에 저장
        검증에서 통과한 main_cohort를 state["cohorts_json"]에 append
        (통과한 애들은 analysis graph로 넘어가고, 아닌 애들은 재시도 됨)

        return state를 통해 orchestrator 단에서 알아서 처리.

        

    """
    
    
    return state