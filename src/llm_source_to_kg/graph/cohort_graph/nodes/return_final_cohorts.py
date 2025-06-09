from llm_source_to_kg.graph.cohort_graph.state import CohortGraphState
from typing import List, Dict, Any


def format_criteria_to_markdown(criteria: List[List[str]]) -> str:
    """
    새로운 중첩 배열 구조의 criteria를 마크다운 형식으로 변환합니다.
    
    Args:
        criteria: 변환할 criteria 배열 (배열의 배열 구조)
                 예: [["diabetes", "age>=18"], ["prediabetes", "high_risk"]]
                 의미: (diabetes AND age>=18) OR (prediabetes AND high_risk)
        
    Returns:
        마크다운 형식의 문자열
    """
    if not isinstance(criteria, list) or not criteria:
        return ""
    
    # 각 AND 그룹을 괄호로 묶어서 처리
    or_groups = []
    
    for and_group in criteria:
        if not isinstance(and_group, list) or not and_group:
            continue
        
        # 모든 조건 그룹을 괄호로 감싸기 (단일 조건 포함)
        quoted_conditions = [f'"{condition}"' for condition in and_group]
        and_conditions = " AND ".join(quoted_conditions)
        or_groups.append(f"({and_conditions})")
    
    if not or_groups:
        return ""
    
    # OR로 연결하여 최종 문자열 생성
    criteria_text = " OR ".join(or_groups)
    return f"- {criteria_text}\n"


def cohort_to_markdown(cohort: Dict[str, Any]) -> str:
    """
    코호트 데이터를 마크다운 형식으로 변환합니다.
    
    Args:
        cohort: 변환할 코호트 데이터
        
    Returns:
        마크다운 형식의 문자열
    """
    markdown = f"# Cohort: {cohort['subject']}\n"
    
    if 'details' in cohort and cohort['details']:
        markdown += f"{cohort['details']}\n\n"
    
    for i, sub_cohort in enumerate(cohort.get('sub_cohorts', []), 1):
        description = sub_cohort.get('description', {})
        subject = description.get('subject', '')
        details = description.get('details', '')
        
        markdown += f"## {i}. {subject}\n"
        if details:
            markdown += f"{details}\n\n"
        
        # 새로운 중첩 배열 구조의 inclusion_criteria 처리
        inclusion_criteria = sub_cohort.get('inclusion_criteria', [])
        if inclusion_criteria:
            markdown += "### Inclusion Criteria\n"
            criteria_text = format_criteria_to_markdown(inclusion_criteria)
            if criteria_text:
                markdown += criteria_text
            markdown += "\n"
        
        # 새로운 중첩 배열 구조의 exclusion_criteria 처리
        exclusion_criteria = sub_cohort.get('exclusion_criteria', [])
        if exclusion_criteria:
            markdown += "### Exclusion Criteria\n"
            criteria_text = format_criteria_to_markdown(exclusion_criteria)
            if criteria_text:
                markdown += criteria_text
            markdown += "\n"
        
        source_sentences = sub_cohort.get('source_sentences', [])
        if source_sentences:
            markdown += "### Reference Sentences\n"
            for sentence in source_sentences:
                markdown += f"- {sentence}\n"
            markdown += "\n"
    
    return markdown


def return_final_cohorts(state: CohortGraphState) -> CohortGraphState:
    """
    코호트 결과를 마크다운 형식으로 변환하여 상태에 저장합니다.
    
    Args:
        state: 그래프 상태
        
    Returns:
        업데이트된 그래프 상태
    """
    cohorts_json = state['cohorts_json']
    main_cohorts = cohorts_json if isinstance(cohorts_json, list) else []
    
    markdown_cohorts = [cohort_to_markdown(cohort) for cohort in main_cohorts]
    
    state['cohorts_markdown'] = markdown_cohorts
    
    return state

