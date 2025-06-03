from llm_source_to_kg.graph.cohort_graph.state import CohortGraphState
from typing import List, Dict, Any


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
        
        inclusion_criteria = sub_cohort.get('inclusion_criteria', [])
        if inclusion_criteria:
            markdown += "### Inclusion Criteria\n"
            for criterion in inclusion_criteria:
                markdown += f"- {criterion}\n"
            markdown += "\n"
        
        exclusion_criteria = sub_cohort.get('exclusion_criteria', [])
        if exclusion_criteria:
            markdown += "### Exclusion Criteria\n"
            for criterion in exclusion_criteria:
                markdown += f"- {criterion}\n"
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

