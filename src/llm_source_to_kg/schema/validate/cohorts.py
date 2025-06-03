from typing import Dict, Any, List

def validate_cohort_schema(cohorts_json: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    추출된 코호트 JSON의 스키마를 검증합니다.
    
    Args:
        cohorts_json: 검증할 코호트 JSON 데이터
        
    Returns:
        tuple[bool, List[str]]: (검증 성공 여부, 오류 메시지 목록)
    """
    errors = []
    
    # main_cohorts 필드 존재 및 타입 검증
    if "main_cohorts" not in cohorts_json:
        errors.append("Missing required field: 'main_cohorts'")
        return False, errors
    
    main_cohorts = cohorts_json["main_cohorts"]
    if not isinstance(main_cohorts, list):
        errors.append("'main_cohorts' must be a list")
        return False, errors
    
    # 각 main_cohort 검증
    for i, cohort in enumerate(main_cohorts):
        if not isinstance(cohort, dict):
            errors.append(f"Main cohort {i} must be a dictionary")
            continue
            
        # 필수 필드: subject
        if "subject" not in cohort or not isinstance(cohort["subject"], str) or not cohort["subject"].strip():
            errors.append(f"Main cohort {i}: Missing or invalid 'subject' field")
        
        # 선택적 필드: details
        if "details" in cohort and not isinstance(cohort["details"], str):
            errors.append(f"Main cohort {i}: 'details' must be a string")
        
        # sub_cohorts 검증 (선택적)
        if "sub_cohorts" in cohort:
            sub_cohorts = cohort["sub_cohorts"]
            if not isinstance(sub_cohorts, list):
                errors.append(f"Main cohort {i}: 'sub_cohorts' must be a list")
            else:
                for j, sub_cohort in enumerate(sub_cohorts):
                    sub_errors = _validate_sub_cohort_schema(sub_cohort, i, j)
                    errors.extend(sub_errors)
    
    return len(errors) == 0, errors

def _validate_sub_cohort_schema(sub_cohort: Dict[str, Any], cohort_index: int, sub_index: int) -> List[str]:
    """
    서브 코호트의 스키마를 검증합니다.
    
    Args:
        sub_cohort: 검증할 서브 코호트 데이터
        cohort_index: 상위 코호트 인덱스
        sub_index: 서브 코호트 인덱스
        
    Returns:
        List[str]: 오류 메시지 목록
    """
    errors = []
    
    if not isinstance(sub_cohort, dict):
        errors.append(f"Cohort {cohort_index}, Sub-cohort {sub_index}: must be a dictionary")
        return errors
    
    # description 필드 검증 (필수)
    if "description" not in sub_cohort:
        errors.append(f"Cohort {cohort_index}, Sub-cohort {sub_index}: Missing required field 'description'")
    else:
        description = sub_cohort["description"]
        if not isinstance(description, dict):
            errors.append(f"Cohort {cohort_index}, Sub-cohort {sub_index}: 'description' must be a dictionary")
        else:
            # description.subject 필수
            if "subject" not in description or not isinstance(description["subject"], str) or not description["subject"].strip():
                errors.append(f"Cohort {cohort_index}, Sub-cohort {sub_index}: Missing or invalid 'subject' in description")
            
            # description.details 선택적
            if "details" in description and not isinstance(description["details"], str):
                errors.append(f"Cohort {cohort_index}, Sub-cohort {sub_index}: 'details' in description must be a string")
    
    # 선택적 필드들 검증
    optional_list_fields = ["inclusion_criteria", "exclusion_criteria", "source_sentences"]
    for field in optional_list_fields:
        if field in sub_cohort:
            if not isinstance(sub_cohort[field], list):
                errors.append(f"Cohort {cohort_index}, Sub-cohort {sub_index}: '{field}' must be a list")
            else:
                # 리스트 요소들이 문자열인지 검증
                for k, item in enumerate(sub_cohort[field]):
                    if not isinstance(item, str):
                        errors.append(f"Cohort {cohort_index}, Sub-cohort {sub_index}: '{field}[{k}]' must be a string")
    
    return errors 