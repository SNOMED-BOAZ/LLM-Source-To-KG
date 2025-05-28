from src.llm_source_to_kg.graph.analysis_graph.state import AnalysisGraphState
from llm_source_to_kg.utils.llm_util import get_llm
from llm_source_to_kg.schema.llm import LLMMessage, LLMConfig, LLMRole
import json
from typing import Dict, Any
import asyncio
from pathlib import Path
import re

def load_prompt(template_name: str) -> str:
    """프롬프트 템플릿 파일을 로드합니다."""
    current_dir = Path(__file__).parent
    prompt_path = current_dir / "prompts" / f"{template_name}.txt"
    
    with open(prompt_path, 'r', encoding='utf-8') as file:
        return file.read()

def clean_llm_response(response: str) -> str:
    """LLM 응답에서 JSON 부분만 추출하고 정제합니다."""
    try:
        # 1. 코드 블록에서 JSON 추출 시도
        json_pattern = r"```(?:json)?\s*({[\s\S]*?})\s*```"
        match = re.search(json_pattern, response)
        
        if match:
            json_str = match.group(1)
        else:
            # 2. 코드 블록이 없는 경우, JSON 객체 패턴으로 찾기
            json_pattern = r"({[\s\S]*})"
            match = re.search(json_pattern, response)
            if match:
                json_str = match.group(1)
            else:
                # 3. 마지막 시도: 전체 응답을 JSON으로 간주
                json_str = response.strip()
        
        # 4. JSON 문자열 정제
        # 불필요한 공백 제거
        json_str = re.sub(r'\s+', ' ', json_str)
        # 줄바꿈 문자 제거
        json_str = json_str.replace('\n', ' ').replace('\r', '')
        # 따옴표 정규화
        json_str = re.sub(r'["\']', '"', json_str)
        # 콤마 정규화
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        # 불필요한 공백 제거
        json_str = re.sub(r'\s*:\s*', ':', json_str)
        json_str = re.sub(r'\s*,\s*', ',', json_str)
        
        return json_str.strip()
        
    except Exception as e:
        print(f"JSON 정제 중 오류 발생: {str(e)}")
        print(f"원본 응답: {response[:200]}...")
        raise

async def validate_analysis(state: AnalysisGraphState) -> AnalysisGraphState:
    """
    LLM을 사용하여 분석 결과가 원본 코호트 내용과 일치하는지 검증합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 그래프 상태 (is_valid 필드 포함)
    """
    print("\n=== Starting validate_analysis ===")
    analysis = state["analysis"]
    cohort_data = state["cohort"]
    
    # LLM 클라이언트 초기화
    print("Initializing LLM client...")
    llm = get_llm(llm_type="gemini", model="gemini-2.0-flash")
    llm_config = LLMConfig(
        temperature=0.1,
        top_p=0.95,
        max_output_tokens=8192
    )
    print("LLM client initialized")
    
    # 프롬프트 템플릿 로드
    print("Loading prompt template...")
    prompt_template = load_prompt("validate_analysis")
    print("Prompt template loaded")
    
    validation_results = {}
    total_valid = 0
    total_cohorts = 0
    
    # main_cohorts 단위로 검증
    for main_cohort in cohort_data.get("main_cohorts", []):
        main_subject = main_cohort["subject"]
        print(f"\nValidating main cohort: {main_subject}")
        total_cohorts += 1
        
        # 해당 main_cohort에 대한 분석 결과 찾기
        cohort_analysis = None
        for cohort_id, analysis_result in analysis.items():
            if main_subject in cohort_id:
                cohort_analysis = analysis_result
                break
        
        if not cohort_analysis:
            validation_results[main_subject] = {
                "is_valid": False,
                "reason": f"분석 결과를 찾을 수 없음: {main_subject}"
            }
            continue
        
        if cohort_analysis.get("status") != "success":
            validation_results[main_subject] = {
                "is_valid": False,
                "reason": f"분석 실패: {cohort_analysis.get('error', 'Unknown error')}"
            }
            continue
        
        # 프롬프트 구성
        formatted_prompt = prompt_template.replace(
            "{analysis_result}", 
            json.dumps(cohort_analysis.get("entities", {}), ensure_ascii=False, indent=2)
        ).replace(
            "{cohort_data}", 
            json.dumps(main_cohort, ensure_ascii=False, indent=2)
        )
        
        try:
            # LLM 호출을 위한 메시지 구성
            messages = [
                LLMMessage(role=LLMRole.SYSTEM, content="You are a medical data validation expert. Your task is to validate if the analysis results match the original cohort data. You must respond ONLY with a valid JSON object, no other text or explanation."),
                LLMMessage(role=LLMRole.USER, content=formatted_prompt)
            ]
            
            # LLM 호출
            print("Calling LLM for validation...")
            response = await llm.chat_llm(messages, llm_config)
            print("LLM response received")
            
            # LLM 응답 정제
            print("Cleaning LLM response...")
            cleaned_response = clean_llm_response(response.content)
            print(f"Cleaned response: {cleaned_response[:200]}...")
            
            # 검증 결과 파싱
            print("Parsing validation result...")
            validation_result = json.loads(cleaned_response)
            validation_results[main_subject] = validation_result
            
            if validation_result["is_valid"]:
                total_valid += 1
                
        except Exception as e:
            print(f"검증 중 오류 발생 (main_cohort: {main_subject}): {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            validation_results[main_subject] = {
                "is_valid": False,
                "reason": f"검증 오류: {str(e)}"
            }
    
    # 전체 검증 결과 결정
    overall_valid = (total_valid == total_cohorts) and total_cohorts > 0
    
    # 상태 업데이트
    state["is_valid"] = overall_valid
    state["validation_results"] = validation_results
    
    # 실패한 경우 재시도 카운트 증가
    if not overall_valid:
        state["retries"] = state.get("retries", 0) + 1
    
    print("=== Completed validate_analysis ===")
    return state

async def test_validation(analysis_file: str, cohort_file: str) -> None:
    """
    검증 함수를 테스트하기 위한 메인 함수입니다.
    
    Args:
        analysis_file: 분석 결과 JSON 파일 경로
        cohort_file: 코호트 데이터 JSON 파일 경로
    """
    print("\n=== Starting validation test ===")
    
    try:
        # 파일 로드
        print(f"Loading analysis file: {analysis_file}")
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
            
        print(f"Loading cohort file: {cohort_file}")
        with open(cohort_file, 'r', encoding='utf-8') as f:
            cohort_data = json.load(f)
            
        # 초기 상태 설정
        initial_state = AnalysisGraphState(
            context="Validating analysis results against original cohort data",
            question="Do the analysis results match the original cohort data?",
            answer="",  # LLM이 채울 응답
            source_reference_number="TEST",
            is_valid=True,
            retries=0,
            cohort=cohort_data,
            analysis=analysis_data
        )
        
        # 검증 수행
        print("\nPerforming validation...")
        result_state = await validate_analysis(initial_state)
        
        # 결과 출력
        print("\n=== Validation Results ===")
        print(f"Overall validation status: {'Valid' if result_state['is_valid'] else 'Invalid'}")
        print("\nDetailed results:")
        
        for cohort_id, validation_result in result_state["validation_results"].items():
            print(f"\nCohort: {cohort_id}")
            print(f"Status: {'Valid' if validation_result['is_valid'] else 'Invalid'}")
            
            if not validation_result['is_valid']:
                print("Validation details:")
                print(json.dumps(validation_result, indent=2, ensure_ascii=False))
        
        # 결과 저장
        output_file = analysis_file.replace('.json', '_validation_results.json')
        print(f"\nSaving validation results to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_state["validation_results"], f, indent=2, ensure_ascii=False)
            
        print("\n=== Validation test completed ===")
        
    except Exception as e:
        print(f"\nError during validation test: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    import os
    import sys
    
    # 프로젝트 루트 디렉토리 설정
    PROJECT_ROOT = "/Users/hyejiyu/Desktop/2025/BOAZ_ADV/langGraph/BOAZ-SNUH_llm_source_to_kg"
    
    # 테스트할 파일 경로
    analysis_file = os.path.join(PROJECT_ROOT, "datasets/results/analysis/NG238_analysis_results.json")
    cohort_file = os.path.join(PROJECT_ROOT, "datasets/results/cohorts/NG238_cohorts.json")
    
    # 파일 존재 확인
    if not os.path.exists(analysis_file):
        print(f"Error: Analysis file not found: {analysis_file}")
        sys.exit(1)
    if not os.path.exists(cohort_file):
        print(f"Error: Cohort file not found: {cohort_file}")
        sys.exit(1)
    
    # 비동기 메인 함수 실행
    asyncio.run(test_validation(analysis_file, cohort_file))