from src.llm_source_to_kg.graph.analysis_graph.state import AnalysisGraphState
from src.llm_source_to_kg.schema.state import AnalysisSchema
from llm_source_to_kg.utils.llm_util import get_llm
from llm_source_to_kg.schema.llm import LLMMessage, LLMConfig, LLMRole
import json
import asyncio
import os
from pathlib import Path
from typing import Dict, Any
import re
from json_repair import repair_json

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
        
        # 5. JSON 복구 시도
        try:
            # 먼저 일반 JSON 파싱 시도
            json.loads(json_str)
        except json.JSONDecodeError:
            # 실패하면 json_repair 사용
            json_str = repair_json(json_str)
        
        return json_str.strip()
        
    except Exception as e:
        print(f"JSON 정제 중 오류 발생: {str(e)}")
        print(f"원본 응답: {response[:200]}...")
        print(f"정제 시도 후 응답: {json_str[:200]}...")
        raise

def validate_entity_structure(entity: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
    """엔티티 구조를 검증하고 정제합니다."""
    if entity_type == "condition_entities":
        required_fields = ["concept_name", "condition_category", "severity", "staging", 
                         "risk_factors", "complications", "evidence_level", "source_text"]
        for field in required_fields:
            if field not in entity:
                if field == "staging":
                    entity[field] = {"system": "", "stage_value": "", "criteria": ""}
                elif field in ["risk_factors", "complications"]:
                    entity[field] = []
                else:
                    entity[field] = ""
    
    elif entity_type == "condition_relationships":
        required_fields = ["source_condition", "target_entity", "relationship_type", 
                         "details", "certainty", "evidence"]
        for field in required_fields:
            if field not in entity:
                entity[field] = ""
    
    elif entity_type == "diagnostic_pathways":
        required_fields = ["name", "description", "steps", "evidence_level"]
        for field in required_fields:
            if field not in entity:
                if field == "steps":
                    entity[field] = []
                else:
                    entity[field] = ""
    
    elif entity_type == "condition_cohorts":
        required_fields = ["name", "description", "target_population", 
                         "inclusion_criteria", "exclusion_criteria", "condition_occurrences"]
        for field in required_fields:
            if field not in entity:
                if field in ["inclusion_criteria", "exclusion_criteria", "condition_occurrences"]:
                    entity[field] = []
                else:
                    entity[field] = ""
    
    return entity

def validate_extracted_entities(entities: Dict[str, Any]) -> Dict[str, list]:
    """추출된 엔티티를 검증하고 정리합니다."""
    expected_keys = ["condition_entities", "condition_relationships", 
                    "diagnostic_pathways", "condition_cohorts", "detailed_analysis"]
    validated = {}
    
    for key in expected_keys:
        if key == "detailed_analysis":
            validated[key] = entities.get(key, "")
            continue
            
        if key in entities and isinstance(entities[key], list):
            # 각 엔티티 검증 및 정제
            validated[key] = [
                validate_entity_structure(item, key)
                for item in entities[key]
                if item and isinstance(item, dict)
            ]
        else:
            validated[key] = []
    
    return validated

async def analyze_cohort(state: AnalysisGraphState) -> AnalysisGraphState:
    """
    코호트 데이터를 분석하여 medical entities를 추출합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 그래프 상태 (analysis 필드 포함)
    """
    print("\n=== Starting analyze_cohort ===")
    cohort_data = state["cohort"]
    print(f"Number of cohorts to process: {len(cohort_data)}")
    
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
    prompt_template = load_prompt("extract_analysis")
    print("Prompt template loaded")
    
    # 코호트별로 분석 수행
    analysis_results = {}
    
    for cohort_id, cohort_content in cohort_data.items():
        print(f"\nProcessing cohort: {cohort_id}")
        
        # 프롬프트 구성
        print("Formatting cohort content...")
        print(f"cohort_content: {cohort_content}")
        formatted_content = {
            "main_cohort": cohort_content["main_cohort"],
            "sub_cohorts": cohort_content["sub_cohorts"]
        }
        
        # 프롬프트에 컨텍스트와 질문 추가
        formatted_prompt = prompt_template.replace("{manager_response}", json.dumps(formatted_content, ensure_ascii=False, indent=2))
        prompt = f"""Context: {state['context']}
Question: {state['question']}

{formatted_prompt}

중요: 반드시 JSON 형식으로만 응답해주세요. 다른 설명이나 텍스트는 포함하지 마세요."""

        print("Sending prompt to LLM...")
        
        try:
            # LLM 호출을 위한 메시지 구성
            print("Constructing LLM messages...")
            messages = [
                LLMMessage(role=LLMRole.SYSTEM, content="You are a medical entity extraction assistant. Your task is to analyze cohort data and extract medical entities, relationships, and diagnostic pathways in JSON format. You must respond ONLY with a valid JSON object, no other text or explanation."),
                LLMMessage(role=LLMRole.USER, content=prompt)
            ]
            
            # LLM 호출
            print("Calling LLM...")
            response = await llm.chat_llm(messages, llm_config)
            print(f"Raw LLM response: {response.content[:200]}...")
            
            # JSON 파싱 시도
            if isinstance(response.content, str):
                print("Cleaning LLM response...")
                # LLM 응답 정제
                cleaned_response = clean_llm_response(response.content)
                print(f"Cleaned response: {cleaned_response[:200]}...")
                
                try:
                    print("Attempting JSON parsing...")
                    # 일반 JSON 파싱 시도
                    extracted_entities = json.loads(cleaned_response)
                    print("JSON parsing successful")
                except json.JSONDecodeError:
                    print("JSON parsing failed, attempting repair...")
                    # 실패하면 json_repair 사용
                    repaired_json = repair_json(cleaned_response)
                    extracted_entities = json.loads(repaired_json)
                    print("JSON repair successful")
            else:
                print("Response is not a string, using as is")
                extracted_entities = response.content
                
            # 엔티티 검증 및 정리
            print("Validating extracted entities...")
            validated_entities = validate_extracted_entities(extracted_entities)
            print("Entity validation completed")
                
            analysis_results[cohort_id] = {
                "entities": validated_entities,
                "status": "success",
                "cohort_content": cohort_content,
                "raw_response": response.content[:500]  # 원본 응답 일부 저장
            }
            
            # 상태 업데이트
            state["answer"] = response.content[:500]  # LLM 응답 저장
            print("Analysis completed successfully")
            
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류 (cohort_id: {cohort_id}): {str(e)}")
            print(f"원본 응답: {response.content if 'response' in locals() else 'No response'}")
            # JSON 파싱 실패시 더미 응답 사용
            print("Generating dummy entities...")
            dummy_entities = generate_dummy_entities(cohort_content)
            analysis_results[cohort_id] = {
                "entities": dummy_entities,
                "status": "error_fallback",
                "error": f"JSON 파싱 오류: {str(e)}",
                "cohort_content": cohort_content,
                "raw_response": response.content[:500] if 'response' in locals() else ""
            }
            state["answer"] = f"Error: {str(e)}"
            state["is_valid"] = False
            print("Fallback to dummy entities completed")
            
        except Exception as e:
            print(f"분석 오류 (cohort_id: {cohort_id}): {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            analysis_results[cohort_id] = {
                "entities": {},
                "status": "error", 
                "error": f"분석 오류: {str(e)}",
                "cohort_content": cohort_content
            }
            state["answer"] = f"Error: {str(e)}"
            state["is_valid"] = False
    
    # AnalysisSchema 형태로 변환
    print("\nConverting to AnalysisSchema...")
    analysis_schema = AnalysisSchema(
        cohort_analyses=analysis_results,
        summary={
            "total_cohorts": len(cohort_data),
            "successful_analyses": len([r for r in analysis_results.values() if r["status"] == "success"]),
            "failed_analyses": len([r for r in analysis_results.values() if r["status"] == "error"]),
            "fallback_analyses": len([r for r in analysis_results.values() if r["status"] == "error_fallback"])
        }
    )
    
    # 상태 업데이트
    state["analysis"] = analysis_schema
    state["retries"] = state.get("retries", 0)
    
    print("=== Completed analyze_cohort ===")
    return state

def generate_dummy_entities(cohort_content: str) -> Dict[str, list]:
    """코호트 내용 기반 더미 엔티티 생성 (LLM 실패시 폴백용)"""
    # 더미 데이터 생성
    dummy_entities = {
        "condition_entities": [
            {
                "concept_name": "Test Condition",
                "condition_category": "Test Category",
                "severity": "Mild",
                "staging": {
                    "system": "Test System",
                    "stage_value": "Stage 1",
                    "criteria": "Test Criteria"
                },
                "risk_factors": ["Test Risk Factor"],
                "complications": ["Test Complication"],
                "evidence_level": "Low",
                "source_text": cohort_content[:100]
            }
        ],
        "condition_relationships": [],
        "diagnostic_pathways": [],
        "condition_cohorts": [],
        "detailed_analysis": "Failed to generate detailed analysis."
    }
    
    return dummy_entities

async def process_cohorts(cohorts_file_path: str) -> Dict[str, Any]:
    """
    여러 코호트를 순차적으로 처리합니다.
    
    Args:
        cohorts_file_path: 코호트 JSON 파일 경로
        
    Returns:
        모든 코호트의 분석 결과
    """
    print("\n=== Starting process_cohorts ===")
    print(f"Loading cohorts from: {cohorts_file_path}")
    
    # JSON 파일 로드
    try:
        with open(cohorts_file_path, 'r', encoding='utf-8') as f:
            cohorts_data = json.load(f)
        print(f"Successfully loaded {len(cohorts_data.get('main_cohorts', []))} main cohorts")
    except Exception as e:
        print(f"Error loading cohorts file: {str(e)}")
        raise
    
    all_results = {}
    
    # 각 메인 코호트 처리
    for main_cohort in cohorts_data.get("main_cohorts", []):
        main_subject = main_cohort["subject"]
        print(f"\nProcessing main cohort: {main_subject}")
        
        # 모든 sub_cohorts의 정보를 통합
        integrated_content = {
            "main_cohort": {
                "subject": main_cohort["subject"],
                "details": main_cohort["details"]
            },
            "sub_cohorts": [
                {
                    "subject": sub_cohort["description"]["subject"],
                    "details": sub_cohort["description"]["details"],
                    "inclusion_criteria": sub_cohort["inclusion_criteria"],
                    "exclusion_criteria": sub_cohort["exclusion_criteria"],
                    "source_sentences": sub_cohort["source_sentences"]
                }
                for sub_cohort in main_cohort.get("sub_cohorts", [])
            ]
        }
        
        print("Setting up initial state...")
        # 초기 상태 설정
        initial_state = AnalysisGraphState(
            context="Analyzing medical entities and relationships from cohort data",
            question="Extract medical entities, relationships, and diagnostic pathways from the given cohort data",
            answer="",  # LLM이 채울 응답
            source_reference_number="NG238",
            is_valid=True,
            retries=0,
            cohort={main_subject: integrated_content},  # 통합된 내용을 main_subject를 키로 저장
            analysis=None  # LLM이 채울 분석 결과
        )
        
        try:
            print("Calling analyze_cohort...")
            # 코호트 분석 수행
            result_state = await analyze_cohort(initial_state)
            print("analyze_cohort completed successfully")
            
            # 분석 결과 저장
            if result_state["analysis"] and "cohort_analyses" in result_state["analysis"]:
                all_results[main_subject] = result_state["analysis"]["cohort_analyses"].get(main_subject)
            
        except Exception as e:
            print(f"Error processing main cohort {main_subject}: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            all_results[main_subject] = {
                "status": "error",
                "error": str(e)
            }
    
    print("\n=== Completed process_cohorts ===")
    return all_results

async def main():
    """메인 실행 함수"""
    print("\n=== Starting main function ===")

    PROJECT_ROOT = "/Users/hyejiyu/Desktop/2025/BOAZ_ADV/langGraph/BOAZ-SNUH_llm_source_to_kg"
    cohorts_file_path = os.path.join(PROJECT_ROOT, "datasets/results/cohorts/NG238_cohorts.json")
    print(f"Cohorts file path: {cohorts_file_path}")

    try:
        print("Starting process_cohorts...")
        # 모든 코호트 처리
        results = await process_cohorts(cohorts_file_path)
        print("process_cohorts completed")
        
        # 결과 저장
        output_path = os.path.join(PROJECT_ROOT, "datasets/results/analysis/NG238_analysis_results.json")
        print(f"Saving results to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print(f"Analysis completed. Results saved to {output_path}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    print("\n=== Script started ===")
    # 비동기 메인 함수 실행
    asyncio.run(main())
    print("\n=== Script completed ===")