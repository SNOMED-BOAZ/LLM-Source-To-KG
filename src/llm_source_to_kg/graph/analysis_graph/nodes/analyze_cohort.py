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
    cohort_markdown = state["cohort"]
    print(f"Processing single cohort markdown (length: {len(cohort_markdown) if cohort_markdown else 0})")
    
    if not cohort_markdown:
        print("Error: No cohort data provided")
        state["answer"] = "Error: No cohort data provided"
        state["is_valid"] = False
        return state
    
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
    try:
        prompt_template = load_prompt("extract_analysis")
        print("Prompt template loaded")
    except Exception as e:
        print(f"Error loading prompt template: {str(e)}")
        state["answer"] = f"Error loading prompt template: {str(e)}"
        state["is_valid"] = False
        return state
    
    print("Formatting cohort content...")
    
    # 프롬프트에 마크다운 내용 직접 포함
    formatted_prompt = prompt_template.replace("{manager_response}", cohort_markdown)
    prompt = f"""Context: {state['context']}
Question: {state['question']}

{formatted_prompt}

중요: 반드시 JSON 형식으로만 응답해주세요. 다른 설명이나 텍스트는 포함하지 마세요."""

    print("Sending prompt to LLM...")
    
    try:
        # LLM 호출을 위한 메시지 구성
        print("Constructing LLM messages...")
        messages = [
            LLMMessage(
                role=LLMRole.SYSTEM, 
                content="You are a medical entity extraction assistant. Your task is to analyze cohort data and extract medical entities, relationships, and diagnostic pathways in JSON format. You must respond ONLY with a valid JSON object, no other text or explanation."
            ),
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
        
        # 분석 결과 구성
        analysis_result = {
            "single_cohort": {
                "entities": validated_entities,
                "status": "success",
                "cohort_content": cohort_markdown,  # 처음 1000자만 저장
                "raw_response": response.content[:500]  # 원본 응답 일부 저장
            }
        }
        
        # 딕셔너리 형태로 분석 결과 구성
        print("Creating analysis result dictionary...")
        analysis_dict = {
            "cohort_analyses": analysis_result,
            "summary": {
                "total_cohorts": 1,
                "successful_analyses": 1,
                "failed_analyses": 0,
                "fallback_analyses": 0
            }
        }
        
        # 상태 업데이트
        state["analysis"] = analysis_dict
        state["answer"] = response.content[:500]  # LLM 응답 저장
        state["is_valid"] = True
        print("Analysis completed successfully")
        
    except Exception as e:
        print(f"분석 오류: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        # 에러 발생시 상태만 업데이트하고 analysis는 None으로 유지
        state["answer"] = f"Error: {str(e)}"
        state["is_valid"] = False
    
    print("=== Completed analyze_cohort ===")
    return state