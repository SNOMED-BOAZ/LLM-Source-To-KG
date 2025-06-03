from src.llm_source_to_kg.graph.analysis_graph.state import AnalysisGraphState
from src.llm_source_to_kg.schema.state import (
    AnalysisSchema, DrugSchema, DiagnosticSchema, 
    MedicalTestSchema, SurgerySchema, OMOPSchema
)
from llm_source_to_kg.utils.llm_util import get_llm
from llm_source_to_kg.schema.llm import LLMMessage, LLMConfig, LLMRole
from llm_source_to_kg.utils.logger import get_logger
import json
import asyncio
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import re
from json_repair import repair_json
from pydantic import BaseModel, ValidationError

# 전역 로거 사용
logger = get_logger("analysis_graph")

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
        logger.error(f"JSON 정제 중 오류 발생: {str(e)}")
        logger.error(f"원본 응답: {response[:200]}...")
        logger.error(f"정제 시도 후 응답: {json_str[:200]}...")
        raise

def convert_to_schema(entity: Dict[str, Any], entity_type: str) -> BaseModel:
    """엔티티를 해당하는 스키마 모델로 변환합니다."""
    try:
        if entity_type == "drug":
            return DrugSchema(**entity)
        elif entity_type == "diagnostic":
            return DiagnosticSchema(**entity)
        elif entity_type == "test":
            return MedicalTestSchema(**entity)
        elif entity_type == "surgery":
            return SurgerySchema(**entity)
        else:
            raise ValueError(f"Unknown entity type: {entity_type}")
    except ValidationError as e:
        logger.error(f"Schema validation error for {entity_type}: {str(e)}")
        raise

def validate_entity_structure(entity: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
    """엔티티 구조를 검증하고 정제합니다."""
    logger.debug(f"Validating {entity_type} entity structure")
    
    if entity_type == "drug":
        required_fields = ["concept_name", "concept_id", "domain_id", "vocabulary_id", 
                         "concept_code", "standard_concept", "mapping_confidence", "drug_name"]
        for field in required_fields:
            if field not in entity:
                if field == "mapping_confidence":
                    entity[field] = 0.0
                    logger.debug(f"Added default mapping_confidence for drug entity")
                elif field == "drug_name":
                    entity[field] = None
                    logger.debug(f"Added default drug_name for drug entity")
                else:
                    entity[field] = ""
                    logger.debug(f"Added empty {field} for drug entity")
        
        # DrugSchema로 변환 시도
        try:
            drug_schema = convert_to_schema(entity, "drug")
            return drug_schema
        except ValidationError as e:
            logger.error(f"Drug schema validation failed: {str(e)}")
            raise
    
    elif entity_type == "diagnostic":
        required_fields = ["concept_name", "concept_id", "domain_id", "vocabulary_id", 
                         "concept_code", "standard_concept", "mapping_confidence", 
                         "icd_code", "snomed_code"]
        for field in required_fields:
            if field not in entity:
                if field == "mapping_confidence":
                    entity[field] = 0.0
                    logger.debug(f"Added default mapping_confidence for diagnostic entity")
                elif field in ["icd_code", "snomed_code"]:
                    entity[field] = None
                    logger.debug(f"Added default {field} for diagnostic entity")
                else:
                    entity[field] = ""
                    logger.debug(f"Added empty {field} for diagnostic entity")
        
        # DiagnosticSchema로 변환 시도
        try:
            diagnostic_schema = convert_to_schema(entity, "diagnostic")
            return diagnostic_schema
        except ValidationError as e:
            logger.error(f"Diagnostic schema validation failed: {str(e)}")
            raise
    
    elif entity_type == "test":
        required_fields = ["concept_name", "concept_id", "domain_id", "vocabulary_id", 
                         "concept_code", "standard_concept", "mapping_confidence",
                         "test_name", "operator", "value", "unit"]
        for field in required_fields:
            if field not in entity:
                if field == "mapping_confidence":
                    entity[field] = 0.0
                    logger.debug(f"Added default mapping_confidence for test entity")
                elif field == "value":
                    entity[field] = 0.0
                    logger.debug(f"Added default value for test entity")
                elif field == "operator":
                    entity[field] = "="
                    logger.debug(f"Added default operator for test entity")
                else:
                    entity[field] = ""
                    logger.debug(f"Added empty {field} for test entity")
        
        # MedicalTestSchema로 변환 시도
        try:
            test_schema = convert_to_schema(entity, "test")
            return test_schema
        except ValidationError as e:
            logger.error(f"Test schema validation failed: {str(e)}")
            raise
    
    elif entity_type == "surgery":
        required_fields = ["concept_name", "concept_id", "domain_id", "vocabulary_id", 
                         "concept_code", "standard_concept", "mapping_confidence", "surgery_name"]
        for field in required_fields:
            if field not in entity:
                if field == "mapping_confidence":
                    entity[field] = 0.0
                    logger.debug(f"Added default mapping_confidence for surgery entity")
                else:
                    entity[field] = ""
                    logger.debug(f"Added empty {field} for surgery entity")
        
        # SurgerySchema로 변환 시도
        try:
            surgery_schema = convert_to_schema(entity, "surgery")
            return surgery_schema
        except ValidationError as e:
            logger.error(f"Surgery schema validation failed: {str(e)}")
            raise
    
    logger.debug(f"Entity validation completed for {entity_type}")
    return entity

def validate_analysis_schema(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """AnalysisSchema에 맞게 분석 결과를 검증하고 정제합니다."""
    logger.info("Starting analysis schema validation")
    validated = {}
    
    # 각 엔티티 타입별 검증 및 스키마 변환
    for entity_type in ["drug", "diagnostic", "test", "surgery"]:
        if entity_type in analysis and analysis[entity_type]:
            logger.debug(f"Validating {entity_type} entity")
            try:
                # 엔티티가 리스트인 경우 첫 번째 항목만 사용
                entity = analysis[entity_type]
                if isinstance(entity, list):
                    if not entity:
                        logger.warning(f"Empty list for {entity_type} entity")
                        validated[entity_type] = None
                        continue
                    entity = entity[0]
                    logger.debug(f"Using first item from {entity_type} entity list")
                
                validated[entity_type] = validate_entity_structure(entity, entity_type)
            except ValidationError as e:
                logger.error(f"Validation failed for {entity_type}: {str(e)}")
                validated[entity_type] = None
        else:
            logger.debug(f"No {entity_type} entity found, setting to None")
            validated[entity_type] = None
    
    # 추가 필드 검증
    for field in ["etc", "temporal_relation", "source_text_span"]:
        if field in analysis:
            validated[field] = analysis[field]
            logger.debug(f"Added {field} to validated analysis")
    
    # AnalysisSchema로 변환 시도
    try:
        analysis_schema = AnalysisSchema(**validated)
        logger.info("Analysis schema validation completed successfully")
        return validated
    except ValidationError as e:
        logger.error(f"Analysis schema validation failed: {str(e)}")
        raise

async def analyze_cohort(state: AnalysisGraphState) -> AnalysisGraphState:
    """
    코호트 데이터를 분석하여 medical entities를 추출합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 그래프 상태 (analysis 필드 포함)
    """
    logger.info("Starting cohort analysis")
    cohort_markdown = state["cohort"]
    logger.debug(f"Processing cohort markdown (length: {len(cohort_markdown) if cohort_markdown else 0})")
    
    if not cohort_markdown:
        logger.error("No cohort data provided")
        state["answer"] = "Error: No cohort data provided"
        state["is_valid"] = False
        state["validation_feedback"] = "No cohort data available for analysis"
        return state
    
    # LLM 클라이언트 초기화
    logger.info("Initializing LLM client")
    llm = get_llm(llm_type="gemini", model="gemini-2.0-flash")
    llm_config = LLMConfig(
        temperature=0.1,
        top_p=0.95,
        max_output_tokens=8192
    )
    logger.debug("LLM client initialized")
    
    # 프롬프트 템플릿 로드
    logger.info("Loading prompt template")
    try:
        prompt_template = load_prompt("extract_analysis")
        logger.debug("Prompt template loaded successfully")
    except Exception as e:
        logger.error(f"Error loading prompt template: {str(e)}")
        state["answer"] = f"Error loading prompt template: {str(e)}"
        state["is_valid"] = False
        state["validation_feedback"] = f"Failed to load prompt template: {str(e)}"
        return state
    
    logger.debug("Formatting cohort content")
    
    # 프롬프트에 마크다운 내용 직접 포함
    formatted_prompt = prompt_template.replace("{manager_response}", cohort_markdown)
    prompt = f"""Context: {state['context']}
Question: {state['question']}

{formatted_prompt}

중요: 반드시 AnalysisSchema 형식에 맞는 JSON으로만 응답해주세요. 다른 설명이나 텍스트는 포함하지 마세요."""

    logger.info("Sending prompt to LLM")
    
    try:
        # LLM 호출을 위한 메시지 구성
        logger.debug("Constructing LLM messages")
        messages = [
            LLMMessage(
                role=LLMRole.SYSTEM, 
                content="You are a medical entity extraction assistant. Your task is to analyze cohort data and extract medical entities according to the AnalysisSchema format. You must respond ONLY with a valid JSON object matching the schema, no other text or explanation."
            ),
            LLMMessage(role=LLMRole.USER, content=prompt)
        ]
        
        # LLM 호출
        logger.info("Calling LLM")
        response = await llm.chat_llm(messages, llm_config)
        logger.debug(f"Raw LLM response: {response.content[:200]}...")
        
        # JSON 파싱 시도
        if isinstance(response.content, str):
            logger.debug("Cleaning LLM response")
            # LLM 응답 정제
            cleaned_response = clean_llm_response(response.content)
            logger.debug(f"Cleaned response: {cleaned_response[:200]}...")
            
            try:
                logger.debug("Attempting JSON parsing")
                # 일반 JSON 파싱 시도
                extracted_entities = json.loads(cleaned_response)
                logger.debug("JSON parsing successful")
            except json.JSONDecodeError:
                logger.warning("JSON parsing failed, attempting repair")
                # 실패하면 json_repair 사용
                repaired_json = repair_json(cleaned_response)
                extracted_entities = json.loads(repaired_json)
                logger.debug("JSON repair successful")
        else:
            logger.debug("Response is not a string, using as is")
            extracted_entities = response.content
            
        # AnalysisSchema 기반 검증 및 변환
        logger.info("Validating extracted entities against AnalysisSchema")
        try:
            validated_analysis = validate_analysis_schema(extracted_entities)
            logger.debug("Entity validation completed")
            
            # 상태 업데이트
            state["analysis"] = validated_analysis
            state["is_valid"] = True
            state["answer"] = "Analysis completed successfully"
            state["validation_feedback"] = "Analysis completed and validated successfully"
            
            logger.info("Cohort analysis completed successfully")
            return state
            
        except ValidationError as e:
            logger.error(f"Schema validation failed: {str(e)}")
            state["answer"] = f"Schema validation failed: {str(e)}"
            state["is_valid"] = False
            state["validation_feedback"] = f"Analysis failed due to schema validation: {str(e)}"
            return state
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        state["answer"] = f"Error during analysis: {str(e)}"
        state["is_valid"] = False
        state["validation_feedback"] = f"Analysis failed: {str(e)}"
        return state