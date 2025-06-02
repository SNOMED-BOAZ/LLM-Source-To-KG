from llm_source_to_kg.graph.analysis_graph.state import AnalysisGraphState
import requests
import json
from typing import Dict, Any, List, Optional
import asyncio
from dataclasses import dataclass
from elasticsearch import Elasticsearch
import logging
from llm_source_to_kg.config import config

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OMOPMapping:
    concept_id: str
    concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    standard_concept: str
    confidence_score: float
    source_text: Optional[str] = None

@dataclass
class KnowledgeGraphNode:
    node_id: str
    node_type: str
    properties: Dict[str, Any]
    relationships: List[Dict[str, Any]]

async def mapping_to_omop(state: AnalysisGraphState) -> AnalysisGraphState:
    """
    분석된 의료 엔티티를 OMOP CDM으로 매핑하고 지식 그래프 노드를 생성합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 그래프 상태 (mapping_result와 kg_nodes 필드 포함)
    """
    analysis = state["analysis"]
    
    # Elasticsearch 클라이언트 설정
    # es = Elasticsearch("http://localhost:9200")
    es = Elasticsearch(f"http://{config.ES_SERVER_HOST}:{config.ES_SERVER_PORT}")
    
    mapping_results = {}
    kg_nodes = []
    
    # AnalysisSchema에서 엔티티 추출
    entities_to_map = []
    
    # 진단 관련 엔티티 추출
    if "diagnostic" in analysis and analysis["diagnostic"]:
        diagnostic = analysis["diagnostic"]
        entities_to_map.append({
            "entity_type": "diagnostic",
            "entity_name": diagnostic["concept_name"],
            "domain_id": diagnostic.get("domain_id", "Condition"),
            "vocabulary_id": diagnostic.get("vocabulary_id", "SNOMED")
        })
    
    # 약물 관련 엔티티 추출
    if "drug" in analysis and analysis["drug"]:
        drug = analysis["drug"]
        entities_to_map.append({
            "entity_type": "drug",
            "entity_name": drug["concept_name"],
            "domain_id": drug.get("domain_id", "Drug"),
            "vocabulary_id": drug.get("vocabulary_id", "RxNorm")
        })
    
    # 검사 관련 엔티티 추출
    if "test" in analysis and analysis["test"]:
        test = analysis["test"]
        entities_to_map.append({
            "entity_type": "test",
            "entity_name": test["concept_name"],
            "domain_id": test.get("domain_id", "Measurement"),
            "vocabulary_id": test.get("vocabulary_id", "LOINC")
        })
    
    # 수술 관련 엔티티 추출
    if "surgery" in analysis and analysis["surgery"]:
        surgery = analysis["surgery"]
        entities_to_map.append({
            "entity_type": "surgery",
            "entity_name": surgery["concept_name"],
            "domain_id": surgery.get("domain_id", "Procedure"),
            "vocabulary_id": surgery.get("vocabulary_id", "SNOMED")
        })
    
    # 기타 텍스트에서 추가 엔티티 추출 (source_text_span에서)
    if "source_text_span" in analysis and analysis["source_text_span"]:
        try:
            source_data = json.loads(analysis["source_text_span"])
            # 추가 엔티티 정보가 있다면 여기서 처리
        except (json.JSONDecodeError, TypeError):
            pass
    
    # 단일 코호트로 처리
    cohort_id = "single_cohort"
    mapped_entities = []
    
    # 각 엔티티를 OMOP에 매핑
    for entity_info in entities_to_map:
        entity_name = entity_info["entity_name"]
        entity_type = entity_info["entity_type"]
        
        try:
            # Elasticsearch에서 유사한 개념 검색
            query = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "concept_name": {
                                        "query": entity_name,
                                        "boost": 2.0
                                    }
                                }
                            },
                            {
                                "match_phrase": {
                                    "concept_name": {
                                        "query": entity_name,
                                        "boost": 3.0
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "size": 5
            }
            
            response = es.search(
                index="omop_concepts",
                body=query
            )
            
            hits = response["hits"]["hits"]
            
            if hits:
                # 가장 높은 점수의 매핑 선택
                best_match = hits[0]
                score = best_match["_score"]
                source = best_match["_source"]
                
                omop_mapping = OMOPMapping(
                    concept_id=source["concept_id"],
                    concept_name=source["concept_name"],
                    domain_id=source["domain_id"],
                    vocabulary_id=source["vocabulary_id"],
                    concept_class_id=source["concept_class_id"],
                    standard_concept=source["standard_concept"],
                    confidence_score=min(score / 10, 1.0),  # 점수를 0-1 범위로 정규화
                    source_text=entity_name
                )
                
                # 지식 그래프 노드 생성
                kg_node = create_knowledge_graph_node(omop_mapping, entity_type)
                kg_nodes.append(kg_node)
                
                mapped_entities.append({
                    "original_entity": entity_name,
                    "omop_mapping": omop_mapping.__dict__,
                    "mapping_status": "success",
                    "kg_node": kg_node.__dict__
                })
            else:
                # 매핑 실패 시 기본 정보로 노드 생성
                omop_mapping = OMOPMapping(
                    concept_id=f"UNMAPPED_{hash(entity_name) % 1000000}",
                    concept_name=entity_name,
                    domain_id=entity_info["domain_id"],
                    vocabulary_id=entity_info["vocabulary_id"],
                    concept_class_id=get_omop_concept_class(entity_type),
                    standard_concept="N",
                    confidence_score=0.0,
                    source_text=entity_name
                )
                
                kg_node = create_knowledge_graph_node(omop_mapping, entity_type)
                kg_nodes.append(kg_node)
                
                mapped_entities.append({
                    "original_entity": entity_name,
                    "omop_mapping": omop_mapping.__dict__,
                    "mapping_status": "unmapped",
                    "kg_node": kg_node.__dict__
                })
            
        except Exception as e:
            logger.error(f"매핑 중 오류 발생: {str(e)}")
            mapped_entities.append({
                "original_entity": entity_name,
                "omop_mapping": None,
                "mapping_status": "failed",
                "error": str(e)
            })
    
    # 매핑 결과 정리
    mapping_results[cohort_id] = {
        "status": "success",
        "mappings": {
            "all_entities": mapped_entities
        },
        "summary": {
            "total_entities": len(entities_to_map),
            "mapped_entities": len([m for m in mapped_entities if m["mapping_status"] == "success"])
        }
    }
    
    # 상태 업데이트
    state["mapping_result"] = mapping_results
    state["kg_nodes"] = kg_nodes
    
    return state


def create_default_mapping(entity_text: str, entity_type: str) -> OMOPMapping:
    """기본 OMOP 매핑을 생성합니다."""
    return OMOPMapping(
        concept_id=f"UNMAPPED_{hash(entity_text) % 1000000}",
        concept_name=entity_text,
        domain_id=get_omop_domain(entity_type),
        vocabulary_id=get_omop_vocabulary(entity_type),
        concept_class_id=get_omop_concept_class(entity_type),
        standard_concept="N",
        confidence_score=0.0,
        source_text=entity_text
    )

def create_knowledge_graph_node(omop_mapping: OMOPMapping, entity_type: str) -> KnowledgeGraphNode:
    """OMOP 매핑 정보를 기반으로 지식 그래프 노드를 생성합니다."""
    return KnowledgeGraphNode(
        node_id=omop_mapping.concept_id,
        node_type=omop_mapping.domain_id,
        properties={
            "concept_name": omop_mapping.concept_name,
            "vocabulary_id": omop_mapping.vocabulary_id,
            "concept_class_id": omop_mapping.concept_class_id,
            "standard_concept": omop_mapping.standard_concept,
            "confidence_score": omop_mapping.confidence_score,
            "source_text": omop_mapping.source_text
        },
        relationships=[]
    )

def get_omop_domain(entity_type: str) -> str:
    """엔티티 타입에 따른 OMOP 도메인 반환"""
    domain_mapping = {
        "condition_entities": "Condition",
        "condition_relationships": "Observation",
        "diagnostic_pathways": "Procedure",
        "condition_cohorts": "Observation",
        # 기존 매핑 유지 (하위 호환성)
        "drug": "Drug",
        "diagnostic": "Condition", 
        "medicalTest": "Measurement",
        "surgery": "Procedure",
        "symptoms": "Observation",
        "procedures": "Procedure"
    }
    return domain_mapping.get(entity_type, "Observation")

def get_omop_vocabulary(entity_type: str) -> str:
    """엔티티 타입에 따른 OMOP 어휘 반환"""
    vocab_mapping = {
        "condition_entities": "SNOMED",
        "condition_relationships": "SNOMED",
        "diagnostic_pathways": "SNOMED",
        "condition_cohorts": "SNOMED",
        # 기존 매핑 유지 (하위 호환성)
        "drug": "RxNorm",
        "diagnostic": "SNOMED",
        "medicalTest": "LOINC", 
        "surgery": "SNOMED",
        "symptoms": "SNOMED",
        "procedures": "SNOMED"
    }
    return vocab_mapping.get(entity_type, "SNOMED")

def get_omop_concept_class(entity_type: str) -> str:
    """엔티티 타입에 따른 OMOP 컨셉 클래스 반환"""
    class_mapping = {
        "condition_entities": "Clinical Finding",
        "condition_relationships": "Clinical Finding",
        "diagnostic_pathways": "Procedure",
        "condition_cohorts": "Clinical Finding",
        # 기존 매핑 유지 (하위 호환성)
        "drug": "Clinical Drug",
        "diagnostic": "Clinical Finding",
        "medicalTest": "Lab Test",
        "surgery": "Procedure", 
        "symptoms": "Clinical Finding",
        "procedures": "Procedure"
    }
    return class_mapping.get(entity_type, "Clinical Finding")

async def main():
    """테스트를 위한 메인 함수"""
    # 테스트용 상태 객체 생성
    test_state = {
        "analysis": {
            "cohort_analyses": {
                "test_cohort": {
                    "status": "success",
                    "entities": {
                        "diagnostic": ["Cardiovascular disease", "High risk of CVD"],
                        "drug": ["Atorvastatin", "Statin therapy"],
                        "medicalTest": ["QRISK3 score", "Lipid profile"]
                    }
                }
            }
        }
    }
    
    # 매핑 실행
    result_state = await mapping_to_omop(test_state)
    
    # 결과 출력
    print("매핑 결과:")
    print(json.dumps(result_state["mapping_result"], indent=2))
    print("\n지식 그래프 노드:")
    print(json.dumps([node.__dict__ for node in result_state["kg_nodes"]], indent=2))

if __name__ == "__main__":
    asyncio.run(main())
