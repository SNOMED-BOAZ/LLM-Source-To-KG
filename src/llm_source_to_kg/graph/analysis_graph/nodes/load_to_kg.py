from src.llm_source_to_kg.graph.analysis_graph.state import AnalysisGraphState
import json
from typing import Dict, Any, List
import requests
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_to_kg(state: AnalysisGraphState) -> AnalysisGraphState:
    return state # TODO: 테스트 중 임시 용도
    """
    OMOP 매핑 결과를 Neo4j 지식 그래프에 적재합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 그래프 상태 (kg_load_result 필드 포함)
    """
    mapping_result = state["mapping_result"]
    kg_nodes = state["kg_nodes"]
    
    # Neo4j 설정
    NEO4J_URI = "http://localhost:7474"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"  # 실제 비밀번호로 변경 필요
    
    load_results = {}
    total_loaded = 0
    
    # 각 코호트별 매핑 결과를 지식 그래프에 적재
    for cohort_id, mapping_data in mapping_result.items():
        if mapping_data["status"] != "success":
            load_results[cohort_id] = {
                "status": "skipped",
                "reason": "매핑 실패"
            }
            continue
        
        try:
            # 코호트 노드 생성
            cohort_node_query = create_cohort_node_query(cohort_id, state["cohort"][cohort_id])
            
            # 각 엔티티별 노드 및 관계 생성
            entity_queries = []
            mappings = mapping_data["mappings"]
            
            for entity_type, entities in mappings.items():
                for entity_data in entities:
                    if entity_data["mapping_status"] == "success":
                        omop_mapping = entity_data["omop_mapping"]
                        kg_node = entity_data["kg_node"]
                        
                        # 엔티티 노드 생성 쿼리
                        entity_query = create_entity_node_query(
                            entity_type,
                            kg_node
                        )
                        entity_queries.append(entity_query)
                        
                        # 코호트-엔티티 관계 생성 쿼리
                        relation_query = create_cohort_entity_relation_query(
                            cohort_id,
                            kg_node["node_id"],
                            entity_type
                        )
                        entity_queries.append(relation_query)
            
            # 전체 쿼리 구성
            all_queries = [cohort_node_query] + entity_queries
            
            # Neo4j에 쿼리 실행
            for query in all_queries:
                try:
                    response = requests.post(
                        f"{NEO4J_URI}/db/data/transaction/commit",
                        auth=(NEO4J_USER, NEO4J_PASSWORD),
                        json={"statements": [{"statement": query}]},
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                except Exception as e:
                    logger.error(f"쿼리 실행 중 오류 발생: {str(e)}")
                    raise
            
            load_results[cohort_id] = {
                "status": "success",
                "loaded_entities": len([e for et in mappings.values() for e in et if e["mapping_status"] == "success"]),
                "queries_executed": len(all_queries),
                "timestamp": datetime.now().isoformat()
            }
            
            total_loaded += load_results[cohort_id]["loaded_entities"]
            
        except Exception as e:
            load_results[cohort_id] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # 전체 적재 결과
    kg_load_result = {
        "overall_status": "success" if all(r["status"] == "success" for r in load_results.values()) else "partial",
        "cohort_results": load_results,
        "summary": {
            "total_cohorts": len(mapping_result),
            "successful_loads": len([r for r in load_results.values() if r["status"] == "success"]),
            "total_entities_loaded": total_loaded
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # 상태 업데이트
    state["kg_load_result"] = kg_load_result
    
    return state

def create_cohort_node_query(cohort_id: str, cohort_content: Dict[str, Any]) -> str:
    """코호트 노드 생성 Cypher 쿼리"""
    name_escaped = cohort_content.get("name", "").replace("'", "\\'")
    description_escaped = cohort_content.get("description", "").replace("'", "\\'")
    
    return f"""
    MERGE (c:Cohort {{id: '{cohort_id}'}})
    SET c.name = '{name_escaped}',
        c.description = '{description_escaped}',
        c.created_at = datetime()
    """

def create_entity_node_query(entity_type: str, kg_node: Dict[str, Any]) -> str:
    """엔티티 노드 생성 Cypher 쿼리"""
    properties = kg_node["properties"]
    concept_name_escaped = properties["concept_name"].replace("'", "\\'")
    source_text_escaped = properties.get("source_text", "").replace("'", "\\'")
    
    return f"""
    MERGE (e:Entity:OMOP {{concept_id: '{kg_node["node_id"]}'}})
    SET e += {{
        concept_name: '{concept_name_escaped}',
        domain_id: '{properties.get("domain_id", "")}',
        vocabulary_id: '{properties.get("vocabulary_id", "")}',
        concept_class_id: '{properties.get("concept_class_id", "")}',
        standard_concept: '{properties.get("standard_concept", "")}',
        confidence_score: {properties.get("confidence_score", 0.0)},
        source_text: '{source_text_escaped}',
        entity_type: '{entity_type}',
        created_at: datetime()
    }}
    """

def create_cohort_entity_relation_query(cohort_id: str, concept_id: str, entity_type: str) -> str:
    """코호트-엔티티 관계 생성 Cypher 쿼리"""
    relation_type = f"HAS_{entity_type.upper()}"
    
    return f"""
    MATCH (c:Cohort {{id: '{cohort_id}'}}), (e:Entity {{concept_id: '{concept_id}'}})
    MERGE (c)-[r:{relation_type}]->(e)
    SET r.created_at = datetime()
    """