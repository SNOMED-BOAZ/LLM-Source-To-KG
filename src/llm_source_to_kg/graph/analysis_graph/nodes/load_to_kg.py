from llm_source_to_kg.graph.analysis_graph.state import AnalysisGraphState
import json
from typing import Dict, Any, List
import requests
from datetime import datetime
import logging
from neo4j import GraphDatabase, basic_auth
from llm_source_to_kg.config import config

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEO4J_DATABASE = "rxnorm3"

def convert_to_neo4j_node(node: dict) -> dict:
    """
    node_type을 labels로, 나머지 필드는 properties로 변환.
    node_id, confidence_score는 properties에서 제외.
    """
    if not isinstance(node, dict):
        # KnowledgeGraphNode라면 dict로 변환
        if hasattr(node, 'dict'):
            node = node.dict()
        else:
            node = node.__dict__
    labels = [node.get("node_type", "Entity")]
    exclude_keys = {"node_type", "relationships", "labels", "properties", "node_id"}
    base_props = {k: v for k, v in node.items() if k not in exclude_keys}
    merged_props = dict(node.get("properties", {}))
    merged_props.update(base_props)
    # node_id, confidence_score는 properties에서 제거
    merged_props.pop("node_id", None)
    merged_props.pop("confidence_score", None)
    return {"labels": labels, "properties": merged_props}

def create_node_query(node: dict) -> str:
    """
    변환된 노드(dict)에서 Cypher MERGE 쿼리 생성. None/null 값은 제외.
    라벨은 대문자로 변환.
    """
    labels = node["labels"]
    label_str = ":" + ":".join(label.upper() for label in labels)
    props = node["properties"]
    prop_str = ", ".join([
        f"{k}: {json.dumps(v)}" for k, v in props.items() if v is not None
    ])
    return f"MERGE (n{label_str} {{{prop_str}}}) RETURN elementId(n) as neo4j_id, n"

def load_to_kg(state: AnalysisGraphState) -> AnalysisGraphState:
    """
    OMOP 매핑 결과를 Neo4j 지식 그래프에 적재합니다.
    Args:
        state: 현재 그래프 상태
    Returns:
        업데이트된 그래프 상태 (kg_nodes 필드 포함)
    """
    kg_nodes = state["kg_nodes"]
    NEO4J_URI = config.NEO4J_SERVER_URI
    NEO4J_USER = config.NEO4J_SERVER_USER
    NEO4J_PASSWORD = config.NEO4J_SERVER_PASSWORD
    NEO4J_DATABASE = config.NEO4J_SERVER_DATABASE

    logger.info(f"Neo4j({NEO4J_URI})에 {len(kg_nodes)}개 노드 적재 시도 (DB: {NEO4J_DATABASE})")

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD)
    )

    results = []
    with driver.session(database=NEO4J_DATABASE) as session:
        for idx, node in enumerate(kg_nodes, 1):
            try:
                neo4j_node = convert_to_neo4j_node(node)
                cypher = create_node_query(neo4j_node)
                logger.info(f"[{idx}] Cypher: {cypher}")
                result = session.run(cypher)
                record = result.single()
                neo4j_id = record["neo4j_id"]
                node_obj = record["n"]
                node_props = neo4j_node["properties"]
                # 주요 식별자 추출 (concept_name, name, 등)
                main_name = node_props.get("concept_name") or node_props.get("name") or None
                results.append({
                    "labels": neo4j_node["labels"],
                    "neo4j_id": neo4j_id,
                    "main_name": main_name,
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"[{idx}] 노드 생성 실패: {e}")
                results.append({
                    "labels": node.get("labels", [node.get("node_type", "Entity")]),
                    "neo4j_id": None,
                    "main_name": node.get("concept_name") or node.get("name"),
                    "status": "fail",
                    "error": str(e)
                })

    driver.close()
    logger.info(f"총 {len(results)}개 노드 처리 완료. (성공: {sum(1 for r in results if r['status']=='success')}, 실패: {sum(1 for r in results if r['status']=='fail')})")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    state["kg_load_result"] = results
    return state