from datetime import datetime
from llm_source_to_kg.config import config
from llm_source_to_kg.graph.analysis_graph.nodes.mapping_to_omop import OMOPMapping
from src.llm_source_to_kg.graph.analysis_graph.state import AnalysisGraphState
from src.llm_source_to_kg.utils.grpc_asset import synonyms_data_pb2, synonyms_data_pb2_grpc
import grpc
import logging

logger = logging.getLogger(__name__)


def extract_synonym_pairs(state: AnalysisGraphState) -> list[synonyms_data_pb2.Synonym]:
    """state에서 원본-매핑된 concept_name 쌍을 추출한다."""
    synonyms = []

    mapping_result = state.get("mapping_result", {})
    for cohort_data in mapping_result.values():
        if cohort_data["status"] != "success":
            continue

        mappings = cohort_data.get("mappings", {})
        for entity_type_mappings in mappings.values():
            for entity_map in entity_type_mappings:
                if entity_map["mapping_status"] == "success":
                    original = entity_map["original_entity"]
                    mapped = entity_map["omop_mapping"]["concept_name"]
                    if original.lower() != mapped.lower():
                        synonym = synonyms_data_pb2.Synonym(
                            origin=original,
                            new=mapped,
                        )

                        synonyms.append(synonym)

    return synonyms


def send_synonyms_to_server(synonyms: list[synonyms_data_pb2.Synonym], grpc_host: str = f"{config.ES_SERVER_HOST}:{config.ES_SERVER_PORT}"):
    """gRPC를 통해 synonym 쌍을 서버로 전송한다."""
    with grpc.insecure_channel(grpc_host) as channel:
        stub = synonyms_data_pb2_grpc.SynonymServiceStub(channel)
        try:
            request = synonyms_data_pb2.SynonymDataRequest(
                time=str(datetime.now()),
                synonyms=synonyms
            )

            response = stub.addSynonym(request)
            logger.info(f"Response {response}")
        except Exception as e:
            logger.error(f"gRPC 전송 실패 - {type(e).__name__}: {str(e)}")


def update_synonyms(state: AnalysisGraphState) -> AnalysisGraphState:
    """동의어 사전 업데이트를 위한 gRPC 호출"""
    synonyms = extract_synonym_pairs(state)

    if not synonyms:
        logger.info("업데이트할 동의어가 없습니다.")
        return state

    logger.info(f"{len(synonyms)}개의 동의어를 gRPC 서버에 전송합니다.")
    send_synonyms_to_server(synonyms)

    return state


if __name__ == '__main__':
    test_omop_mapping = OMOPMapping(
        concept_id="concept_id_1",
        concept_name="test_4",
        domain_id="domain_id_1",
        vocabulary_id="vocabulary_id_1",
        concept_class_id="concept_class_id_1",
        standard_concept="S",
        confidence_score=0.0,
        source_text=None
    )

    test_cohort_mapping = {
        "entity_type_1": [
            {
                "original_entity": "entits_4",
                "omop_mapping": test_omop_mapping.__dict__,
                "mapping_status": "success",
                "kg_node": None
            }
        ]
    }

    test_states = {
        "mapping_result": {
            "cohort_id_1": {
                "status": "success",
                "mappings": test_cohort_mapping,
                "summary": None
            }
        }
    }

    update_synonyms(AnalysisGraphState(test_states))