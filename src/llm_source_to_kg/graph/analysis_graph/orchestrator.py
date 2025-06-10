from langgraph.graph import END, StateGraph

from src.llm_source_to_kg.graph.analysis_graph.state import AnalysisGraphState
from src.llm_source_to_kg.graph.analysis_graph.nodes import (
    analyze_cohort,
    load_to_kg,
    mapping_to_omop,
    validate_analysis,
    update_synonyms
)
from llm_source_to_kg.utils.logger import get_logger
import json

# 전역 로거 초기화
logger = get_logger("analysis_graph")

MAX_RETRIES = 3

def route_after_validation(state: AnalysisGraphState) -> str:
    """검증 후 라우팅 결정"""
    if not state["is_valid"] and state["retries"] < MAX_RETRIES:
        return "retry_analysis"
    elif state["is_valid"]:
        return "mapping_to_omop"
    else:
        return END

def build_analysis_graph() -> StateGraph:
    """
    분석 그래프 구성하기
    
    플로우:
    1. 코호트 분석 - 코호트 데이터를 분석합니다.
    2. 분석 검증 - 분석 결과의 유효성을 검증합니다.
    3. 조건부 분기:
       a. 유효하지 않고 재시도 가능 → 분석 재시도
       b. 유효함 → OMOP 매핑 진행
       c. 재시도 불가 → 종료
    4. OMOP 매핑 - 분석 결과를 OMOP CDM으로 매핑합니다.
    5. 매핑된 용어를 ES 서버로 보내 동의어 사전을 업데이트합니다.
    6. 지식 그래프 적재 - 매핑 결과를 지식 그래프에 적재합니다.
    
    Returns:
        StateGraph: 구성된 분석 그래프
    """
    # 그래프 초기화
    analysis_graph = StateGraph(AnalysisGraphState)

    # 노드 추가 - 각 단계별 처리 함수 등록
    analysis_graph.add_node("analyze_cohort", analyze_cohort)
    # TODO: 테스트용 임시 제거 - analysis_graph.add_node("validate_analysis", validate_analysis)
    analysis_graph.add_node("mapping_to_omop", mapping_to_omop)
    analysis_graph.add_node("update_synonyms", update_synonyms)
    analysis_graph.add_node("load_to_kg", load_to_kg)
    
    # 엣지 정의 - 메인 플로우
    # TODO: 테스트용 임시 제거 - 1. 코호트 분석 → 분석 검증
    # TODO: 테스트용 임시 제거 - analysis_graph.add_edge("analyze_cohort", "validate_analysis")
    
    # 테스트용 임시 변경: 코호트 분석 → OMOP 매핑 (검증 단계 건너뛰기)
    analysis_graph.add_edge("analyze_cohort", "mapping_to_omop")
    
    # TODO: 테스트용 임시 제거 - 조건부 엣지 정의 - 검증 결과에 따라 다른 경로로 진행
    # analysis_graph.add_conditional_edges(
    #     "validate_analysis",
    #     route_after_validation,
    #     {
    #         "retry_analysis": "analyze_cohort",  # 분석 재시도
    #         "mapping_to_omop": "mapping_to_omop",  # OMOP 매핑 진행
    #         END: END  # 재시도 한계 도달 시 종료
    #     }
    # )
    
    # OMOP 매핑 → 지식 그래프 적재
    analysis_graph.add_edge("mapping_to_omop", "update_synonyms")

    # 매핑된 용어 → ES 서버로 전송 및 동의어 사전 업데이트
    analysis_graph.add_edge("update_synonyms", "load_to_kg")
    
    # 지식 그래프 적재 후 종료
    analysis_graph.add_edge("load_to_kg", END)
    
    # 시작 노드 설정
    analysis_graph.set_entry_point("analyze_cohort")

    return analysis_graph

def get_analysis_chain():
    """
    분석 체인 인스턴스 반환 - 컴파일된 그래프 반환
    
    Returns:
        컴파일된 분석 그래프 체인
    """
    graph = build_analysis_graph()
    return graph.compile()

def visualize_analysis_graph():
    """
    분석 그래프를 시각화하여 저장합니다.
    
    Returns:
        None: 그래프를 'analysis_graph.png' 파일로 저장합니다.
    """
    try:
        import graphviz
        
        graph = build_analysis_graph()
        dot = graph.get_graph().draw_graphviz(engine="dot")
        dot.render("analysis_graph", format="png", cleanup=True)
        print("분석 그래프가 'analysis_graph.png' 파일로 시각화되었습니다.")
    except ImportError:
        print("그래프 시각화를 위해 graphviz를 설치해주세요: pip install graphviz")
    except Exception as e:
        print(f"그래프 시각화 중 오류 발생: {e}")


async def test_run():
    """
    분석 그래프 테스트 실행 함수
    
    test_outputs 디렉터리에서 source_reference_number 디렉터리 안의 *.md 파일들을 
    코호트로 사용하여 분석 그래프를 테스트합니다.
    """
    import os
    import glob
    
    # 테스트 설정
    source_reference_number = "NG238"
    test_outputs_dir = "test_outputs"
    cohort_dir = f"{test_outputs_dir}/{source_reference_number}"
    
    # 코호트 마크다운 파일들 로드
    if not os.path.exists(cohort_dir):
        logger.error(f"오류: '{cohort_dir}' 디렉터리가 존재하지 않습니다.")
        return
    
    # *.md 파일들 찾기
    md_files = glob.glob(f"{cohort_dir}/*.md")
    if not md_files:
        logger.error(f"오류: '{cohort_dir}' 디렉터리에 마크다운 파일이 없습니다.")
        return
    
    # 모든 코호트 파일 처리
    for test_cohort_file in sorted(md_files):
        with open(test_cohort_file, 'r', encoding='utf-8') as f:
            cohort_content = f.read()
        
        logger.info(f"테스트할 코호트 파일: {test_cohort_file}")
        # 첫 번째 제목 추출
        lines = cohort_content.split('\n')
        title = next((line[2:].strip() for line in lines if line.startswith('# ')), 'Unknown')
        logger.info(f"코호트 제목: {title}")
        
        # 분석 그래프 실행을 위한 초기 상태 설정
        initial_state = AnalysisGraphState(
            source_reference_number=source_reference_number,
            cohort=cohort_content,  # 단일 cohort 마크다운 문자열
            question="이 코호트에 대한 의료 분석을 수행해주세요.",
            context="NICE 가이드라인 NG238에서 추출된 코호트 데이터에 대한 분석",
            answer="",
            retries=0,
            is_valid=False,
            analysis={}  # 빈 딕셔너리
        )
        
        # 분석 체인 생성 및 실행
        analysis_chain = get_analysis_chain()
        
        try:
            logger.info(f"\n{'='*20} {test_cohort_file} 분석 시작 {'='*20}")
            result = await analysis_chain.ainvoke(initial_state)
            
            logger.info("=" * 50)
            logger.info(f"파일: {test_cohort_file}")
            logger.info("분석 그래프 테스트 결과:")
            logger.info("=" * 50)
            logger.info(f"Source Reference Number: {result.get('source_reference_number', 'N/A')}")
            logger.info(f"분석 유효성: {result.get('is_valid', False)}")
            logger.info(f"재시도 횟수: {result.get('retries', 0)}")
            logger.info(f"응답: {result.get('answer', 'N/A')[:200]}...")
            
            # 분석 결과 출력
            if result.get('mapping_result'):
                mapping_result = result['mapping_result']
                logger.info(f"\n매핑 결과:")
                for cohort_id, cohort_result in mapping_result.items():
                    logger.info(f"\n코호트 ID: {cohort_id}")
                    logger.info(f"  - 상태: {cohort_result.get('status', 'N/A')}")
                    logger.info(f"  - 총 엔티티 수: {cohort_result['summary']['total_entities']}")
                    logger.info(f"  - 매핑된 엔티티 수: {cohort_result['summary']['mapped_entities']}")
                    
                    if 'mappings' in cohort_result and 'all_entities' in cohort_result['mappings']:
                        logger.info("\n  매핑된 엔티티:")
                        for entity in cohort_result['mappings']['all_entities']:
                            logger.info(f"    - 원본 엔티티: {entity['original_entity']}")
                            logger.info(f"      매핑 상태: {entity['mapping_status']}")
                            if entity['mapping_status'] == 'success':
                                omop = entity['omop_mapping']
                                logger.info(f"      OMOP 매핑: {omop['concept_name']} (ID: {omop['concept_id']})")
                                logger.info(f"      신뢰도: {omop['confidence_score']:.2f}")
            
            # 중간 상태 로깅
            logger.info("\n중간 상태:")
            logger.info(f"analysis 상태: {json.dumps(result.get('analysis', {}), indent=2, ensure_ascii=False)}")
            logger.info(f"mapping_result 상태: {json.dumps(result.get('mapping_result', {}), indent=2, ensure_ascii=False)}")
            logger.info(f"kg_nodes 상태: {json.dumps([node.__dict__ for node in result.get('kg_nodes', [])], indent=2, ensure_ascii=False)}")
            
            logger.info("=" * 50)
            logger.info("분석 그래프 테스트가 완료되었습니다.")
            
            # 분석 결과를 JSON 파일로 저장
            analysis_results_dir = f"{cohort_dir}/analysis_results"
            os.makedirs(analysis_results_dir, exist_ok=True)  # 디렉토리가 없으면 생성
            
            # 파일 이름에서 경로 제거하고 파일명만 추출
            file_name = os.path.basename(test_cohort_file)
            output_filename = f"{analysis_results_dir}/{file_name[:-3]}_analysis.json"  # .md를 _analysis.json으로 변경
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "source_reference_number": result.get('source_reference_number', 'N/A'),
                    "is_valid": result.get('is_valid', False),
                    "retries": result.get('retries', 0),
                    "answer": result.get('answer', 'N/A'),
                    "analysis": result.get('analysis', {}),
                    "mapping_result": result.get('mapping_result', {}),
                    "kg_nodes": [node.__dict__ for node in result.get('kg_nodes', [])]
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"분석 결과가 {output_filename}에 저장되었습니다.")
            
        except Exception as e:
            logger.error(f"분석 그래프 실행 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())


def main():
    """
    메인 함수 - test_run 코루틴을 실행합니다.
    poetry test-analysis-graph 명령어로 실행됩니다.
    """
    import asyncio
    asyncio.run(test_run())


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_run())