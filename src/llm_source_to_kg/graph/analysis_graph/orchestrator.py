from langgraph.graph import END, StateGraph

from src.llm_source_to_kg.graph.analysis_graph.state import AnalysisGraphState
from src.llm_source_to_kg.graph.analysis_graph.nodes import (
    analyze_cohort,
    load_to_kg,
    mapping_to_omop,
    validate_analysis,
    update_synonyms
)

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
    analysis_graph.add_node("validate_analysis", validate_analysis)
    analysis_graph.add_node("mapping_to_omop", mapping_to_omop)
    analysis_graph.add_node("update_synonyms", update_synonyms)
    analysis_graph.add_node("load_to_kg", load_to_kg)
    
    # 엣지 정의 - 메인 플로우
    # 1. 코호트 분석 → 분석 검증
    analysis_graph.add_edge("analyze_cohort", "validate_analysis")
    
    # 조건부 엣지 정의 - 검증 결과에 따라 다른 경로로 진행
    analysis_graph.add_conditional_edges(
        "validate_analysis",
        route_after_validation,
        {
            "retry_analysis": "analyze_cohort",  # 분석 재시도
            "mapping_to_omop": "mapping_to_omop"  # OMOP 매핑 진행
        }
    )
    
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