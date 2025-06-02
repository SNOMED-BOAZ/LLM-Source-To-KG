# Orchestrator for LangGraph
from langgraph.graph import END, StateGraph

from llm_source_to_kg.graph.cohort_graph.state import CohortGraphState
from llm_source_to_kg.graph.cohort_graph.utils import route_after_validation
from llm_source_to_kg.graph.cohort_graph.nodes import (
    load_source_content,
    extract_cohorts,
    validate_cohort,
    retry_extract_cohort,
    return_final_cohorts
)

import asyncio


def build_cohort_graph() -> StateGraph:
    """
    코호트 그래프 구성하기
    
    플로우:
    1. 소스 컨텐츠 로드 - 소스 문서를 로드합니다.
    2. 코호트 추출 - 문서에서 코호트 정보를 추출합니다.
    3. 코호트 검증 - 추출된 코호트의 유효성을 검증합니다.
    4. 조건부 분기:
       a. 유효하지 않은 코호트가 있고 재시도 가능 → 개별 코호트 재시도
       b. 모든 코호트가 유효하거나 재시도 불가 → 최종 코호트 목록 반환
    5. 최종 코호트 목록 반환 - 유효한 코호트만 필터링하여 반환합니다.
    
    Returns:
        StateGraph: 구성된 코호트 그래프
    """
    # 그래프 초기화
    cohort_graph = StateGraph(CohortGraphState)
    
    # 노드 추가 - 각 단계별 처리 함수 등록
    cohort_graph.add_node("load_source_content", load_source_content)
    cohort_graph.add_node("extract_cohorts", extract_cohorts)
    cohort_graph.add_node("validate_cohort", validate_cohort)
    cohort_graph.add_node("retry_extract_cohort", retry_extract_cohort)
    cohort_graph.add_node("return_final_cohorts", return_final_cohorts)
    
    # 엣지 정의 - 메인 플로우
    # 1. 소스 컨텐츠 로드 → 코호트 추출
    cohort_graph.add_edge("load_source_content", "extract_cohorts")
    # 2. 코호트 추출 → 코호트 검증
    cohort_graph.add_edge("extract_cohorts", "validate_cohort")
    
    # 조건부 엣지 정의 - 검증 결과에 따라 다른 경로로 진행
    # utils.route_after_validation 함수에서 다음 단계 결정
    cohort_graph.add_conditional_edges(
        "validate_cohort",
        route_after_validation,
        {
            "retry_extract_cohort": "retry_extract_cohort",  # 유효하지 않은 코호트 재시도
            "return_final_cohorts": "return_final_cohorts"   # 최종 결과 반환
        }
    )
    
    # 재시도 후 다시 검증 단계로
    cohort_graph.add_edge("retry_extract_cohort", "validate_cohort")
    
    # 최종 코호트 반환 후 종료
    cohort_graph.add_edge("return_final_cohorts", END)
    
    # 시작 노드 설정
    cohort_graph.set_entry_point("load_source_content")
    
    return cohort_graph


def get_cohort_chain():
    """
    코호트 체인 인스턴스 반환 - 컴파일된 그래프 반환
    
    Returns:
        컴파일된 코호트 그래프 체인
    """
    graph = build_cohort_graph()
    return graph.compile()


def visualize_cohort_graph():
    """
    코호트 그래프를 시각화하여 저장합니다.
    
    Returns:
        None: 그래프를 'cohort_graph.png' 파일로 저장합니다.
    """
    try:
        graph = build_cohort_graph()
        
        # LangGraph 0.4.3+ 버전에서는 draw_png() 사용 (GraphViz 사용)
        try:
            png_data = graph.get_graph().draw_png()
            with open("cohort_graph.png", "wb") as f:
                f.write(png_data)
            print("코호트 그래프가 'cohort_graph.png' 파일로 시각화되었습니다.")
        except Exception as graphviz_error:
            # GraphViz가 없거나 실패한 경우 Mermaid PNG 사용
            print(f"GraphViz 시각화 실패: {graphviz_error}")
            print("Mermaid PNG 시각화를 시도합니다...")
            
            mermaid_png_data = graph.get_graph().draw_mermaid_png()
            with open("cohort_graph_mermaid.png", "wb") as f:
                f.write(mermaid_png_data)
            print("코호트 그래프가 'cohort_graph_mermaid.png' 파일로 시각화되었습니다.")
            
    except ImportError as e:
        print(f"시각화 라이브러리 가져오기 실패: {e}")
        print("그래프 시각화를 위해 다음을 시도해보세요:")
        print("  pip install graphviz")
        print("  또는 시스템에 Graphviz를 설치하세요")
    except Exception as e:
        print(f"그래프 시각화 중 오류 발생: {e}")


async def test_run():
    # 기존의 build_cohort_graph() 함수 활용
    source_reference_number = "NG238"
    output_dir = "test_outputs"
    
    # 기존 코호트 그래프 함수 사용
    cohort_chain = get_cohort_chain()
    
    # 그래프 실행
    result = await cohort_chain.ainvoke({"source_reference_number": source_reference_number})
    
    # print("테스트 결과:")
    # print(result)

    print("마크다운 변환된 코호트 배열")
    import os
    
    # outputs 디렉토리 생성 (없는 경우)
    os.makedirs(f"{output_dir}/{result['source_reference_number']}", exist_ok=True)
    
    # 마크다운 결과를 각각 파일로 저장
    for i, markdown in enumerate(result['cohorts_markdown'], 1):
        output_filename = f"{output_dir}/{result['source_reference_number']}/{i}.md"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(markdown)
    
    print(f"코호트 마크다운이 '{output_dir}/{result['source_reference_number']}/' 디렉터리에 저장되었습니다.")
    
    # visualize_cohort_graph()

def main():
    """
    메인 함수 - test_run 코루틴을 실행합니다.
    """
    asyncio.run(test_run())

if __name__ == "__main__":
    asyncio.run(test_run())