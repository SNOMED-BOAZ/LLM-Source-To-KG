from llm_source_to_kg.graph.cohort_graph.orchestrator import build_cohort_graph
from llm_source_to_kg.graph.analysis_graph.orchestrator import build_analysis_graph

def run_full_workflow(input_state):
    # 1. Cohort 단계 실행
    cohort_graph = build_cohort_graph()
    cohort_state = cohort_graph.invoke(input_state)

    # 2. 분석 대상 배열 추출
    cohort_list = cohort_state["cohorts_json"]

    # 3. 각 대상에 대해 AnalysisGraph 실행
    analysis_graph = build_analysis_graph()
    analysis_results = []

    for cohort_item in cohort_list:
        # 상태 구성 (필요한 context만 추출)
        analysis_input = {
            "context": input_state["context"], # TODO: input_state 넣을지 cohort_state 넣을 지 정하기
            "question": input_state["question"], # TODO: analysis에 필요한 prompt로 넣기
            "cohort_subject": cohort_item,
        }
        result = analysis_graph.invoke(analysis_input)
        analysis_results.append(result)

    # 4. 결과 통합 또는 반환
    return {
        "cohort_state": cohort_state,
        "analysis_results": analysis_results
    }