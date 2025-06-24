import asyncio, os, json
from llm_source_to_kg.graph.cohort_graph.orchestrator import get_cohort_chain
from llm_source_to_kg.graph.analysis_graph.orchestrator import get_analysis_chain


def extract_subcohort_markdowns(cohort_state):
    """
    cohort_graph 결과에서 각 sub_cohort를 마크다운으로 변환하여 리스트로 반환
    """
    # cohorts_json: main_cohorts 리스트
    main_cohorts = cohort_state.get("cohorts_json", [])
    subcohort_markdowns = []
    subcohort_contexts = []
    for main in main_cohorts:
        main_subject = main.get("subject", "")
        main_details = main.get("details", "")
        for sub in main.get("sub_cohorts", []):
            desc = sub.get("description", {})
            subject = desc.get("subject", "")
            details = desc.get("details", "")
            # 마크다운 변환
            md = f"# {main_subject} - {subject}\n{main_details}\n{details}\n"
            # 포함/배제 기준 등 추가
            if sub.get("inclusion_criteria"):
                md += "\n## Inclusion Criteria\n" + str(sub["inclusion_criteria"]) + "\n"
            if sub.get("exclusion_criteria"):
                md += "\n## Exclusion Criteria\n" + str(sub["exclusion_criteria"]) + "\n"
            subcohort_markdowns.append(md)
            subcohort_contexts.append({
                "main_subject": main_subject,
                "subject": subject,
                "details": details
            })
    return subcohort_markdowns, subcohort_contexts


async def run_full_workflow(input_state):
    """
    전체 워크플로우: cohort_graph → analysis_graph 순차 실행
    """
    output_dir = "full_workflow_test_outputs"
    source_reference_number = input_state.get("source_reference_number", "")
    cohort_dir = f"{output_dir}/{source_reference_number}"
    
    # 1. Cohort 단계 실행
    cohort_chain = get_cohort_chain()
    cohort_state = await cohort_chain.ainvoke(input_state)

    # 2. 분석 대상(sub_cohort) 추출
    subcohort_markdowns, subcohort_contexts = extract_subcohort_markdowns(cohort_state)

    # outputs 디렉토리 생성 (없는 경우)
    os.makedirs(f"{cohort_dir}", exist_ok=True)
    
    # 마크다운 결과를 각각 파일로 저장
    for i, markdown in enumerate(cohort_state['cohorts_markdown'], 1):
        output_filename = f"{cohort_dir}/{i}.md"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(markdown)
    print(f"코호트 마크다운이 '{output_dir}/{cohort_state['source_reference_number']}/' 디렉터리에 저장되었습니다.")
    
    # 분석 결과를 JSON 파일로 저장
    analysis_results_dir = f"{cohort_dir}/analysis_results"
    os.makedirs(analysis_results_dir, exist_ok=True)  # 디렉토리가 없으면 생성
    
    # 3. 각 sub_cohort에 대해 analysis_graph 실행
    analysis_chain = get_analysis_chain()
    analysis_results = []
    for idx, md in enumerate(subcohort_markdowns, 1):
        analysis_input = {
            "source_reference_number": input_state.get("source_reference_number", ""),
            "cohort": md,
            "question": "",
            "context": input_state.get("context") or "NICE 가이드라인 기반 분석",
            "answer": "",
            "retries": 0,
            "is_valid": False,
            "analysis": {}
        }
        result = await analysis_chain.ainvoke(analysis_input)
        
        output_filename = f"{analysis_results_dir}/{idx}_analysis.json"  # .md를 _analysis.json으로 변경
        
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
        print(f"분석 결과가 {output_filename}에 저장되었습니다.")
        
        analysis_results.append(result)

    # 4. 결과 통합 또는 반환
    return {
        "cohort_state": cohort_state,
        "analysis_results": [
            {
                **r,
                "kg_nodes": [node.__dict__ for node in r.get('kg_nodes', [])]
            }
            for r in analysis_results
        ]
    }


def main():
    """
    예시 input으로 전체 워크플로우 실행 (테스트용)
    """
    import sys
    import json
    if len(sys.argv) > 1:
        source_reference_number = sys.argv[1]
    else:
        source_reference_number = "NG238"
    input_state = { "source_reference_number": source_reference_number }
    result = asyncio.run(run_full_workflow(input_state))
    # print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()