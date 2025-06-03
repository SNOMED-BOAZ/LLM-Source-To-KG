#!/usr/bin/env python3
"""
BM25 기반 코호트 검증 시스템 테스트 스크립트
"""

import asyncio
import json
from llm_source_to_kg.graph.cohort_graph.state import CohortGraphState
from llm_source_to_kg.graph.cohort_graph.nodes.validate_cohort import validate_cohort
from llm_source_to_kg.utils.logger import get_logger


async def test_cohort_validation():
    """BM25 기반 코호트 검증 시스템 테스트"""
    logger = get_logger()
    logger.info("Starting BM25-based cohort validation test...")
    
    # 테스트용 샘플 상태 생성
    test_state = CohortGraphState(
        context="NICE 가이드라인 NG28 - 당뇨병 환자의 발 관리",
        question="당뇨병 환자 코호트 추출",
        answer="",
        is_valid=False,
        retries=0,
        source_reference_number="NG28",
        source_contents="""
        당뇨병 환자의 발 관리 가이드라인

        1.1 개요
        이 가이드라인은 당뇨병성 족부궤양의 예방과 관리에 대한 권고사항을 제공합니다.
        
        1.2 환자 집단
        이 가이드라인은 18세 이상의 당뇨병 환자에게 적용됩니다.
        특히 다음과 같은 고위험 환자들에게 중점을 둡니다:
        - 말초혈관질환이 있는 환자
        - 신경병증이 있는 환자  
        - 발 궤양의 병력이 있는 환자
        - 족부 변형이 있는 환자
        
        1.3 포함 기준
        - 1형 또는 2형 당뇨병 진단을 받은 환자
        - 18세 이상 성인
        - 발 관리가 필요한 환자
        - 당뇨병성 족부 합병증의 위험이 있는 환자
        
        1.4 배제 기준
        - 18세 미만의 소아 환자
        - 임신성 당뇨병 환자 (별도 가이드라인 적용)
        - 급성 감염으로 응급치료가 필요한 환자
        
        2.1 위험도 평가
        모든 당뇨병 환자는 연 1회 이상 족부 검진을 받아야 합니다.
        고위험 환자는 3-6개월마다 정기 검진을 실시합니다.
        
        2.2 예방 조치
        적절한 신발 착용과 일상적인 발 관리 교육이 필수입니다.
        """,
        cohorts_json=[
            {
                "cohort_id": "diabetic_foot_care",
                "name": "당뇨병 발 관리 대상 환자",
                "description": "발 관리가 필요한 18세 이상 당뇨병 환자",
                "inclusion_criteria": [
                    "1형 또는 2형 당뇨병 진단",
                    "18세 이상 성인",
                    "발 관리 필요",
                    "당뇨병성 족부 합병증 위험"
                ],
                "exclusion_criteria": [
                    "18세 미만",
                    "임신성 당뇨병",
                    "급성 감염으로 응급치료 필요"
                ],
                "high_risk_factors": [
                    "말초혈관질환",
                    "신경병증",
                    "발 궤양 병력",
                    "족부 변형"
                ],
                "monitoring_frequency": "고위험군은 3-6개월마다 검진"
            }
        ],
        cohorts_markdown=[
            "## 당뇨병 발 관리 대상 환자\n"
            "- **포함 기준**: 1형/2형 당뇨병, 18세 이상, 족부 합병증 위험\n"
            "- **배제 기준**: 18세 미만, 임신성 당뇨병, 급성 감염\n"
            "- **고위험 요인**: 말초혈관질환, 신경병증, 발 궤양 병력, 족부 변형\n"
            "- **모니터링**: 고위험군 3-6개월마다 정기 검진"
        ],
        validation_details=None
    )
    
    try:
        # 검증 실행
        logger.info("Executing BM25-based cohort validation...")
        result_state = await validate_cohort(test_state)
        
        # 결과 출력
        logger.info("=== BM25 기반 검증 결과 ===")
        logger.info(f"검증 성공: {result_state['is_valid']}")
        
        if result_state.get('validation_details'):
            details = result_state['validation_details']
            logger.info(f"검증 방법: {details.get('method', 'Unknown')}")
            logger.info(f"신뢰도 점수: {details.get('confidence_score', 0):.3f}")
            
            if 'questions_generated' in details:
                logger.info(f"생성된 질문 수: {len(details['questions_generated'])}")
                for i, question in enumerate(details['questions_generated'][:3], 1):
                    logger.info(f"  {i}. {question[:80]}...")
            
            if 'validation_summary' in details:
                summary = details['validation_summary']
                logger.info(f"SUPPORTS 평균: {summary.get('avg_supports', 0):.3f}")
                logger.info(f"REFUTES 평균: {summary.get('avg_refutes', 0):.3f}")
                logger.info(f"NEUTRAL 평균: {summary.get('avg_neutral', 0):.3f}")
                logger.info(f"처리된 증거 수: {summary.get('total_questions', 0)}")
                
            if 'nli_results' in details and details['nli_results']:
                logger.info("=== NLI 평가 샘플 ===")
                for i, result in enumerate(details['nli_results'][:2], 1):
                    logger.info(f"샘플 {i}: {result.get('label', 'unknown')} "
                               f"(신뢰도: {result.get('score', 0):.3f})")
                    logger.info(f"  증거: {result.get('evidence', '')[:60]}...")
        
        return result_state
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


def main():
    """메인 실행 함수"""
    try:
        result = asyncio.run(test_cohort_validation())
        print("\n✅ BM25 기반 코호트 검증 테스트가 완료되었습니다!")
        print(f"검증 결과: {'성공' if result['is_valid'] else '실패'}")
        
        if result.get('validation_details'):
            confidence = result['validation_details'].get('confidence_score', 0)
            method = result['validation_details'].get('method', 'Unknown')
            print(f"검증 방법: {method}")
            print(f"신뢰도: {confidence:.3f}")
            
            summary = result['validation_details'].get('validation_summary', {})
            if summary:
                print(f"SUPPORTS: {summary.get('avg_supports', 0):.3f}")
                print(f"REFUTES: {summary.get('avg_refutes', 0):.3f}")
                print(f"NEUTRAL: {summary.get('avg_neutral', 0):.3f}")
        
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 