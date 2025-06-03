import re
import os
import asyncio
from typing import List, Dict, Any, Tuple
from pathlib import Path

from transformers import pipeline
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

from llm_source_to_kg.graph.cohort_graph.state import CohortGraphState
from llm_source_to_kg.llm.gemini import GeminiLLM
from llm_source_to_kg.config import config
from llm_source_to_kg.utils.logger import get_logger


class CohortValidator:
    """코호트 검증을 위한 클래스"""
    
    def __init__(self):
        self.logger = get_logger()
        self.llm = GeminiLLM()
        self.nli_model = None
        self._load_models()
        self._download_nltk_data()
        
    def _download_nltk_data(self):
        """NLTK 데이터 다운로드"""
        try:
            nltk.download('punkt', quiet=True)
            self.logger.info("NLTK punkt data downloaded successfully")
        except Exception as e:
            self.logger.warning(f"Error downloading NLTK data: {e}")
        
    def _load_models(self):
        """NLI 모델 초기화"""
        try:
            # NLI 모델 로드 (facebook/bart-large-mnli)
            self.logger.info("Loading NLI model...")
            
            # 애플 실리콘 맥북에서 MPS 사용, 그 외에는 CPU 사용
            import torch
            if torch.backends.mps.is_available():
                device = 0  # MPS 사용
                self.logger.info("Using MPS device for NLI model")
            else:
                device = -1  # CPU 사용
                self.logger.info("Using CPU device for NLI model")
            
            self.nli_model = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device
            )
            self.logger.info("NLI model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading NLI model: {e}")
            self.logger.warning("NLI model loading failed, will use fallback verification method")
            self.nli_model = None
    
    def _load_prompt_template(self) -> str:
        """검증 프롬프트 템플릿 로드"""
        try:
            prompt_path = Path(__file__).parent.parent / "prompts" / "cohort_validation_prompt.txt"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error loading prompt template: {e}")
            raise
    
    async def generate_validation_questions(self, cohort_info: str, reference_number: str) -> str:
        """
        1단계: 코호트 분석 진술 검증을 위한 5가지 sub-question 생성
        
        Args:
            cohort_info: 검증할 코호트 정보
            reference_number: 가이드라인 참조번호
            
        Returns:
            str: LLM이 생성한 5가지 검증 질문
        """
        try:
            prompt_template = self._load_prompt_template()
            prompt = prompt_template.format(
                cohort_info=cohort_info,
                reference_number=reference_number
            )
            
            self.logger.info("Generating validation questions with Gemini...")
            response = await self.llm.call_llm(prompt)
            
            return response.content
        
        except Exception as e:
            self.logger.error(f"Error generating validation questions: {e}")
            self.logger.warning("Switching to offline mode with predefined questions...")
            
            # 오프라인 모드: 미리 정의된 검증 질문 사용
            offline_questions = self._get_offline_validation_questions()
            return offline_questions
    
    def _get_offline_validation_questions(self) -> str:
        """
        오프라인 모드용 미리 정의된 검증 질문들
        
        Returns:
            str: 번호가 매겨진 검증 질문들
        """
        return """
1. 코호트에서 정의한 환자 집단(당뇨병 환자)이 가이드라인에서 명시된 대상 환자와 정확히 일치하는가?

2. 코호트의 포함 기준(1형/2형 당뇨병, 18세 이상, 발 관리 필요)이 가이드라인에서 제시한 모든 중요한 조건을 완전히 포함하고 있는가?

3. 코호트의 배제 기준(18세 미만, 임신성 당뇨병, 급성 감염)이 가이드라인의 권고사항과 정확히 일치하는가?

4. 코호트가 적용되는 임상 상황(당뇨병성 족부 관리)이 가이드라인에서 명시한 상황과 적절히 부합하는가?

5. 코호트와 관련된 고위험 요인(말초혈관질환, 신경병증, 발 궤양 병력, 족부 변형)과 모니터링 빈도가 가이드라인의 권고 수준과 일관성을 가지는가?
        """.strip()
    
    def extract_questions_from_response(self, llm_response: str) -> List[str]:
        """
        2단계: 번호로 시작하는 문장을 추출하여 sub-question 리스트로 변환
        
        Args:
            llm_response: LLM 응답 텍스트
            
        Returns:
            List[str]: 추출된 질문 리스트
        """
        try:
            # 번호로 시작하는 문장을 찾는 정규표현식
            # 패턴: 숫자. 로 시작하는 문장
            pattern = r'^\d+\.\s*(.+?)(?=\n\d+\.|$)'
            
            questions = []
            matches = re.findall(pattern, llm_response, re.MULTILINE | re.DOTALL)
            
            for match in matches:
                # 줄바꿈과 불필요한 공백 정리
                question = re.sub(r'\s+', ' ', match.strip())
                if question:
                    questions.append(question)
            
            self.logger.info(f"Extracted {len(questions)} validation questions")
            return questions
        
        except Exception as e:
            self.logger.error(f"Error extracting questions: {e}")
            return []
    
    def retrieve_evidence(self, document_text: str, query: str, top_n: int = 3) -> List[str]:
        """
        3단계: BM25를 사용해 가이드라인 문서에서 증거 검색
        
        Args:
            document_text: 가이드라인 문서 전체 텍스트
            query: 검색 질문
            top_n: 반환할 증거의 최대 개수
            
        Returns:
            List[str]: 관련 증거 텍스트 리스트
        """
        try:
            # 문서를 문장 단위로 분할
            sentences = re.split(r'(?<=[.!?]) +', document_text)
            
            # 빈 문장 제거 및 최소 길이 필터링
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if not sentences:
                self.logger.warning("No valid sentences found in document")
                return []
            
            # 문장들을 토큰화
            tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
            
            # BM25 모델 생성
            bm25 = BM25Okapi(tokenized_sentences)
            
            # 질문을 토큰화
            tokenized_query = word_tokenize(query.lower())
            
            # BM25 스코어 계산
            scores = bm25.get_scores(tokenized_query)
            
            # 스코어가 높은 순으로 정렬하여 상위 문장들 추출
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            evidence = [sentences[i] for i in top_indices if scores[i] > 0][:top_n]
            
            self.logger.info(f"Retrieved {len(evidence)} evidence pieces using BM25")
            return evidence
        
        except Exception as e:
            self.logger.error(f"Error in BM25 evidence retrieval: {e}")
            return []
    
    def search_evidence_in_guideline(self, question: str, guideline_content: str, reference_number: str) -> List[str]:
        """
        3단계: 각 sub-question에 대해 가이드라인 문서에서 증거 검색 (BM25 기반)
        
        Args:
            question: 검증 질문
            guideline_content: 가이드라인 전체 내용
            reference_number: 가이드라인 참조번호
            
        Returns:
            List[str]: 관련 증거 텍스트 리스트
        """
        try:
            # BM25를 사용한 증거 검색
            evidence_list = self.retrieve_evidence(guideline_content, question, top_n=5)
            
            self.logger.info(f"Found {len(evidence_list)} evidence pieces for question")
            return evidence_list
        
        except Exception as e:
            self.logger.error(f"Error searching evidence: {e}")
            return []
    
    def verify_claim_with_evidence(self, claim: str, evidence_list: List[str]) -> List[Dict[str, Any]]:
        """
        4단계: 증거와 진술 간의 관계를 NLI 모델로 평가 (개선된 버전)
        
        Args:
            claim: 검증할 주장/질문
            evidence_list: 증거 텍스트 리스트
            
        Returns:
            List[Dict[str, Any]]: 각 증거에 대한 NLI 평가 결과
        """
        verification_results = []
        
        for evidence in evidence_list:
            if not evidence.strip():
                continue
                
            try:
                if self.nli_model is not None:
                    # NLI 모델 사용
                    result = self.nli_model(evidence, candidate_labels=["supports", "refutes", "neutral"])
                    
                    verification_results.append({
                        "evidence": evidence[:200] + "..." if len(evidence) > 200 else evidence,
                        "label": result['labels'][0],
                        "score": result['scores'][0],
                        "all_scores": dict(zip(result['labels'], result['scores'])),
                        "method": "nli_model"
                    })
                else:
                    # 폴백 검증 방법 사용
                    self.logger.debug("Using fallback verification method")
                    result = self._fallback_verification(claim, evidence)
                    result["evidence"] = evidence[:200] + "..." if len(evidence) > 200 else evidence
                    verification_results.append(result)
                    
            except Exception as e:
                self.logger.error(f"Error in verification for evidence: {e}")
                verification_results.append({
                    "evidence": evidence[:200] + "..." if len(evidence) > 200 else evidence,
                    "label": "neutral",
                    "score": 0.5,
                    "method": "error_fallback",
                    "error": str(e)
                })
        
        return verification_results
    
    def evaluate_nli_relationship(self, question: str, evidence: str) -> Dict[str, Any]:
        """
        4단계: 증거와 진술 간의 관계를 NLI 모델로 평가 (단일 증거용)
        
        Args:
            question: 검증 질문
            evidence: 증거 텍스트
            
        Returns:
            Dict[str, Any]: NLI 평가 결과
        """
        try:
            if not evidence.strip():
                return {"label": "neutral", "score": 0.5, "method": "empty_evidence"}
            
            if self.nli_model is not None:
                # NLI 분류를 위한 라벨 (supports, refutes, neutral로 변경)
                candidate_labels = ["supports", "refutes", "neutral"]
                
                # NLI 모델 실행
                result = self.nli_model(evidence, candidate_labels)
                
                # 결과 정리
                nli_result = {
                    "label": result["labels"][0],
                    "score": result["scores"][0],
                    "all_scores": dict(zip(result["labels"], result["scores"])),
                    "method": "nli_model"
                }
                
                self.logger.debug(f"NLI result: {nli_result}")
                return nli_result
            else:
                # 폴백 검증 방법 사용
                self.logger.debug("Using fallback verification method")
                return self._fallback_verification(question, evidence)
        
        except Exception as e:
            self.logger.error(f"Error in NLI evaluation: {e}")
            return {
                "label": "neutral", 
                "score": 0.5,
                "method": "error_fallback",
                "error": str(e)
            }
    
    def calculate_validation_score(self, nli_results: List[Dict[str, Any]]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        검증 결과를 종합하여 최종 점수 계산 (supports/refutes 기준으로 변경)
        
        Args:
            nli_results: NLI 평가 결과 리스트
            
        Returns:
            Tuple[bool, float, Dict]: (is_valid, confidence_score, details)
        """
        try:
            if not nli_results:
                return False, 0.0, {"error": "No NLI results available"}
            
            supports_scores = []
            refutes_scores = []
            neutral_scores = []
            
            for result in nli_results:
                if result["label"] == "supports":
                    supports_scores.append(result["score"])
                elif result["label"] == "refutes":
                    refutes_scores.append(result["score"])
                else:
                    neutral_scores.append(result["score"])
            
            # 점수 계산
            avg_supports = sum(supports_scores) / len(supports_scores) if supports_scores else 0
            avg_refutes = sum(refutes_scores) / len(refutes_scores) if refutes_scores else 0
            avg_neutral = sum(neutral_scores) / len(neutral_scores) if neutral_scores else 0
            
            # 검증 기준
            # - SUPPORTS 평균이 0.6 이상이고
            # - REFUTES 평균이 0.3 이하이면 유효
            is_valid = avg_supports >= 0.6 and avg_refutes <= 0.3
            confidence_score = max(0, avg_supports - avg_refutes)
            
            details = {
                "supports_count": len(supports_scores),
                "refutes_count": len(refutes_scores),
                "neutral_count": len(neutral_scores),
                "avg_supports": avg_supports,
                "avg_refutes": avg_refutes,
                "avg_neutral": avg_neutral,
                "total_questions": len(nli_results)
            }
            
            return is_valid, confidence_score, details
        
        except Exception as e:
            self.logger.error(f"Error calculating validation score: {e}")
            return False, 0.0, {"error": str(e)}
    
    def _fallback_verification(self, question: str, evidence: str) -> Dict[str, Any]:
        """
        NLI 모델이 없을 때 사용할 폴백 검증 방법
        키워드 매칭 기반의 간단한 검증
        
        Args:
            question: 검증 질문
            evidence: 증거 텍스트
            
        Returns:
            Dict[str, Any]: 폴백 검증 결과
        """
        try:
            # 질문에서 주요 키워드 추출
            question_words = set(re.findall(r'\b\w{3,}\b', question.lower()))
            evidence_words = set(re.findall(r'\b\w{3,}\b', evidence.lower()))
            
            # 겹치는 키워드 비율 계산
            common_words = question_words & evidence_words
            if question_words:
                overlap_ratio = len(common_words) / len(question_words)
            else:
                overlap_ratio = 0
            
            # 간단한 규칙 기반 판정
            if overlap_ratio >= 0.4:  # 40% 이상 겹치면 supports
                label = "supports"
                score = min(0.8, overlap_ratio * 2)  # 최대 0.8
            elif overlap_ratio <= 0.1:  # 10% 이하면 neutral
                label = "neutral"
                score = 0.5
            else:
                label = "supports"  # 중간 정도는 약한 supports
                score = overlap_ratio
            
            return {
                "label": label,
                "score": score,
                "method": "keyword_fallback",
                "overlap_ratio": overlap_ratio,
                "common_words": list(common_words)[:5]  # 최대 5개만
            }
            
        except Exception as e:
            self.logger.error(f"Error in fallback verification: {e}")
            return {
                "label": "neutral",
                "score": 0.5,
                "method": "fallback_error",
                "error": str(e)
            }


async def validate_cohort(state: CohortGraphState) -> CohortGraphState:
    """
    5단계: 검증 결과를 CohortGraphState에 반영
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        CohortGraphState: 검증 결과가 포함된 상태
    """
    logger = get_logger()
    logger.info("Starting BM25-based cohort validation process...")
    
    try:
        validator = CohortValidator()
        
        # 검증할 코호트 정보 준비
        cohort_info = ""
        if state.get("cohorts_json"):
            # JSON 형태의 코호트 정보를 문자열로 변환
            cohort_info = str(state["cohorts_json"])
        elif state.get("cohorts_markdown"):
            # Markdown 형태의 코호트 정보 사용
            cohort_info = "\n".join(state["cohorts_markdown"])
        else:
            logger.error("No cohort information found in state")
            state["is_valid"] = False
            return state
        
        reference_number = state.get("source_reference_number", "")
        guideline_content = state.get("source_contents", "")
        
        # 1단계: 검증 질문 생성
        logger.info("Step 1: Generating validation questions...")
        questions_response = await validator.generate_validation_questions(
            cohort_info, reference_number
        )
        
        # 2단계: 질문 추출
        logger.info("Step 2: Extracting questions from response...")
        questions = validator.extract_questions_from_response(questions_response)
        
        if not questions:
            logger.error("No questions extracted from LLM response")
            state["is_valid"] = False
            return state
        
        # 3단계 & 4단계: 각 질문에 대해 BM25 기반 증거 검색 및 NLI 평가
        logger.info("Step 3&4: Searching evidence with BM25 and evaluating with NLI...")
        all_nli_results = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            
            # BM25 기반 증거 검색
            evidence_list = validator.search_evidence_in_guideline(
                question, guideline_content, reference_number
            )
            
            # 각 질문에 대한 증거들을 함께 평가
            if evidence_list:
                verification_results = validator.verify_claim_with_evidence(question, evidence_list)
                
                # 질문 정보 추가
                for result in verification_results:
                    result["question"] = question
                    all_nli_results.append(result)
        
        # 5단계: 최종 검증 결과 계산
        logger.info("Step 5: Calculating final validation score...")
        is_valid, confidence_score, details = validator.calculate_validation_score(all_nli_results)
        
        # 상태 업데이트
        state["is_valid"] = is_valid
        state["validation_details"] = {
            "confidence_score": confidence_score,
            "nli_results": all_nli_results,
            "questions_generated": questions,
            "validation_summary": details,
            "method": "BM25_based"
        }
        logger.info(f"BM25-based validation completed. Result: {is_valid}, Confidence: {confidence_score:.3f}")

        return state
        
    except Exception as e:
        logger.error(f"Error during cohort validation: {e}")
        state["is_valid"] = False
        state["validation_details"] = {"error": str(e), "method": "BM25_based"}
        return state