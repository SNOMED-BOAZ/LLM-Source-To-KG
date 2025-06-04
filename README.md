# BOAZ-SNUH LLM Source to Knowledge Graph

이 프로젝트에 대한 협업 규칙은 [CONVENTION.md](./CONVENTION.md)를 참조하세요.

## 프로젝트 개요

이 프로젝트는 LLM(Large Language Model)을 활용하여 의료 가이드라인과 OMOP 데이터를 분석하고 지식 그래프(Knowledge Graph)를 생성합니다. LangGraph를 사용한 워크플로우 기반 아키텍처로 구성되어 있습니다.

## 환경 설정 가이드

이 프로젝트는 Python 3.11 이상 및 Poetry를 사용하여 의존성을 관리합니다. 아래 단계를 따라 개발 환경을 설정하세요.

### 사전 요구사항

- Python 3.11 이상
- Poetry 설치

### 1. Poetry 설치

#### macOS / Linux
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### Windows
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

### 2. 프로젝트 클론

```bash
git clone https://github.com/GU-0/BOAZ-SNUH_llm_source_to_kg.git
cd BOAZ-SNUH_llm_source_to_kg
```

### 3. Poetry 환경 설정

```bash
# Poetry 가상 환경 생성 및 의존성 설치
poetry install

# 가상 환경 활성화
poetry shell
```

### 4. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음 환경 변수들을 설정하세요:

```bash
# .env 파일
GEMINI_API_KEY=your_gemini_api_key_here
ES_SERVER_HOST=your_elasticsearch_host
ES_SERVER_PORT=your_elasticsearch_port
GRPC_SERVER_PORT=your_grpc_server_port
AWS_PROFILE=boaz-snuh
AWS_REGION=ap-northeast-2
AWS_S3_BUCKET=source-to-kg
```

## 프로젝트 실행 방법

Poetry에 등록된 스크립트를 사용하여 다양한 기능을 실행할 수 있습니다:

### 1. LLM 테스트
```bash
# Gemini LLM 연결 테스트
poetry run test-gemini

# 로깅 시스템 테스트
poetry run test-logger
```

### 2. 그래프 워크플로우 실행
```bash
# 코호트 그래프 워크플로우 실행
poetry run test-cohort-graph

# 분석 그래프 워크플로우 실행
poetry run test-analysis-graph

# 코호트 검증 워크플로우 실행
poetry run test-cohort-validation
```

## 프로젝트 구조

```
.
├── pyproject.toml         # Poetry 프로젝트 설정 파일
├── CONVENTION.md          # 협업 규칙 및 컨벤션 문서
├── .env                   # 환경 변수 설정 파일 (생성 필요)
├── prompts/               # LLM 프롬프트 템플릿 디렉토리
│   └── sample.txt         # 샘플 프롬프트 파일
├── outputs/               # 실행 결과 출력 디렉토리
├── test_outputs/          # 테스트 결과 출력 디렉토리
├── logs/                  # 로그 파일 디렉토리
├── src/                   # 소스 코드
│   └── llm_source_to_kg/  # 메인 패키지
│       ├── __init__.py    # 패키지 초기화 파일
│       ├── config.py      # 프로젝트 설정 파일
│       ├── graph/         # LangGraph 워크플로우
│       │   ├── orchestrator.py     # 전체 워크플로우 조율
│       │   ├── cohort_graph/       # 코호트 그래프 워크플로우
│       │   │   ├── __init__.py
│       │   │   ├── orchestrator.py # 코호트 그래프 오케스트레이터
│       │   │   ├── state.py        # 코호트 그래프 상태 관리
│       │   │   ├── utils.py        # 코호트 그래프 유틸리티
│       │   │   ├── prompts/        # 코호트 전용 프롬프트
│       │   │   └── nodes/          # 코호트 그래프 노드 구현
│       │   └── analysis_graph/     # 분석 그래프 워크플로우
│       │       ├── __init__.py
│       │       ├── orchestrator.py # 분석 그래프 오케스트레이터
│       │       ├── state.py        # 분석 그래프 상태 관리
│       │       └── nodes/          # 분석 그래프 노드 구현
│       │           ├── mapping_to_omop.py  # OMOP 매핑 노드
│       │           └── load_to_kg.py       # 지식 그래프 로딩 노드
│       ├── llm/           # LLM 관련 모듈
│       │   ├── __init__.py
│       │   ├── common_llm_interface.py  # LLM 공통 인터페이스
│       │   └── gemini.py               # Gemini 모델 구현
│       ├── schema/        # 데이터 스키마 정의
│       │   ├── state.py   # 상태 스키마
│       │   ├── llm.py     # LLM 스키마
│       │   └── validate/  # 검증 스키마
│       ├── utils/         # 유틸리티 함수
│       │   ├── util.py    # 일반 유틸리티
│       │   └── grpc_asset/ # gRPC 관련 유틸리티
│       └── test/          # 테스트 모듈
│           ├── test_gemini.py           # Gemini LLM 테스트
│           ├── test_logger.py           # 로깅 테스트
│           └── test_cohort_validation.py # 코호트 검증 테스트
└── datasets/              # 데이터셋 디렉토리
    ├── omop/              # OMOP 데이터셋
    ├── guideline/         # 가이드라인 데이터셋
    └── csv_to_db.py       # CSV를 데이터베이스로 변환하는 스크립트
```

## 주요 기능

### 1. 코호트 그래프 워크플로우
- 의료 가이드라인을 분석하여 환자 코호트를 생성
- LLM을 활용한 자연어 처리 및 구조화

### 2. 분석 그래프 워크플로우
- 코호트 데이터를 OMOP CDM에 매핑
- 지식 그래프로 변환 및 저장

### 3. LLM 통합
- Gemini AI 모델 통합
- 확장 가능한 LLM 인터페이스 설계

### 4. 데이터 처리
- Elasticsearch 연동
- AWS S3 연동
- gRPC 서버 통신

## 개발 환경

- **Python**: 3.11+
- **LangGraph**: 워크플로우 관리
- **Gemini AI**: LLM 모델
- **Elasticsearch**: 검색 엔진
- **AWS S3**: 클라우드 스토리지
- **Poetry**: 의존성 관리

## 로그 및 출력

- **로그**: `logs/` 디렉토리에 저장
- **실행 결과**: `outputs/` 디렉토리에 저장
- **테스트 결과**: `test_outputs/` 디렉토리에 저장