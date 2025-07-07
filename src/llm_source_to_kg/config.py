from dotenv import load_dotenv
import os


load_dotenv()

class Config:
    """설정 클래스 - 속성 접근 방식 사용"""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")

    ES_SERVER_HOST = os.getenv("ES_SERVER_HOST")
    if not ES_SERVER_HOST:
        raise ValueError("ES_SERVER_HOST is not set")

    ES_SERVER_PORT = os.getenv("ES_SERVER_PORT")
    if not ES_SERVER_PORT:
        raise ValueError("ES_SERVER_PORT is not set")

    ES_SERVER_USERNAME = os.getenv("ES_SERVER_USERNAME")
    if not ES_SERVER_USERNAME:
        raise ValueError("ES_SERVER_USERNAME is not set")

    ES_SERVER_PASSWORD = os.getenv("ES_SERVER_PASSWORD")
    if not ES_SERVER_PASSWORD:
        raise ValueError("ES_SERVER_PASSWORD is not set")

    GRPC_SERVER_PORT = os.getenv("GRPC_SERVER_PORT")
    if not GRPC_SERVER_PORT:
        raise ValueError("GRPC_SERVER_PORT is not set")
    
    NEO4J_SERVER_URI = os.getenv("NEO4J_SERVER_URI")
    if not NEO4J_SERVER_URI:
        raise ValueError("NEO4J_SERVER_URI is not set")
    
    NEO4J_SERVER_USER = os.getenv("NEO4J_SERVER_USER")
    if not NEO4J_SERVER_USER:
        raise ValueError("NEO4J_SERVER_USER is not set")
    
    NEO4J_SERVER_PASSWORD = os.getenv("NEO4J_SERVER_PASSWORD")
    if not NEO4J_SERVER_PASSWORD:
        raise ValueError("NEO4J_SERVER_PASSWORD is not set")
    
    NEO4J_SERVER_DATABASE = os.getenv("NEO4J_SERVER_DATABASE")
    if not NEO4J_SERVER_DATABASE:
        raise ValueError("NEO4J_SERVER_DATABASE is not set")

    
    # AWS S3 관련 설정
    AWS_PROFILE = os.getenv("AWS_PROFILE", "boaz-snuh")  # 기본값 boaz-snuh 프로필    
    if not AWS_PROFILE:
        raise ValueError("AWS_PROFILE is not set")
    AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")  # 기본값 서울 리전
    AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "source-to-kg")  # 기본 버킷명 (필요시 사용)


# 전역 설정 인스턴스 생성
config = Config()

# 다음과 같이 사용 가능:
# from llm_source_to_kg.config import config
# value = config.GEMINI_API_KEY  # 속성 접근 방식
