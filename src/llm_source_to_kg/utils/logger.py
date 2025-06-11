import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any

class Logger:
    """
    BOAZ-SNUH 프로젝트를 위한 로거 클래스
    다양한 로깅 레벨과 포맷을 지원합니다.
    """
    
    # 로그 레벨 상수 정의
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    # 기본 로그 포맷
    DEFAULT_FORMAT = "%(asctime)s %(levelname)s: %(message)s"
    
    # 현재 로그 디렉터리 (모든 로거가 공유)
    _current_log_dir = None
    
    # 이미 설정된 로거 이름 추적 (메모리 효율성을 위해)
    _configured_loggers = set()
    
    @classmethod
    def get_log_directory(cls) -> Path:
        """
        현재 로그 디렉터리를 반환합니다.
        없으면 새로 생성합니다.
        
        Returns:
            Path: 로그 디렉터리 경로
        """
        if cls._current_log_dir is None:
            # logs/MMDDHHMM 형식의 디렉터리 생성
            base_log_dir = Path("logs")
            base_log_dir.mkdir(exist_ok=True)
            
            date_str = datetime.now().strftime("%m%d%H%M")
            cls._current_log_dir = base_log_dir / date_str
            cls._current_log_dir.mkdir(exist_ok=True)
            
        return cls._current_log_dir
    
    @classmethod
    def remove_logger(cls, name: str):
        """
        특정 이름의 로거 설정을 제거합니다.
        
        Args:
            name: 제거할 로거 이름
        """
        if name in cls._configured_loggers:
            logger = logging.getLogger(name)
            
            # 모든 핸들러 닫기 및 제거
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            
            # 설정된 로거 목록에서 제거
            cls._configured_loggers.remove(name)
    
    def __init__(
        self,
        name: str = "main",
        level: int = logging.INFO,
        log_format: str = DEFAULT_FORMAT,
        log_file: Optional[Union[str, Path]] = None,
        console_output: bool = True,
        file_output: bool = True,
        propagate: bool = False,
    ):
        """
        로거 초기화
        
        Args:
            name: 로거 이름
            level: 로깅 레벨
            log_format: 로그 포맷
            log_file: 로그 파일 경로 (None인 경우 자동 생성)
            console_output: 콘솔 출력 여부
            file_output: 파일 출력 여부
            propagate: 상위 로거로 로그 전파 여부
        """
        # Python 내장 로깅 시스템의 로거를 가져옴 (이미 존재하면 재사용)
        self.logger = logging.getLogger(name)
        self.name = name
        
        # 이미 설정된 로거인지 확인 (중복 설정 방지)
        if name not in self._configured_loggers:
            self.logger.setLevel(level)
            self.logger.propagate = propagate  # 로그 전파 설정
            # 기존 핸들러 제거 (첫 설정 시에만)
            self.logger.handlers = []
            
            # 로그 포맷 설정
            formatter = logging.Formatter(log_format)
            
            # 콘솔 출력 핸들러 추가
            if console_output:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
            
            # 파일 출력 핸들러 추가
            if file_output:
                if log_file is None:
                    # 로그 파일 자동 생성 (logs/MMDDHHMM/name.log 형식)
                    log_dir = self.get_log_directory()
                    log_file = log_dir / f"{name}.log"
                
                # 경로 객체를 문자열로 변환
                if isinstance(log_file, Path):
                    log_file = str(log_file)
                    
                # 로그 파일 디렉토리 생성
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            
            # 설정 완료된 로거 이름 추가
            self._configured_loggers.add(name)
    
    def debug(self, msg: str, *args, **kwargs):
        """디버그 레벨 로그 기록"""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """정보 레벨 로그 기록"""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """경고 레벨 로그 기록"""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """에러 레벨 로그 기록"""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """치명적 에러 레벨 로그 기록"""
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, exc_info=True, **kwargs):
        """예외 정보와 함께 에러 로그 기록"""
        self.logger.exception(msg, *args, exc_info=exc_info, **kwargs)
    
    def log_dict(self, data: Dict[str, Any], level: int = logging.INFO, prefix: str = ""):
        """
        사전 형태의 데이터를 로깅합니다.
        
        Args:
            data: 로깅할 사전 데이터
            level: 로깅 레벨
            prefix: 로그 메시지 접두사
        """
        for key, value in data.items():
            log_msg = f"{prefix}{key}: {value}"
            self.logger.log(level, log_msg)
    
    def set_level(self, level: int):
        """로깅 레벨 설정"""
        self.logger.setLevel(level)
    
    def add_file_handler(self, log_file: Union[str, Path], log_format: Optional[str] = None):
        """
        파일 핸들러 추가
        
        Args:
            log_file: 로그 파일 경로
            log_format: 로그 포맷 (None인 경우 기본 포맷 사용)
        """
        if isinstance(log_file, Path):
            log_file = str(log_file)
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        formatter = logging.Formatter(log_format or self.DEFAULT_FORMAT)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def close(self):
        """
        로거의 모든 핸들러를 닫고 정리합니다.
        로거 사용이 완료되면 호출하여 리소스를 정리합니다.
        """
        Logger.remove_logger(self.name)


# 로거 캐시 (메모리 효율성을 위해 이름별로 로거 인스턴스 캐싱)
_logger_cache = {}

def get_logger(
    name: str = "main",
    level: int = logging.INFO,
    log_format: str = Logger.DEFAULT_FORMAT,
    log_file: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    file_output: bool = True,
    force_new: bool = False,
    propagate: bool = False,
) -> Logger:
    """
    이름에 해당하는 로거 인스턴스를 가져옵니다.
    이름이 다르면 각각 다른 로거 인스턴스가 생성됩니다.
    
    Args:
        name: 로거 이름 (logs/MMDDHHMM/name.log 파일명으로 사용됨)
        level: 로깅 레벨
        log_format: 로그 포맷
        log_file: 로그 파일 경로 (None인 경우 자동 생성)
        console_output: 콘솔 출력 여부
        file_output: 파일 출력 여부 (기본값 True로 변경)
        force_new: 새 로거 인스턴스 강제 생성 여부
        propagate: 상위 로거로 로그 전파 여부
    
    Returns:
        Logger 인스턴스
    """
    global _logger_cache
    
    # 캐시에 없거나 강제 재생성 요청이면 새로 생성
    if name not in _logger_cache or force_new:
        _logger_cache[name] = Logger(
            name=name,
            level=level,
            log_format=log_format,
            log_file=log_file,
            console_output=console_output,
            file_output=file_output,
            propagate=propagate,
        )
    
    return _logger_cache[name]


def remove_logger(name: str):
    """
    특정 이름의 로거 인스턴스를 제거합니다.
    로거의 핸들러를 정리하고 캐시에서 제거합니다.
    
    Args:
        name: 제거할 로거 이름
    """
    global _logger_cache
    
    # 로거 설정 제거
    Logger.remove_logger(name)
    
    # 캐시에서 제거
    if name in _logger_cache:
        del _logger_cache[name] 