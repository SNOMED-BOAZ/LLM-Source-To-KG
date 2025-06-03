from llm_source_to_kg.schema.state import *
from typing import List, Dict, Any, Annotated
import operator


class AnalysisGraphState(TypedDict):
    context: Annotated[str, operator.add]
    question: Annotated[str, 'analysis extraction prompt']
    answer: Annotated[str, 'llm answer']

    is_valid: Annotated[bool, 'whether analysis validation was successful']
    validation_feedback: Annotated[str, 'feedback for valid analysis']
    retries: Annotated[int, 'retry count']
    retry_analysis: Annotated[List[Dict[str, Any]], 'retry target analysis']

    source_reference_number: Annotated[str, 'NICE Guideline reference Number']
    source_contents: Annotated[str, 'NICE Guideline contents']

    cohort: Annotated[str, 'cohort markdown to analyze']
    analysis: Annotated[Dict[str, Any], 'analysis result']
