"""
LLM Comprehensive Evaluation Pipeline

A modular, extensible system for evaluating LLMs across safety, security, and reliability metrics.
"""

from .core.config import EvaluationConfig, RiskLevel
from .core.orchestrator import EvaluationOrchestrator
from .core.model_interface import ModelInterface, OpenAIModel, AnthropicModel, GroqModel, LocalModel, create_model
from .core.results import EvaluationResults

# Research dataset evaluators
from .frameworks.safety.agentharm_hf import AgentHarmHFEvaluator
from .frameworks.safety.safetybench_hf import SafetyBenchHFEvaluator
from .frameworks.reliability.promptrobust_hf import PromptRobustHFEvaluator

__version__ = "0.1.0"
__all__ = [
    "EvaluationConfig",
    "RiskLevel", 
    "EvaluationOrchestrator",
    "ModelInterface",
    "OpenAIModel",
    "AnthropicModel", 
    "GroqModel",
    "LocalModel",
    "create_model",
    "EvaluationResults",
    "AgentHarmHFEvaluator",
    "SafetyBenchHFEvaluator",
    "PromptRobustHFEvaluator"
]