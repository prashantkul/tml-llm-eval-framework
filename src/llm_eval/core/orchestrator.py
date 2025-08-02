"""Main evaluation orchestrator."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from .config import EvaluationConfig
from .model_interface import ModelInterface
from .results import EvaluationResults, FrameworkResult

# Import framework evaluators
from ..frameworks.safety.agentharm import AgentHarmEvaluator
from ..frameworks.safety.agent_safetybench import AgentSafetyBenchEvaluator
from ..frameworks.security.houyi import HouYiAttackEvaluator
from ..frameworks.security.cia_attacks import CIAAttackEvaluator
from ..frameworks.robustness.autoevoeval import AutoEvoEvalEvaluator
from ..frameworks.robustness.promptrobust import PromptRobustEvaluator
from ..frameworks.robustness.selfprompt import SelfPromptEvaluator

logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """Orchestrates comprehensive LLM evaluation across multiple frameworks."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.safety_evaluators = {}
        self.security_evaluators = {}
        self.reliability_evaluators = {}
        
        self.initialize_frameworks()
    
    def initialize_frameworks(self) -> None:
        """Initialize all enabled evaluation frameworks."""
        logger.info("Initializing evaluation frameworks...")
        
        # Initialize safety frameworks
        if self.config.enable_safety:
            try:
                self.safety_evaluators['agentharm'] = AgentHarmEvaluator(self.config)
                logger.info("AgentHarm evaluator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize AgentHarm: {e}")
            
            try:
                self.safety_evaluators['agent_safetybench'] = AgentSafetyBenchEvaluator(self.config)
                logger.info("Agent-SafetyBench evaluator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Agent-SafetyBench: {e}")
        
        # Initialize security frameworks
        if self.config.enable_security:
            try:
                self.security_evaluators['houyi'] = HouYiAttackEvaluator(self.config)
                logger.info("HouYi evaluator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize HouYi: {e}")
            
            try:
                self.security_evaluators['cia_attacks'] = CIAAttackEvaluator(self.config)
                logger.info("CIA Attack evaluator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize CIA Attacks: {e}")
        
        # Initialize reliability frameworks
        if self.config.enable_reliability:
            try:
                self.reliability_evaluators['autoevoeval'] = AutoEvoEvalEvaluator(self.config)
                logger.info("AutoEvoEval evaluator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize AutoEvoEval: {e}")
            
            try:
                self.reliability_evaluators['promptrobust'] = PromptRobustEvaluator(self.config)
                logger.info("PromptRobust evaluator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PromptRobust: {e}")
            
            try:
                self.reliability_evaluators['selfprompt'] = SelfPromptEvaluator(self.config)
                logger.info("SelfPrompt evaluator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize SelfPrompt: {e}")
        
        total_frameworks = (len(self.safety_evaluators) + 
                          len(self.security_evaluators) + 
                          len(self.reliability_evaluators))
        logger.info(f"Initialized {total_frameworks} evaluation frameworks")
    
    async def run_evaluation(self, 
                           model: ModelInterface, 
                           datasets: Optional[List[str]] = None) -> EvaluationResults:
        """Run comprehensive evaluation across all enabled frameworks."""
        logger.info(f"Starting evaluation for model: {model.model_name}")
        start_time = time.time()
        
        # Initialize results
        results = EvaluationResults(
            model_name=model.model_name,
            timestamp=datetime.now(),
            config=self.config
        )
        
        # Run evaluations concurrently by category
        evaluation_tasks = []
        
        if self.config.enable_safety and self.safety_evaluators:
            evaluation_tasks.append(self._run_safety_evaluation(model))
        
        if self.config.enable_security and self.security_evaluators:
            evaluation_tasks.append(self._run_security_evaluation(model))
        
        if self.config.enable_reliability and self.reliability_evaluators:
            evaluation_tasks.append(self._run_reliability_evaluation(model, datasets))
        
        # Execute all evaluations
        if evaluation_tasks:
            category_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(category_results):
                if isinstance(result, Exception):
                    logger.error(f"Evaluation category {i} failed: {result}")
                    continue
                
                category, framework_results = result
                if category == "safety":
                    results.safety_results = framework_results
                elif category == "security":
                    results.security_results = framework_results
                elif category == "reliability":
                    results.reliability_results = framework_results
        
        # Record execution time
        results.total_execution_time = time.time() - start_time
        
        # Aggregate metrics and check compliance
        results.aggregate_metrics()
        
        # Record any framework failures
        results.framework_failures = self._get_framework_failures(results)
        
        logger.info(f"Evaluation completed in {results.total_execution_time:.2f}s")
        logger.info(f"Overall score: {results.overall_score:.3f}")
        logger.info(f"Risk compliance: {results.risk_compliance}")
        
        return results
    
    async def _run_safety_evaluation(self, model: ModelInterface) -> tuple:
        """Run all safety frameworks."""
        logger.info("Running safety evaluation...")
        framework_results = []
        
        # Run safety evaluators concurrently
        tasks = []
        for name, evaluator in self.safety_evaluators.items():
            tasks.append(self._run_framework_evaluation(name, evaluator, model))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            framework_results = [r for r in results if not isinstance(r, Exception)]
            
            # Log any failures
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Safety framework failed: {result}")
        
        logger.info(f"Safety evaluation completed with {len(framework_results)} frameworks")
        return "safety", framework_results
    
    async def _run_security_evaluation(self, model: ModelInterface) -> tuple:
        """Run all security frameworks."""
        logger.info("Running security evaluation...")
        framework_results = []
        
        # Run security evaluators concurrently
        tasks = []
        for name, evaluator in self.security_evaluators.items():
            tasks.append(self._run_framework_evaluation(name, evaluator, model))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            framework_results = [r for r in results if not isinstance(r, Exception)]
            
            # Log any failures
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Security framework failed: {result}")
        
        logger.info(f"Security evaluation completed with {len(framework_results)} frameworks")
        return "security", framework_results
    
    async def _run_reliability_evaluation(self, 
                                        model: ModelInterface, 
                                        datasets: Optional[List[str]]) -> tuple:
        """Run all reliability frameworks."""
        logger.info("Running reliability evaluation...")
        framework_results = []
        
        # Run reliability evaluators concurrently
        tasks = []
        for name, evaluator in self.reliability_evaluators.items():
            if hasattr(evaluator, 'evaluate') and hasattr(evaluator.evaluate, '__code__'):
                # Check if evaluator accepts datasets parameter
                if 'datasets' in evaluator.evaluate.__code__.co_varnames:
                    tasks.append(self._run_framework_evaluation_with_datasets(name, evaluator, model, datasets))
                else:
                    tasks.append(self._run_framework_evaluation(name, evaluator, model))
            else:
                tasks.append(self._run_framework_evaluation(name, evaluator, model))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            framework_results = [r for r in results if not isinstance(r, Exception)]
            
            # Log any failures
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Reliability framework failed: {result}")
        
        logger.info(f"Reliability evaluation completed with {len(framework_results)} frameworks")
        return "reliability", framework_results
    
    async def _run_framework_evaluation(self, 
                                      name: str, 
                                      evaluator: Any, 
                                      model: ModelInterface) -> FrameworkResult:
        """Run evaluation for a single framework."""
        logger.info(f"Running {name} evaluation...")
        start_time = time.time()
        
        try:
            # Apply timeout and retry logic
            for attempt in range(self.config.retry_attempts + 1):
                try:
                    metrics = await asyncio.wait_for(
                        evaluator.evaluate(model),
                        timeout=self.config.timeout_seconds
                    )
                    
                    execution_time = time.time() - start_time
                    
                    result = FrameworkResult(
                        framework_name=name,
                        metrics=metrics,
                        metadata=getattr(evaluator, 'get_metadata', lambda: {})(),
                        execution_time=execution_time,
                        success=True
                    )
                    
                    logger.info(f"{name} evaluation completed in {execution_time:.2f}s")
                    return result
                
                except asyncio.TimeoutError:
                    if attempt < self.config.retry_attempts:
                        logger.warning(f"{name} evaluation timed out, retrying ({attempt + 1}/{self.config.retry_attempts})")
                        await asyncio.sleep(1)  # Brief delay before retry
                    else:
                        raise
                except Exception as e:
                    if attempt < self.config.retry_attempts:
                        logger.warning(f"{name} evaluation failed, retrying ({attempt + 1}/{self.config.retry_attempts}): {e}")
                        await asyncio.sleep(1)
                    else:
                        raise
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{name} evaluation failed after {self.config.retry_attempts + 1} attempts: {e}")
            
            return FrameworkResult(
                framework_name=name,
                metrics={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    async def _run_framework_evaluation_with_datasets(self, 
                                                    name: str, 
                                                    evaluator: Any, 
                                                    model: ModelInterface,
                                                    datasets: Optional[List[str]]) -> FrameworkResult:
        """Run evaluation for a framework that supports datasets."""
        logger.info(f"Running {name} evaluation with datasets...")
        start_time = time.time()
        
        try:
            # Apply timeout and retry logic
            for attempt in range(self.config.retry_attempts + 1):
                try:
                    metrics = await asyncio.wait_for(
                        evaluator.evaluate(model, datasets),
                        timeout=self.config.timeout_seconds
                    )
                    
                    execution_time = time.time() - start_time
                    
                    result = FrameworkResult(
                        framework_name=name,
                        metrics=metrics,
                        metadata=getattr(evaluator, 'get_metadata', lambda: {})(),
                        execution_time=execution_time,
                        success=True
                    )
                    
                    logger.info(f"{name} evaluation completed in {execution_time:.2f}s")
                    return result
                
                except asyncio.TimeoutError:
                    if attempt < self.config.retry_attempts:
                        logger.warning(f"{name} evaluation timed out, retrying ({attempt + 1}/{self.config.retry_attempts})")
                        await asyncio.sleep(1)
                    else:
                        raise
                except Exception as e:
                    if attempt < self.config.retry_attempts:
                        logger.warning(f"{name} evaluation failed, retrying ({attempt + 1}/{self.config.retry_attempts}): {e}")
                        await asyncio.sleep(1)
                    else:
                        raise
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{name} evaluation failed after {self.config.retry_attempts + 1} attempts: {e}")
            
            return FrameworkResult(
                framework_name=name,
                metrics={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _get_framework_failures(self, results: EvaluationResults) -> List[str]:
        """Get list of failed frameworks."""
        failures = []
        
        for result in results.safety_results + results.security_results + results.reliability_results:
            if not result.success:
                failures.append(f"{result.framework_name}: {result.error_message}")
        
        return failures
    
    def get_available_frameworks(self) -> Dict[str, List[str]]:
        """Get list of available frameworks by category."""
        return {
            "safety": list(self.safety_evaluators.keys()),
            "security": list(self.security_evaluators.keys()),
            "reliability": list(self.reliability_evaluators.keys())
        }
    
    async def run_framework_subset(self, 
                                 model: ModelInterface,
                                 frameworks: Dict[str, List[str]],
                                 datasets: Optional[List[str]] = None) -> EvaluationResults:
        """Run evaluation on a subset of frameworks."""
        logger.info(f"Starting subset evaluation for model: {model.model_name}")
        start_time = time.time()
        
        # Initialize results
        results = EvaluationResults(
            model_name=model.model_name,
            timestamp=datetime.now(),
            config=self.config
        )
        
        # Run specified frameworks
        tasks = []
        
        # Safety frameworks
        if "safety" in frameworks:
            for framework_name in frameworks["safety"]:
                if framework_name in self.safety_evaluators:
                    tasks.append(("safety", self._run_framework_evaluation(
                        framework_name, self.safety_evaluators[framework_name], model)))
        
        # Security frameworks
        if "security" in frameworks:
            for framework_name in frameworks["security"]:
                if framework_name in self.security_evaluators:
                    tasks.append(("security", self._run_framework_evaluation(
                        framework_name, self.security_evaluators[framework_name], model)))
        
        # Reliability frameworks
        if "reliability" in frameworks:
            for framework_name in frameworks["reliability"]:
                if framework_name in self.reliability_evaluators:
                    evaluator = self.reliability_evaluators[framework_name]
                    if hasattr(evaluator.evaluate, '__code__') and 'datasets' in evaluator.evaluate.__code__.co_varnames:
                        tasks.append(("reliability", self._run_framework_evaluation_with_datasets(
                            framework_name, evaluator, model, datasets)))
                    else:
                        tasks.append(("reliability", self._run_framework_evaluation(
                            framework_name, evaluator, model)))
        
        # Execute all tasks
        if tasks:
            task_results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
            
            # Organize results by category
            for i, (category, _) in enumerate(tasks):
                result = task_results[i]
                if not isinstance(result, Exception):
                    if category == "safety":
                        results.safety_results.append(result)
                    elif category == "security":
                        results.security_results.append(result)
                    elif category == "reliability":
                        results.reliability_results.append(result)
        
        # Record execution time and aggregate results
        results.total_execution_time = time.time() - start_time
        results.aggregate_metrics()
        results.framework_failures = self._get_framework_failures(results)
        
        logger.info(f"Subset evaluation completed in {results.total_execution_time:.2f}s")
        return results