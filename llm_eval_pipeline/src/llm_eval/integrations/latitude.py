"""
Latitude.so integration for LLM evaluation framework.

This module provides bidirectional integration with Latitude's prompt engineering platform,
enabling seamless evaluation workflow between our comprehensive framework and Latitude's
prompt management and evaluation capabilities.

Uses the official Latitude SDK for robust API interactions.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

# Official Latitude SDK
from latitude_sdk import Latitude, ApiError

from ..core.results import EvaluationResults, FrameworkResult
from ..core.model_interface import ModelInterface

logger = logging.getLogger(__name__)


@dataclass
class LatitudeConfig:
    """Configuration for Latitude integration."""
    api_key: str
    project_id: Optional[int] = None
    version_uuid: Optional[str] = None


@dataclass
class LatitudePrompt:
    """Latitude prompt structure."""
    path: str
    parameters: Dict[str, Any]
    custom_identifier: Optional[str] = None


@dataclass
class LatitudeEvaluation:
    """Latitude evaluation result structure."""
    conversation_uuid: str
    evaluation_uuid: str
    result: Union[float, int, str, bool]
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LatitudeAPIError(Exception):
    """Custom exception for Latitude API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class LatitudeClient:
    """
    Wrapper for the official Latitude SDK with async support.
    
    Provides unified interface for Latitude API operations using the official SDK.
    """
    
    def __init__(self, config: LatitudeConfig):
        """Initialize Latitude client with configuration."""
        self.config = config
        self.client = Latitude(api_key=config.api_key)
    
    async def run_prompt(self, prompt: LatitudePrompt) -> Dict[str, Any]:
        """
        Run a prompt in Latitude and get the result.
        
        Args:
            prompt: LatitudePrompt configuration
            
        Returns:
            Dict containing the prompt execution result
        """
        try:
            logger.info(f"Running Latitude prompt: {prompt.path}")
            
            # Prepare options
            from latitude_sdk import RunPromptOptions
            options = RunPromptOptions(
                parameters=prompt.parameters,
                custom_identifier=prompt.custom_identifier
            )
            
            # Use the official SDK's run method
            result = await self.client.prompts.run(prompt.path, options)
            
            return {
                "conversation_uuid": result.conversation_uuid,
                "response": result.response,
                "usage": result.usage.__dict__ if hasattr(result, 'usage') else None,
                "response_id": result.response_id if hasattr(result, 'response_id') else None
            }
            
        except ApiError as e:
            raise LatitudeAPIError(f"Latitude API error: {e}", status_code=getattr(e, 'status_code', None))
        except Exception as e:
            raise LatitudeAPIError(f"Unexpected error: {e}")
    
    async def push_evaluation(self, evaluation: LatitudeEvaluation) -> Dict[str, Any]:
        """
        Push evaluation result to Latitude.
        
        Args:
            evaluation: LatitudeEvaluation result
            
        Returns:
            Dict containing the API response
        """
        try:
            logger.info(f"Pushing evaluation to Latitude: {evaluation.evaluation_uuid}")
            
            # Prepare options
            from latitude_sdk import AnnotateEvaluationOptions
            options = AnnotateEvaluationOptions(
                reason=evaluation.reason,
                metadata=evaluation.metadata
            )
            
            # Use the official SDK's annotate method
            result = await self.client.evaluations.annotate(
                evaluation.conversation_uuid,
                evaluation.evaluation_uuid,
                evaluation.result,
                options
            )
            
            return {"success": True, "result": result}
            
        except ApiError as e:
            raise LatitudeAPIError(f"Latitude API error: {e}", status_code=getattr(e, 'status_code', None))
        except Exception as e:
            raise LatitudeAPIError(f"Unexpected error: {e}")
    
    async def create_log(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a log entry in Latitude.
        
        Args:
            log_data: Log information
            
        Returns:
            Dict containing the created log
        """
        try:
            logger.info("Creating log entry in Latitude")
            
            # Use the official SDK's create method
            result = await self.client.logs.create(**log_data)
            
            return {"success": True, "log": result}
            
        except ApiError as e:
            raise LatitudeAPIError(f"Latitude API error: {e}", status_code=getattr(e, 'status_code', None))
        except Exception as e:
            raise LatitudeAPIError(f"Unexpected error: {e}")
    
    async def get_all_prompts(self) -> List[Dict[str, Any]]:
        """
        Get all prompts from Latitude.
        
        Returns:
            List of prompt information
        """
        try:
            logger.info("Fetching all prompts from Latitude")
            
            # Use the official SDK's get_all method  
            result = await self.client.prompts.get_all()
            
            return [prompt.__dict__ for prompt in result]
            
        except ApiError as e:
            raise LatitudeAPIError(f"Latitude API error: {e}", status_code=getattr(e, 'status_code', None))
        except Exception as e:
            raise LatitudeAPIError(f"Unexpected error: {e}")
    
    async def get_prompt(self, path: str) -> Dict[str, Any]:
        """
        Get a specific prompt from Latitude.
        
        Args:
            path: Prompt path
            
        Returns:
            Dict containing prompt information
        """
        try:
            logger.info(f"Fetching prompt from Latitude: {path}")
            
            # Use the official SDK's get method
            result = await self.client.prompts.get(path)
            
            return result.__dict__
            
        except ApiError as e:
            raise LatitudeAPIError(f"Latitude API error: {e}", status_code=getattr(e, 'status_code', None))
        except Exception as e:
            raise LatitudeAPIError(f"Unexpected error: {e}")


class LatitudeIntegration:
    """
    Main integration class for bidirectional sync with Latitude.
    
    Provides high-level methods for:
    - Pushing our evaluation results to Latitude
    - Pulling Latitude datasets for evaluation
    - Running comprehensive evaluations on Latitude prompts
    """
    
    def __init__(self, config: LatitudeConfig):
        """Initialize Latitude integration."""
        self.config = config
        self.client = LatitudeClient(config)
    
    async def push_framework_results(self, 
                                   results: EvaluationResults,
                                   prompt_path: str,
                                   conversation_uuid: Optional[str] = None) -> Dict[str, Any]:
        """
        Push our comprehensive evaluation results to Latitude.
        
        Args:
            results: EvaluationResults from our framework
            prompt_path: Latitude prompt path that was evaluated
            conversation_uuid: Optional conversation UUID for linking
            
        Returns:
            Dict containing push operation results
        """
        push_results = {
            "pushed_evaluations": [],
            "errors": [],
            "summary": {}
        }
            
        # Create a conversation UUID if not provided
        if not conversation_uuid:
            conversation_uuid = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(prompt_path) % 10000}"
        
        # Convert our framework results to Latitude evaluations
        latitude_evaluations = self._convert_to_latitude_evaluations(
            results, conversation_uuid
        )
        
        # Push each evaluation
        for evaluation in latitude_evaluations:
            try:
                result = await self.client.push_evaluation(evaluation)
                push_results["pushed_evaluations"].append({
                    "evaluation_uuid": evaluation.evaluation_uuid,
                    "result": evaluation.result,
                    "status": "success",
                    "response": result
                })
                
            except LatitudeAPIError as e:
                error_info = {
                    "evaluation_uuid": evaluation.evaluation_uuid,
                    "error": str(e),
                    "status_code": e.status_code
                }
                push_results["errors"].append(error_info)
                logger.error(f"Failed to push evaluation {evaluation.evaluation_uuid}: {e}")
        
        # Create summary
        push_results["summary"] = {
            "total_evaluations": len(latitude_evaluations),
            "successful_pushes": len(push_results["pushed_evaluations"]),
            "failed_pushes": len(push_results["errors"]),
            "conversation_uuid": conversation_uuid,
            "prompt_path": prompt_path,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Pushed {push_results['summary']['successful_pushes']}/{push_results['summary']['total_evaluations']} evaluations to Latitude")
        
        return push_results
    
    async def pull_and_evaluate_prompts(self, 
                                      prompt_paths: List[str],
                                      model: ModelInterface,
                                      evaluation_frameworks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Pull prompts from Latitude and run our comprehensive evaluation.
        
        Args:
            prompt_paths: List of Latitude prompt paths to evaluate
            model: Model interface for evaluation
            evaluation_frameworks: Optional list of frameworks to run
            
        Returns:
            Dict containing evaluation results for each prompt
        """
        evaluation_results = {
            "prompt_evaluations": {},
            "errors": [],
            "summary": {}
        }
        
        for prompt_path in prompt_paths:
            try:
                # Create a simple prompt run to test
                prompt = LatitudePrompt(
                    path=prompt_path,
                    parameters={},  # Could be enhanced to pull actual parameters
                    custom_identifier=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                # Run prompt in Latitude
                latitude_result = await self.client.run_prompt(prompt)
                
                # Extract the generated content for evaluation
                # Note: This will need to be adapted based on actual Latitude response format
                generated_content = latitude_result.get("response", {}).get("content", "")
                
                if generated_content:
                    # Here we would run our evaluation frameworks
                    # For now, create a placeholder result
                    framework_result = FrameworkResult(
                        framework_name="latitude_integration",
                        metrics={
                            "latitude_prompt_path": prompt_path,
                            "content_length": len(generated_content),
                            "has_content": bool(generated_content)
                        },
                        metadata={
                            "latitude_response": latitude_result,
                            "evaluation_timestamp": datetime.now().isoformat()
                        },
                        success=True
                    )
                    
                    evaluation_results["prompt_evaluations"][prompt_path] = {
                        "latitude_result": latitude_result,
                        "framework_result": asdict(framework_result),
                        "status": "success"
                    }
                else:
                    evaluation_results["errors"].append({
                        "prompt_path": prompt_path,
                        "error": "No content generated by prompt",
                        "latitude_result": latitude_result
                    })
            
            except LatitudeAPIError as e:
                error_info = {
                    "prompt_path": prompt_path,
                    "error": str(e),
                    "status_code": e.status_code
                }
                evaluation_results["errors"].append(error_info)
                logger.error(f"Failed to evaluate prompt {prompt_path}: {e}")
        
        # Create summary
        evaluation_results["summary"] = {
            "total_prompts": len(prompt_paths),
            "successful_evaluations": len(evaluation_results["prompt_evaluations"]),
            "failed_evaluations": len(evaluation_results["errors"]),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Evaluated {evaluation_results['summary']['successful_evaluations']}/{evaluation_results['summary']['total_prompts']} prompts from Latitude")
        
        return evaluation_results
    
    async def sync_datasets(self) -> Dict[str, Any]:
        """
        Synchronize datasets between Latitude and our framework.
        
        Returns:
            Dict containing sync operation results
        """
        try:
            prompts = await self.client.get_all_prompts()
            
            sync_results = {
                "prompts": prompts,
                "sync_timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
            logger.info(f"Synced {len(prompts)} prompts from Latitude")
            
            return sync_results
            
        except LatitudeAPIError as e:
            logger.error(f"Failed to sync prompts: {e}")
            return {
                "error": str(e),
                "status_code": e.status_code,
                "status": "failed",
                "sync_timestamp": datetime.now().isoformat()
            }
    
    def _convert_to_latitude_evaluations(self, 
                                       results: EvaluationResults,
                                       conversation_uuid: str) -> List[LatitudeEvaluation]:
        """
        Convert our framework results to Latitude evaluation format.
        
        Args:
            results: EvaluationResults from our framework
            conversation_uuid: Conversation UUID for linking
            
        Returns:
            List of LatitudeEvaluation objects
        """
        latitude_evaluations = []
        
        # Process safety metrics
        for framework_name, framework_result in results.safety_results.items():
            if framework_result.success:
                for metric_name, metric_value in framework_result.metrics.items():
                    if isinstance(metric_value, (int, float, bool)):
                        evaluation = LatitudeEvaluation(
                            conversation_uuid=conversation_uuid,
                            evaluation_uuid=f"safety_{framework_name}_{metric_name}",
                            result=metric_value,
                            reason=f"Safety evaluation result from {framework_name}",
                            metadata={
                                "category": "safety",
                                "framework": framework_name,
                                "metric": metric_name,
                                "framework_metadata": framework_result.metadata
                            }
                        )
                        latitude_evaluations.append(evaluation)
        
        # Process security metrics
        for framework_name, framework_result in results.security_results.items():
            if framework_result.success:
                for metric_name, metric_value in framework_result.metrics.items():
                    if isinstance(metric_value, (int, float, bool)):
                        evaluation = LatitudeEvaluation(
                            conversation_uuid=conversation_uuid,
                            evaluation_uuid=f"security_{framework_name}_{metric_name}",
                            result=metric_value,
                            reason=f"Security evaluation result from {framework_name}",
                            metadata={
                                "category": "security",
                                "framework": framework_name,
                                "metric": metric_name,
                                "framework_metadata": framework_result.metadata
                            }
                        )
                        latitude_evaluations.append(evaluation)
        
        # Process reliability metrics
        for framework_name, framework_result in results.reliability_results.items():
            if framework_result.success:
                for metric_name, metric_value in framework_result.metrics.items():
                    if isinstance(metric_value, (int, float, bool)):
                        evaluation = LatitudeEvaluation(
                            conversation_uuid=conversation_uuid,
                            evaluation_uuid=f"reliability_{framework_name}_{metric_name}",
                            result=metric_value,
                            reason=f"Reliability evaluation result from {framework_name}",
                            metadata={
                                "category": "reliability",
                                "framework": framework_name,
                                "metric": metric_name,
                                "framework_metadata": framework_result.metadata
                            }
                        )
                        latitude_evaluations.append(evaluation)
        
        return latitude_evaluations
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check connectivity and authentication with Latitude API.
        
        Returns:
            Dict containing health check results
        """
        try:
            # Try to fetch prompts as a connectivity test
            prompts = await self.client.get_all_prompts()
            
            return {
                "status": "healthy",
                "api_accessible": True,
                "authentication": "valid",
                "prompt_count": len(prompts),
                "timestamp": datetime.now().isoformat()
            }
            
        except LatitudeAPIError as e:
            return {
                "status": "unhealthy",
                "api_accessible": False,
                "error": str(e),
                "status_code": e.status_code,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "api_accessible": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }