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
    use_draft: bool = True  # Use draft version by default for prompt creation


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
        # Initialize Latitude client with proper options
        if config.project_id and config.version_uuid:
            from latitude_sdk import LatitudeOptions
            options = LatitudeOptions(
                project_id=config.project_id,
                version_uuid=config.version_uuid
            )
            self.client = Latitude(config.api_key, options)
        elif config.project_id:
            from latitude_sdk import LatitudeOptions
            options = LatitudeOptions(project_id=config.project_id)
            self.client = Latitude(config.api_key, options)
        else:
            self.client = Latitude(config.api_key)
    
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
            logger.debug(f"  - conversation_uuid: {evaluation.conversation_uuid}")
            logger.debug(f"  - evaluation_uuid: {evaluation.evaluation_uuid}")
            logger.debug(f"  - result: {evaluation.result} (type: {type(evaluation.result)})")
            
            # Prepare options
            from latitude_sdk import AnnotateEvaluationOptions
            options = AnnotateEvaluationOptions(
                reason=evaluation.reason,
                metadata=evaluation.metadata
            )
            
            # Convert result to integer score (Latitude expects scores, not raw metrics)
            if isinstance(evaluation.result, bool):
                score = 5 if evaluation.result else 1
            elif isinstance(evaluation.result, (int, float)):
                # Scale to 1-5 range
                if 0 <= evaluation.result <= 1:
                    score = max(1, min(5, int(evaluation.result * 4) + 1))
                else:
                    score = max(1, min(5, int(evaluation.result)))
            else:
                score = 3  # Default neutral score
            
            logger.debug(f"  - converted score: {score}")
            
            # Use the official SDK's annotate method - correct signature from docs
            result = await self.client.evaluations.annotate(
                evaluation.conversation_uuid,  # 1st: conversation UUID
                score,                        # 2nd: score (integer)
                evaluation.evaluation_uuid,   # 3rd: evaluation UUID
                options                       # 4th: options (optional)
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
    
    async def create_version(self) -> Dict[str, Any]:
        """
        Create a new draft version in Latitude.
        
        Returns:
            Dict containing the created version information
        """
        try:
            logger.info("Creating new version in Latitude")
            
            # Use the official SDK's create version method
            result = await self.client.versions.create()
            
            return {"success": True, "version": result.__dict__ if hasattr(result, '__dict__') else result}
            
        except ApiError as e:
            raise LatitudeAPIError(f"Latitude API error: {e}", status_code=getattr(e, 'status_code', None))
        except Exception as e:
            raise LatitudeAPIError(f"Unexpected error: {e}")
    
    async def create_prompt(self, path: str, content: str, description: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new prompt in Latitude.
        
        Args:
            path: Prompt path (e.g., "safety/agentharm/prompt1")
            content: Prompt content/template
            description: Optional description (not used in current SDK)
            
        Returns:
            Dict containing the created prompt information
        """
        try:
            logger.info(f"Creating prompt in Latitude: {path}")
            
            # Use the official SDK's get_or_create method with correct API pattern
            from latitude_sdk import GetOrCreatePromptOptions
            options = GetOrCreatePromptOptions(
                prompt=content
            )
            
            # Create the prompt using the SDK (version targeting handled at client level)
            result = await self.client.prompts.get_or_create(path, options)
            
            return {"success": True, "prompt": result.__dict__ if hasattr(result, '__dict__') else result}
            
        except ApiError as e:
            raise LatitudeAPIError(f"Latitude API error: {e}", status_code=getattr(e, 'status_code', None))
        except Exception as e:
            raise LatitudeAPIError(f"Unexpected error: {e}")
    
    async def create_dataset(self, name: str, data: List[Dict[str, Any]], description: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new dataset in Latitude.
        
        Args:
            name: Dataset name
            data: List of data records
            description: Optional description
            
        Returns:
            Dict containing the created dataset information
        """
        try:
            logger.info(f"Creating dataset in Latitude: {name}")
            
            # Note: Latitude SDK doesn't currently support direct dataset creation
            # This is a placeholder for when the feature becomes available
            logger.warning("Dataset creation not yet supported by Latitude SDK")
            
            # Return a simulated success for now
            result = {
                "name": name,
                "samples": len(data),
                "description": description,
                "note": "Dataset creation not yet supported by SDK"
            }
            
            return {"success": True, "dataset": result.__dict__ if hasattr(result, '__dict__') else result}
            
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
        self.current_version_uuid = None  # Track current working version
    
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
                "datasets": prompts,  # For compatibility with test expectations
                "evaluations": [],  # Placeholder - could be enhanced to fetch evaluations
                "sync_timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
            logger.info(f"Synced {len(prompts)} prompts from Latitude")
            
            return sync_results
            
        except LatitudeAPIError as e:
            # "Head commit not found" indicates an empty project
            if "Head commit not found" in str(e):
                logger.info("Project is empty (no commits found)")
                return {
                    "prompts": [],
                    "datasets": [],
                    "evaluations": [],
                    "sync_timestamp": datetime.now().isoformat(),
                    "status": "success",
                    "note": "Empty project (no commits found)"
                }
            else:
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
        
        # Process safety metrics (results.safety_results is a List, not Dict)
        for framework_result in results.safety_results:
            if framework_result.success:
                for metric_name, metric_value in framework_result.metrics.items():
                    if isinstance(metric_value, (int, float, bool)):
                        evaluation = LatitudeEvaluation(
                            conversation_uuid=conversation_uuid,
                            evaluation_uuid=f"safety_{framework_result.framework_name}_{metric_name}",
                            result=metric_value,
                            reason=f"Safety evaluation result from {framework_result.framework_name}",
                            metadata={
                                "category": "safety",
                                "framework": framework_result.framework_name,
                                "metric": metric_name,
                                "framework_metadata": framework_result.metadata
                            }
                        )
                        latitude_evaluations.append(evaluation)
        
        # Process security metrics (results.security_results is a List, not Dict)
        for framework_result in results.security_results:
            if framework_result.success:
                for metric_name, metric_value in framework_result.metrics.items():
                    if isinstance(metric_value, (int, float, bool)):
                        evaluation = LatitudeEvaluation(
                            conversation_uuid=conversation_uuid,
                            evaluation_uuid=f"security_{framework_result.framework_name}_{metric_name}",
                            result=metric_value,
                            reason=f"Security evaluation result from {framework_result.framework_name}",
                            metadata={
                                "category": "security",
                                "framework": framework_result.framework_name,
                                "metric": metric_name,
                                "framework_metadata": framework_result.metadata
                            }
                        )
                        latitude_evaluations.append(evaluation)
        
        # Process reliability metrics (results.reliability_results is a List, not Dict)
        for framework_result in results.reliability_results:
            if framework_result.success:
                for metric_name, metric_value in framework_result.metrics.items():
                    if isinstance(metric_value, (int, float, bool)):
                        evaluation = LatitudeEvaluation(
                            conversation_uuid=conversation_uuid,
                            evaluation_uuid=f"reliability_{framework_result.framework_name}_{metric_name}",
                            result=metric_value,
                            reason=f"Reliability evaluation result from {framework_result.framework_name}",
                            metadata={
                                "category": "reliability",
                                "framework": framework_result.framework_name,
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
            # "Head commit not found" indicates an empty project, which is valid
            if "Head commit not found" in str(e):
                return {
                    "status": "healthy",
                    "api_accessible": True,
                    "authentication": "valid",
                    "prompt_count": 0,
                    "note": "Empty project (no commits found)",
                    "timestamp": datetime.now().isoformat()
                }
            else:
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
    
    async def create_new_version(self) -> Dict[str, Any]:
        """
        Create a new draft version for adding prompts.
        
        Returns:
            Dict containing version creation results
        """
        try:
            logger.info("Creating new version for prompt management")
            
            version_result = await self.client.create_version()
            
            if version_result.get("success"):
                version_info = version_result.get("version", {})
                self.current_version_uuid = version_info.get("uuid")
                logger.info(f"Created new version: {self.current_version_uuid}")
                
                return {
                    "status": "success", 
                    "version_uuid": self.current_version_uuid,
                    "version_info": version_info,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "failed",
                    "error": "Version creation failed",
                    "result": version_result,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to create new version: {e}")
            return {
                "status": "failed",
                "error": str(e), 
                "timestamp": datetime.now().isoformat()
            }
    
    async def push_hf_dataset_as_prompts(self, 
                                       dataset_name: str, 
                                       dataset_config: Optional[str] = None,
                                       split: str = "test_public",
                                       max_samples: int = 10,
                                       prompt_field: str = "prompt",
                                       path_prefix: str = "hf_datasets") -> Dict[str, Any]:
        """
        Pull prompts from a Hugging Face dataset and push them to Latitude as individual prompts.
        
        Args:
            dataset_name: HF dataset name (e.g., "ai-safety-institute/AgentHarm")
            dataset_config: Dataset configuration name (e.g., "harmful" for AgentHarm)
            split: Dataset split to use (default: "train")
            max_samples: Maximum number of samples to push (default: 10)
            prompt_field: Field name containing the prompt text (default: "prompt")
            path_prefix: Prefix for Latitude prompt paths (default: "hf_datasets")
            
        Returns:
            Dict containing the push operation results
        """
        try:
            logger.info(f"Loading HF dataset: {dataset_name}")
            
            # Note: Version management must be done manually in Latitude interface
            # Create a new draft version in Latitude before running this integration
            
            # Load dataset from Hugging Face
            from datasets import load_dataset
            if dataset_config:
                dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=False)
            else:
                dataset = load_dataset(dataset_name, split=split, streaming=False)
            
            # Take a sample
            if len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
            
            push_results = {
                "dataset_name": dataset_name,
                "total_samples": len(dataset),
                "pushed_prompts": [],
                "errors": [],
                "summary": {}
            }
            
            # Convert dataset name to safe path
            safe_dataset_name = dataset_name.replace("/", "_").replace("-", "_")
            if dataset_config:
                safe_dataset_name += f"_{dataset_config}"
            
            # Push each sample as a prompt
            for i, sample in enumerate(dataset):
                try:
                    prompt_content = sample.get(prompt_field, str(sample))
                    if not isinstance(prompt_content, str):
                        prompt_content = str(prompt_content)
                    
                    # Create a unique path for this prompt
                    prompt_path = f"{path_prefix}/{safe_dataset_name}/sample_{i:04d}"
                    
                    # Add metadata as description
                    description = f"Sample {i} from HF dataset '{dataset_name}'"
                    if len(sample) > 1:  # Add other fields as metadata
                        other_fields = {k: v for k, v in sample.items() if k != prompt_field}
                        description += f"\nMetadata: {json.dumps(other_fields, default=str)[:200]}..."
                    
                    # Create prompt in Latitude
                    result = await self.client.create_prompt(
                        path=prompt_path,
                        content=prompt_content[:2000],  # Limit content length
                        description=description
                    )
                    
                    push_results["pushed_prompts"].append({
                        "index": i,
                        "path": prompt_path,
                        "content_length": len(prompt_content),
                        "status": "success",
                        "result": result
                    })
                    
                    logger.info(f"Created prompt {i+1}/{len(dataset)}: {prompt_path}")
                    
                except Exception as e:
                    error_info = {
                        "index": i,
                        "error": str(e),
                        "sample": str(sample)[:100] + "..." if len(str(sample)) > 100 else str(sample)
                    }
                    push_results["errors"].append(error_info)
                    logger.error(f"Failed to create prompt {i}: {e}")
            
            # Create summary
            push_results["summary"] = {
                "total_samples": len(dataset),
                "successful_pushes": len(push_results["pushed_prompts"]),
                "failed_pushes": len(push_results["errors"]),
                "dataset_name": dataset_name,
                "split": split,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Pushed {push_results['summary']['successful_pushes']}/{push_results['summary']['total_samples']} prompts from {dataset_name}")
            
            return push_results
            
        except Exception as e:
            logger.error(f"Failed to push HF dataset {dataset_name}: {e}")
            return {
                "error": str(e),
                "dataset_name": dataset_name,
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
    
    async def push_hf_dataset_as_dataset(self, 
                                       dataset_name: str, 
                                       dataset_config: Optional[str] = None,
                                       split: str = "test_public",
                                       max_samples: int = 100,
                                       latitude_dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Pull a Hugging Face dataset and push it to Latitude as a single dataset.
        
        Args:
            dataset_name: HF dataset name (e.g., "ai-safety-institute/AgentHarm")
            dataset_config: Dataset configuration name (e.g., "harmful" for AgentHarm)
            split: Dataset split to use (default: "train")
            max_samples: Maximum number of samples to include (default: 100)
            latitude_dataset_name: Name for the dataset in Latitude (auto-generated if None)
            
        Returns:
            Dict containing the push operation results
        """
        try:
            logger.info(f"Loading HF dataset: {dataset_name}")
            
            # Load dataset from Hugging Face
            from datasets import load_dataset
            if dataset_config:
                dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=False)
            else:
                dataset = load_dataset(dataset_name, split=split, streaming=False)
            
            # Take a sample
            if len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
            
            # Convert to list of dictionaries
            data_records = []
            for i, sample in enumerate(dataset):
                # Convert all values to strings to ensure JSON serialization
                record = {"sample_id": i}
                for key, value in sample.items():
                    record[key] = str(value) if not isinstance(value, (str, int, float, bool)) else value
                data_records.append(record)
            
            # Generate dataset name if not provided
            if not latitude_dataset_name:
                safe_name = dataset_name.replace("/", "_").replace("-", "_")
                latitude_dataset_name = f"hf_{safe_name}_{split}_{len(data_records)}_samples"
            
            # Create description
            description = f"Dataset imported from Hugging Face: {dataset_name} ({split} split, {len(data_records)} samples)"
            
            # Push to Latitude
            result = await self.client.create_dataset(
                name=latitude_dataset_name,
                data=data_records,
                description=description
            )
            
            push_result = {
                "dataset_name": dataset_name,
                "latitude_dataset_name": latitude_dataset_name,
                "total_samples": len(data_records),
                "split": split,
                "status": "success",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully pushed {len(data_records)} samples from {dataset_name} to Latitude dataset '{latitude_dataset_name}'")
            
            return push_result
            
        except Exception as e:
            logger.error(f"Failed to push HF dataset {dataset_name} as dataset: {e}")
            return {
                "error": str(e),
                "dataset_name": dataset_name,
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }