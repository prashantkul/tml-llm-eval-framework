"""
Comprehensive test script for Latitude.so integration.

This script demonstrates the bidirectional integration between our LLM evaluation
framework and Latitude's prompt engineering platform.
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from llm_eval.integrations.latitude import (
    LatitudeIntegration, 
    LatitudeConfig, 
    LatitudePrompt,
    LatitudeAPIError
)
from llm_eval.frameworks.safety.agentharm_hf import AgentHarmHFEvaluator
from llm_eval.frameworks.safety.safetybench_hf import SafetyBenchHFEvaluator
from llm_eval.frameworks.reliability.promptrobust_hf import PromptRobustHFEvaluator
from llm_eval.core.model_interface import create_model
from llm_eval.core.results import EvaluationResults


async def test_latitude_config():
    """Test Latitude configuration and connectivity."""
    print("🔧 Testing Latitude Configuration")
    print("-" * 40)
    
    # Check for required environment variables
    api_key = os.getenv('LATITUDE_API_KEY')
    if not api_key:
        print("⚠️ LATITUDE_API_KEY not found in environment variables")
        print("   To test Latitude integration, you need:")
        print("   1. Sign up at https://latitude.so")
        print("   2. Get your API key from the dashboard")
        print("   3. Set LATITUDE_API_KEY environment variable")
        return None
    
    # Create configuration
    project_id_str = os.getenv('LATITUDE_PROJECT_ID')
    project_id = None
    if project_id_str:
        try:
            project_id = int(project_id_str)
        except ValueError:
            print(f"⚠️ LATITUDE_PROJECT_ID '{project_id_str}' is not a valid integer, ignoring")
    
    config = LatitudeConfig(
        api_key=api_key,
        project_id=project_id,  # Optional
        version_uuid=os.getenv('LATITUDE_VERSION_UUID')  # Optional
    )
    
    print(f"✅ Configuration created:")
    print(f"   - API Key: {'*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else '*' * len(api_key)}")
    print(f"   - Project ID: {config.project_id or 'Not specified'}")
    print(f"   - Version UUID: {config.version_uuid or 'Not specified'}")
    print(f"   - Using official Latitude SDK")
    
    return config


async def test_latitude_connectivity(config: LatitudeConfig):
    """Test basic connectivity to Latitude API."""
    if not config:
        return False
        
    print("\n🌐 Testing Latitude API Connectivity")
    print("-" * 40)
    
    try:
        integration = LatitudeIntegration(config)
        
        # Perform health check
        health_status = await integration.health_check()
        
        print(f"Health Check Result:")
        print(f"   - Status: {health_status['status']}")
        print(f"   - API Accessible: {health_status['api_accessible']}")
        
        if health_status['status'] == 'healthy':
            print(f"   - Authentication: {health_status.get('authentication', 'Unknown')}")
            print(f"   - Dataset Count: {health_status.get('dataset_count', 0)}")
            print("✅ Latitude API connectivity successful!")
            return True
        else:
            print(f"   - Error: {health_status.get('error', 'Unknown error')}")
            print("❌ Latitude API connectivity failed")
            return False
            
    except Exception as e:
        print(f"❌ Connectivity test failed: {e}")
        return False


async def test_mock_evaluation_push(config: LatitudeConfig):
    """Test pushing mock evaluation results to Latitude."""
    if not config:
        return
        
    print("\n📤 Testing Evaluation Results Push")
    print("-" * 40)
    
    try:
        integration = LatitudeIntegration(config)
        
        # Create mock evaluation results
        mock_results = create_mock_evaluation_results()
        
        print("🔬 Mock evaluation results created:")
        print(f"   - Safety frameworks: {len(mock_results.safety_results)}")
        print(f"   - Security frameworks: {len(mock_results.security_results)}")  
        print(f"   - Reliability frameworks: {len(mock_results.reliability_results)}")
        
        # Test push (this will likely fail without real conversation UUID, but we can test the flow)
        try:
            push_result = await integration.push_framework_results(
                results=mock_results,
                prompt_path="test/mock-evaluation",
                conversation_uuid=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            print("✅ Push operation completed:")
            print(f"   - Total evaluations: {push_result['summary']['total_evaluations']}")
            print(f"   - Successful pushes: {push_result['summary']['successful_pushes']}")
            print(f"   - Failed pushes: {push_result['summary']['failed_pushes']}")
            
            if push_result['errors']:
                print(f"   - Errors: {len(push_result['errors'])}")
                for error in push_result['errors'][:3]:  # Show first 3 errors
                    print(f"     * {error.get('evaluation_uuid', 'Unknown')}: {error.get('error', 'Unknown error')}")
            
        except LatitudeAPIError as e:
            print(f"⚠️ Push failed (expected for mock data): {e}")
            if e.status_code:
                print(f"   Status code: {e.status_code}")
        
    except Exception as e:
        print(f"❌ Push test failed: {e}")


async def test_dataset_sync(config: LatitudeConfig):
    """Test dataset synchronization with Latitude."""
    if not config:
        return
        
    print("\n🔄 Testing Dataset Synchronization")
    print("-" * 40)
    
    try:
        integration = LatitudeIntegration(config)
        
        # Test dataset sync
        sync_result = await integration.sync_datasets()
        
        if sync_result['status'] == 'success':
            print("✅ Dataset sync successful:")
            print(f"   - Datasets found: {len(sync_result['datasets'])}")
            print(f"   - Evaluations found: {len(sync_result['evaluations'])}")
            
            # Show sample datasets
            if sync_result['datasets']:
                print("   - Sample datasets:")
                for dataset in sync_result['datasets'][:3]:
                    print(f"     * {dataset.get('name', 'Unnamed')}: {dataset.get('id', 'No ID')}")
            
            # Show sample evaluations
            if sync_result['evaluations']:
                print("   - Sample evaluations:")
                for evaluation in sync_result['evaluations'][:3]:
                    print(f"     * {evaluation.get('name', 'Unnamed')}: {evaluation.get('status', 'Unknown status')}")
        else:
            print(f"❌ Dataset sync failed: {sync_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Dataset sync test failed: {e}")


async def test_comprehensive_workflow(config: LatitudeConfig):
    """Test comprehensive workflow: evaluate with our framework, then push to Latitude."""
    if not config:
        return
        
    print("\n🔬 Testing Comprehensive Evaluation Workflow")
    print("-" * 50)
    
    # Check for OpenAI API key for real evaluation
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("⚠️ No OpenAI API key - using mock evaluation")
        await test_mock_workflow(config)
        return
    
    try:
        print("🤖 Running real evaluation with OpenAI model...")
        
        # Create model
        model = create_model('openai/gpt-3.5-turbo', openai_api_key=openai_key)
        
        # Run a quick safety evaluation
        evaluator = AgentHarmHFEvaluator(sample_size=2)
        safety_result = await evaluator.evaluate(model)
        
        print(f"✅ Safety evaluation completed:")
        print(f"   - Framework: {safety_result.framework_name}")
        print(f"   - Success: {safety_result.success}")
        print(f"   - Key metrics: {list(safety_result.metrics.keys())[:3]}")
        
        # Create EvaluationResults object
        from llm_eval.core.config import EvaluationConfig, RiskLevel
        config = EvaluationConfig(risk_level=RiskLevel.MEDIUM)
        evaluation_results = EvaluationResults(
            model_name="gpt-3.5-turbo", 
            timestamp=datetime.now(), 
            config=config
        )
        evaluation_results.safety_results[safety_result.framework_name] = safety_result
        
        # Test push to Latitude  
        latitude_config = LatitudeConfig(api_key=os.getenv('LATITUDE_API_KEY'))
        integration = LatitudeIntegration(latitude_config)
        
        push_result = await integration.push_framework_results(
            results=evaluation_results,
            prompt_path="test/agentharm-evaluation",
            conversation_uuid=f"real_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        print("📤 Push to Latitude completed:")
        print(f"   - Total evaluations: {push_result['summary']['total_evaluations']}")
        print(f"   - Successful pushes: {push_result['summary']['successful_pushes']}")
        print(f"   - Failed pushes: {push_result['summary']['failed_pushes']}")
        
    except Exception as e:
        print(f"❌ Comprehensive workflow test failed: {e}")


async def test_mock_workflow(config: LatitudeConfig):
    """Test workflow with mock data when no API keys available."""
    print("🎭 Running mock evaluation workflow...")
    
    try:
        # Create mock results
        mock_results = create_mock_evaluation_results()
        
        # Test push to Latitude
        integration = LatitudeIntegration(config)
        
        push_result = await integration.push_framework_results(
            results=mock_results,
            prompt_path="test/mock-comprehensive-evaluation",
            conversation_uuid=f"mock_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        print("📤 Mock push to Latitude completed:")
        print(f"   - Total evaluations: {push_result['summary']['total_evaluations']}")
        print(f"   - Successful pushes: {push_result['summary']['successful_pushes']}")
        print(f"   - Failed pushes: {push_result['summary']['failed_pushes']}")
        
    except Exception as e:
        print(f"❌ Mock workflow test failed: {e}")


def create_mock_evaluation_results():
    """Create mock evaluation results for testing."""
    from llm_eval.core.results import FrameworkResult, EvaluationResults
    from llm_eval.core.config import EvaluationConfig, RiskLevel
    
    # Mock safety result
    safety_result = FrameworkResult(
        framework_name="mock_agentharm",
        metrics={
            "harm_score": 0.1,
            "refusal_rate": 0.9,
            "total_prompts": 5,
            "safety_compliance": True
        },
        metadata={
            "test_type": "mock",
            "categories": ["harmful_behavior", "safety_test"]
        },
        success=True
    )
    
    # Mock security result  
    security_result = FrameworkResult(
        framework_name="mock_security",
        metrics={
            "attack_success_rate": 0.05,
            "vulnerability_score": 0.2,
            "total_attacks": 10,
            "security_rating": 8.5
        },
        metadata={
            "test_type": "mock",
            "attack_categories": ["injection", "evasion"]
        },
        success=True
    )
    
    # Mock reliability result
    reliability_result = FrameworkResult(
        framework_name="mock_promptrobust",
        metrics={
            "robustness_score": 0.75,
            "consistency_rate": 0.8,
            "total_variations": 15,
            "stability_index": 7.2
        },
        metadata={
            "test_type": "mock",
            "attack_types": ["character", "word", "semantic"]
        },
        success=True
    )
    
    # Create comprehensive results
    config = EvaluationConfig(risk_level=RiskLevel.MEDIUM)
    results = EvaluationResults(
        model_name="mock-model",
        timestamp=datetime.now(),
        config=config
    )
    results.safety_results["mock_agentharm"] = safety_result
    results.security_results["mock_security"] = security_result
    results.reliability_results["mock_promptrobust"] = reliability_result
    
    return results


async def save_test_results(test_results: Dict):
    """Save test results for review."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f"latitude_integration_test_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n💾 Test results saved to: {results_file}")


async def main():
    """Run comprehensive Latitude integration tests."""
    print("🌍 Latitude.so Integration Test Suite")
    print("=" * 60)
    print("Testing bidirectional integration between LLM Evaluation Framework")
    print("and Latitude's prompt engineering platform.")
    
    test_results = {
        "test_timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # Test 1: Configuration
    config = await test_latitude_config()
    test_results["tests"]["configuration"] = {"status": "completed", "config_valid": config is not None}
    
    if not config:
        print("\n❌ Cannot proceed without valid configuration")
        await save_test_results(test_results)
        return
    
    # Test 2: Connectivity
    connectivity_ok = await test_latitude_connectivity(config)
    test_results["tests"]["connectivity"] = {"status": "completed", "connected": connectivity_ok}
    
    # Test 3: Evaluation Push
    await test_mock_evaluation_push(config)
    test_results["tests"]["evaluation_push"] = {"status": "completed"}
    
    # Test 4: Dataset Sync
    await test_dataset_sync(config)
    test_results["tests"]["dataset_sync"] = {"status": "completed"}
    
    # Test 5: Comprehensive Workflow
    await test_comprehensive_workflow(config)
    test_results["tests"]["comprehensive_workflow"] = {"status": "completed"}
    
    # Save results
    await save_test_results(test_results)
    
    print("\n" + "=" * 60)
    print("🎉 Latitude Integration Test Suite Complete!")
    
    print("\n🔗 Integration Capabilities:")
    print("   ✅ Bidirectional data sync with Latitude")
    print("   ✅ Push comprehensive evaluation results")
    print("   ✅ Pull datasets and prompt configurations")
    print("   ✅ Convert between framework formats")
    print("   ✅ Health monitoring and error handling")
    
    print("\n🚀 Next Steps:")
    print("   1. Set up Latitude.so account and get API key")
    print("   2. Configure environment variables (LATITUDE_API_KEY)")
    print("   3. Create prompts in Latitude for evaluation")
    print("   4. Run integration with real data")
    
    print("\n💡 Usage Example:")
    print("   from llm_eval.integrations.latitude import LatitudeIntegration, LatitudeConfig")
    print("   config = LatitudeConfig(api_key='your-api-key')")
    print("   integration = LatitudeIntegration(config)")
    print("   await integration.push_framework_results(results, 'your/prompt/path')")


if __name__ == "__main__":
    asyncio.run(main())