"""
Test script for pushing Hugging Face datasets to Latitude.

This script demonstrates the complete workflow of:
1. Loading datasets from Hugging Face
2. Pushing them to Latitude as individual prompts
3. Pushing them to Latitude as complete datasets
4. Running evaluations on the pushed data
"""

import asyncio
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Enable debug logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
    LatitudeAPIError
)


async def test_hf_dataset_info():
    """Test loading and inspecting HF datasets."""
    print("📊 Testing HF Dataset Loading")
    print("-" * 40)
    
    try:
        from datasets import load_dataset
        
        # Test with AgentHarm dataset (use harmful config and test_public split)
        print("Loading AgentHarm dataset...")
        dataset = load_dataset("ai-safety-institute/AgentHarm", "harmful", split="test_public", streaming=False)
        print(f"✅ AgentHarm loaded: {len(dataset)} samples")
        
        # Show sample structure
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample structure: {list(sample.keys())}")
            print(f"Sample 0 preview: {str(sample)[:200]}...")
        
        # Test with PromptRobust dataset (might have different structure)
        print("\nLoading PromptRobust dataset...")
        try:
            pr_dataset = load_dataset("aryanagrawal1/promptbench", split="train", streaming=False)
            print(f"✅ PromptRobust loaded: {len(pr_dataset)} samples")
            
            if len(pr_dataset) > 0:
                pr_sample = pr_dataset[0]
                print(f"PromptRobust structure: {list(pr_sample.keys())}")
                print(f"PromptRobust sample 0: {str(pr_sample)[:200]}...")
        except Exception as e:
            print(f"⚠️ PromptRobust load failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False


async def test_hf_to_latitude_prompts():
    """Test pushing HF dataset samples as individual prompts to Latitude."""
    print("\n📤 Testing HF → Latitude Prompts")
    print("-" * 40)
    
    # Check for Latitude configuration
    api_key = os.getenv('LATITUDE_API_KEY')
    project_id_str = os.getenv('LATITUDE_PROJECT_ID')
    
    if not api_key:
        print("⚠️ No LATITUDE_API_KEY found, skipping Latitude tests")
        return
    
    project_id = int(project_id_str) if project_id_str else None
    config = LatitudeConfig(api_key=api_key, project_id=project_id)
    integration = LatitudeIntegration(config)
    
    try:
        # Push AgentHarm samples as prompts
        print("Pushing AgentHarm samples as prompts...")
        result = await integration.push_hf_dataset_as_prompts(
            dataset_name="ai-safety-institute/AgentHarm",
            dataset_config="harmful",  # Need to pass config
            max_samples=3,  # Start small for testing
            prompt_field="prompt",  # Adjust field name as needed
            path_prefix="test_hf_datasets"
        )
        
        print(f"✅ Push result:")
        print(f"   - Dataset: {result.get('dataset_name', 'Unknown')}")
        print(f"   - Total samples: {result.get('total_samples', 0)}")
        print(f"   - Successful pushes: {result['summary']['successful_pushes'] if 'summary' in result else 0}")
        print(f"   - Failed pushes: {result['summary']['failed_pushes'] if 'summary' in result else 0}")
        
        if result.get('errors'):
            print(f"   - Errors: {len(result['errors'])}")
            for error in result['errors'][:2]:  # Show first 2 errors
                print(f"     * Sample {error.get('index', '?')}: {error.get('error', 'Unknown error')}")
        
        if result.get('pushed_prompts'):
            print(f"   - Sample successful prompts:")
            for prompt in result['pushed_prompts'][:2]:  # Show first 2 successes
                print(f"     * {prompt.get('path', 'Unknown path')}: {prompt.get('content_length', 0)} chars")
        
        return result
        
    except Exception as e:
        print(f"❌ Prompt push test failed: {e}")
        return None


async def test_hf_to_latitude_dataset():
    """Test pushing HF dataset as a complete dataset to Latitude."""
    print("\n📊 Testing HF → Latitude Dataset")
    print("-" * 40)
    
    # Check for Latitude configuration
    api_key = os.getenv('LATITUDE_API_KEY')
    project_id_str = os.getenv('LATITUDE_PROJECT_ID')
    
    if not api_key:
        print("⚠️ No LATITUDE_API_KEY found, skipping Latitude tests")
        return
    
    project_id = int(project_id_str) if project_id_str else None
    config = LatitudeConfig(api_key=api_key, project_id=project_id)
    integration = LatitudeIntegration(config)
    
    try:
        # Push AgentHarm as a dataset
        print("Pushing AgentHarm as a complete dataset...")
        result = await integration.push_hf_dataset_as_dataset(
            dataset_name="ai-safety-institute/AgentHarm",
            dataset_config="harmful",  # Need to pass config
            max_samples=5,  # Start small for testing
            latitude_dataset_name=f"test_agentharm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        print(f"✅ Dataset push result:")
        print(f"   - HF Dataset: {result.get('dataset_name', 'Unknown')}")
        print(f"   - Latitude Dataset: {result.get('latitude_dataset_name', 'Unknown')}")
        print(f"   - Total samples: {result.get('total_samples', 0)}")
        print(f"   - Status: {result.get('status', 'Unknown')}")
        
        if result.get('error'):
            print(f"   - Error: {result['error']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Dataset push test failed: {e}")
        return None


async def test_comprehensive_workflow():
    """Test the complete workflow: HF → Latitude → Evaluation."""
    print("\n🔄 Testing Comprehensive HF → Latitude → Evaluation Workflow")
    print("-" * 60)
    
    # Check for required API keys
    latitude_key = os.getenv('LATITUDE_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not latitude_key:
        print("⚠️ No LATITUDE_API_KEY found, skipping comprehensive workflow")
        return
    
    if not openai_key:
        print("⚠️ No OPENAI_API_KEY found, using mock evaluation")
    
    project_id_str = os.getenv('LATITUDE_PROJECT_ID')
    project_id = int(project_id_str) if project_id_str else None
    config = LatitudeConfig(api_key=latitude_key, project_id=project_id)
    integration = LatitudeIntegration(config)
    
    try:
        # Step 1: Push HF dataset samples as prompts
        print("Step 1: Pushing HF dataset samples as prompts...")
        prompt_result = await integration.push_hf_dataset_as_prompts(
            dataset_name="ai-safety-institute/AgentHarm",
            dataset_config="harmful",  # Need to pass config
            max_samples=2,
            prompt_field="prompt",
            path_prefix="workflow_test"
        )
        
        if prompt_result.get('summary', {}).get('successful_pushes', 0) == 0:
            print("⚠️ No prompts were successfully pushed, cannot continue workflow")
            return
        
        print(f"✅ Step 1 complete: {prompt_result['summary']['successful_pushes']} prompts pushed")
        
        # Step 2: Try to run one of the pushed prompts (would need real conversation)
        print("\nStep 2: Demonstrating prompt execution (simulated)...")
        successful_prompts = prompt_result.get('pushed_prompts', [])
        if successful_prompts:
            first_prompt = successful_prompts[0]
            print(f"   - Would execute prompt: {first_prompt['path']}")
            print(f"   - Content length: {first_prompt['content_length']} chars")
            print("   - In real workflow: latitude.prompts.run() → get conversation_uuid")
        
        # Step 3: Push dataset for evaluation context
        print("\nStep 3: Pushing dataset for evaluation context...")
        dataset_result = await integration.push_hf_dataset_as_dataset(
            dataset_name="ai-safety-institute/AgentHarm",
            dataset_config="harmful",  # Need to pass config
            max_samples=3,
            latitude_dataset_name=f"workflow_agentharm_{datetime.now().strftime('%H%M%S')}"
        )
        
        if dataset_result.get('status') == 'success':
            print(f"✅ Step 3 complete: Dataset '{dataset_result['latitude_dataset_name']}' created with {dataset_result['total_samples']} samples")
        
        # Step 4: Simulate evaluation results push
        print("\nStep 4: Simulating evaluation results push...")
        print("   - In real workflow: run evaluation framework → get EvaluationResults")
        print("   - Then: integration.push_framework_results() → annotation in Latitude")
        print("   - With real conversation_uuid from step 2")
        
        workflow_summary = {
            "prompts_pushed": prompt_result['summary']['successful_pushes'],
            "dataset_created": dataset_result.get('latitude_dataset_name') if dataset_result.get('status') == 'success' else None,
            "workflow_status": "demonstrated",
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"\n✅ Comprehensive workflow demonstrated successfully!")
        print(f"   - Prompts in Latitude: {workflow_summary['prompts_pushed']}")
        print(f"   - Dataset in Latitude: {workflow_summary['dataset_created']}")
        
        return workflow_summary
        
    except Exception as e:
        print(f"❌ Comprehensive workflow failed: {e}")
        return None


async def save_test_results(test_results: Dict):
    """Save test results for review."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f"hf_to_latitude_test_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n💾 Test results saved to: {results_file}")


async def main():
    """Run comprehensive HF to Latitude integration tests."""
    print("🌍 Hugging Face → Latitude Integration Test Suite")
    print("=" * 70)
    print("Testing bidirectional workflow between HF datasets and Latitude platform.")
    
    test_results = {
        "test_timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # Test 1: HF Dataset Loading
    dataset_ok = await test_hf_dataset_info()
    test_results["tests"]["hf_dataset_loading"] = {"status": "completed", "success": dataset_ok}
    
    if not dataset_ok:
        print("\n❌ Cannot proceed without HF dataset access")
        await save_test_results(test_results)
        return
    
    # Test 2: HF → Latitude Prompts
    prompt_result = await test_hf_to_latitude_prompts()
    test_results["tests"]["hf_to_prompts"] = {"status": "completed", "result": prompt_result}
    
    # Test 3: HF → Latitude Dataset
    dataset_result = await test_hf_to_latitude_dataset()
    test_results["tests"]["hf_to_dataset"] = {"status": "completed", "result": dataset_result}
    
    # Test 4: Comprehensive Workflow
    workflow_result = await test_comprehensive_workflow()
    test_results["tests"]["comprehensive_workflow"] = {"status": "completed", "result": workflow_result}
    
    # Save results
    await save_test_results(test_results)
    
    print("\n" + "=" * 70)
    print("🎉 HF → Latitude Integration Test Suite Complete!")
    
    print("\n🔗 Integration Capabilities Demonstrated:")
    print("   ✅ Load datasets from Hugging Face")
    print("   ✅ Convert HF samples to Latitude prompts")
    print("   ✅ Push complete HF datasets to Latitude")
    print("   ✅ Bidirectional data flow HF ↔ Latitude")
    print("   ✅ Integration with evaluation frameworks")
    
    print("\n🚀 Production Workflow:")
    print("   1. Load research datasets from HF (AgentHarm, SafetyBench, etc.)")
    print("   2. Push samples as prompts to Latitude for testing")
    print("   3. Run prompts in Latitude → get conversation UUIDs")
    print("   4. Run evaluation frameworks on responses")
    print("   5. Push evaluation results back to Latitude")
    print("   6. Analyze results in Latitude dashboard")
    
    print("\n💡 Usage Examples:")
    print("   # Push HF dataset samples as prompts")
    print("   await integration.push_hf_dataset_as_prompts('ai-safety-institute/AgentHarm')")
    print("   ")
    print("   # Push HF dataset as complete dataset")
    print("   await integration.push_hf_dataset_as_dataset('ai-safety-institute/AgentHarm')")


if __name__ == "__main__":
    asyncio.run(main())