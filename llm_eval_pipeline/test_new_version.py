"""
Test script for the new Latitude version with specific version UUID.

This script tests pushing HF dataset samples to your new Latitude version:
projects/20572/versions/3d300ce7-0f9f-49c5-b47a-c850e40a6b9b/
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Enable logging
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
)


async def test_new_version_integration():
    """Test integration with the new Latitude version."""
    print("🚀 Testing New Version Integration")
    print("-" * 50)
    print("Target Version: 3d300ce7-0f9f-49c5-b47a-c850e40a6b9b")
    print("-" * 50)
    
    # Check for Latitude configuration
    api_key = os.getenv('LATITUDE_API_KEY')
    project_id_str = os.getenv('LATITUDE_PROJECT_ID')
    
    if not api_key:
        print("⚠️ No LATITUDE_API_KEY found")
        return
    
    project_id = int(project_id_str) if project_id_str else None
    
    # Configure with your specific version UUID
    config = LatitudeConfig(
        api_key=api_key, 
        project_id=project_id,
        version_uuid="3d300ce7-0f9f-49c5-b47a-c850e40a6b9b",  # Your new version
        use_draft=True
    )
    integration = LatitudeIntegration(config)
    
    try:
        # Test 1: Basic connectivity
        print("1️⃣ Testing API connectivity...")
        health_result = await integration.health_check()
        print(f"   Status: {health_result['status']}")
        
        # Test 2: Push a single test prompt first
        print("\n2️⃣ Testing single prompt creation...")
        single_prompt_result = await integration.client.create_prompt(
            path="test/hf_integration_test",
            content="This is a test prompt to verify the new version integration works correctly."
        )
        
        if single_prompt_result["success"]:
            print("   ✅ Single prompt creation successful!")
            print(f"   Prompt created: test/hf_integration_test")
        else:
            print("   ❌ Single prompt creation failed")
            return
        
        # Test 3: Push HF dataset samples
        print("\n3️⃣ Testing HF dataset integration...")
        result = await integration.push_hf_dataset_as_prompts(
            dataset_name="ai-safety-institute/AgentHarm",
            dataset_config="harmful",
            max_samples=3,  # Start with 3 samples
            prompt_field="prompt",
            path_prefix="agentharm_v2"
        )
        
        print(f"📊 HF Integration Results:")
        print(f"   - Dataset: {result.get('dataset_name', 'Unknown')}")
        print(f"   - Total samples: {result.get('total_samples', 0)}")
        
        if 'summary' in result:
            print(f"   - Successful pushes: {result['summary']['successful_pushes']}")
            print(f"   - Failed pushes: {result['summary']['failed_pushes']}")
            
            if result['summary']['successful_pushes'] > 0:
                print("   ✅ SUCCESS: HF prompts pushed to new version!")
                
                # Show created prompts
                if result.get('pushed_prompts'):
                    print(f"   📝 Created prompts:")
                    for prompt in result['pushed_prompts']:
                        print(f"     - {prompt['path']}: {prompt['content_length']} chars")
            else:
                print("   ⚠️ No prompts were successfully pushed")
        
        if result.get('errors'):
            print(f"   ⚠️ Errors: {len(result['errors'])}")
            for error in result['errors'][:2]:  # Show first 2 errors
                print(f"     - Sample {error.get('index', '?')}: {error.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


async def test_connectivity_only():
    """Test just the basic connectivity with the new version."""
    print("🔧 Testing Basic Connectivity")
    print("-" * 30)
    
    api_key = os.getenv('LATITUDE_API_KEY')
    project_id = int(os.getenv('LATITUDE_PROJECT_ID')) if os.getenv('LATITUDE_PROJECT_ID') else None
    
    config = LatitudeConfig(
        api_key=api_key, 
        project_id=project_id,
        version_uuid="3d300ce7-0f9f-49c5-b47a-c850e40a6b9b"
    )
    integration = LatitudeIntegration(config)
    
    try:
        print("Testing health check...")
        health = await integration.health_check()
        print(f"Health Status: {health['status']}")
        
        print("Testing prompt fetch...")
        prompts = await integration.client.get_all_prompts()
        print(f"Found {len(prompts)} prompts in version")
        
        return {"status": "success", "prompts": len(prompts)}
        
    except Exception as e:
        print(f"Connectivity test failed: {e}")
        return {"status": "failed", "error": str(e)}


async def main():
    """Run integration tests with the new version."""
    print("🌟 Latitude New Version Integration Test")
    print("=" * 60)
    print("Version UUID: 3d300ce7-0f9f-49c5-b47a-c850e40a6b9b")
    print("Project ID: 20572")
    print("=" * 60)
    
    # Test basic connectivity first
    connectivity_result = await test_connectivity_only()
    
    if connectivity_result["status"] == "success":
        print(f"\n✅ Basic connectivity successful!")
        print(f"Proceeding with full integration test...\n")
        
        # Run full integration test
        integration_result = await test_new_version_integration()
        
        if integration_result and 'summary' in integration_result:
            success_count = integration_result['summary']['successful_pushes']
            total_count = integration_result['summary']['total_evaluations'] if 'total_evaluations' in integration_result['summary'] else integration_result.get('total_samples', 0)
            
            print(f"\n🎉 Integration Test Complete!")
            print(f"Successfully pushed {success_count} prompts to your new Latitude version.")
            print(f"\n💡 Next Steps:")
            print(f"1. Check your Latitude project to see the new prompts")
            print(f"2. Review the AgentHarm prompts in the draft version")
            print(f"3. Publish the version when ready")
            print(f"4. Run prompts to test model responses")
        
    else:
        print(f"\n❌ Basic connectivity failed: {connectivity_result.get('error')}")
        print(f"Please check your API key and version UUID")


if __name__ == "__main__":
    asyncio.run(main())