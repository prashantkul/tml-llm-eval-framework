"""
Test script for creating new versions and pushing prompts to Latitude.

This script demonstrates how to:
1. Create a new version/draft in Latitude
2. Push HF dataset samples as prompts to the new version
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


async def test_version_creation():
    """Test creating a new version in Latitude."""
    print("🔄 Testing Version Creation")
    print("-" * 40)
    
    # Check for Latitude configuration
    api_key = os.getenv('LATITUDE_API_KEY')
    project_id_str = os.getenv('LATITUDE_PROJECT_ID')
    
    if not api_key:
        print("⚠️ No LATITUDE_API_KEY found")
        return
    
    project_id = int(project_id_str) if project_id_str else None
    config = LatitudeConfig(
        api_key=api_key, 
        project_id=project_id,
        use_draft=True  # Enable draft mode
    )
    integration = LatitudeIntegration(config)
    
    try:
        # Test version creation
        print("Creating new version...")
        version_result = await integration.create_new_version()
        
        print(f"✅ Version creation result:")
        print(f"   - Status: {version_result.get('status')}")
        print(f"   - Version UUID: {version_result.get('version_uuid')}")
        
        if version_result.get('error'):
            print(f"   - Error: {version_result['error']}")
            return version_result
        
        return version_result
        
    except Exception as e:
        print(f"❌ Version creation failed: {e}")
        return {"status": "failed", "error": str(e)}


async def test_hf_prompts_with_version():
    """Test pushing HF dataset samples to new version."""
    print("\n📤 Testing HF → Latitude Prompts (New Version)")
    print("-" * 50)
    
    # Check for Latitude configuration
    api_key = os.getenv('LATITUDE_API_KEY')
    project_id_str = os.getenv('LATITUDE_PROJECT_ID')
    
    if not api_key:
        print("⚠️ No LATITUDE_API_KEY found")
        return
    
    project_id = int(project_id_str) if project_id_str else None
    config = LatitudeConfig(
        api_key=api_key, 
        project_id=project_id,
        use_draft=True  # This will auto-create a new version
    )
    integration = LatitudeIntegration(config)
    
    try:
        # Push AgentHarm samples to new version
        print("Pushing AgentHarm samples to new version...")
        result = await integration.push_hf_dataset_as_prompts(
            dataset_name="ai-safety-institute/AgentHarm",
            dataset_config="harmful",
            max_samples=2,  # Start with just 2 samples
            prompt_field="prompt",
            path_prefix="hf_agentharm_v2"
        )
        
        print(f"✅ Push result:")
        print(f"   - Dataset: {result.get('dataset_name', 'Unknown')}")
        print(f"   - Total samples: {result.get('total_samples', 0)}")
        
        if 'summary' in result:
            print(f"   - Successful pushes: {result['summary']['successful_pushes']}")
            print(f"   - Failed pushes: {result['summary']['failed_pushes']}")
        
        if result.get('errors'):
            print(f"   - Errors: {len(result['errors'])}")
            for error in result['errors'][:2]:  # Show first 2 errors
                print(f"     * Sample {error.get('index', '?')}: {error.get('error', 'Unknown error')}")
        
        if result.get('pushed_prompts'):
            print(f"   - Successfully created prompts:")
            for prompt in result['pushed_prompts'][:2]:  # Show first 2 successes
                print(f"     * {prompt.get('path', 'Unknown path')}: {prompt.get('content_length', 0)} chars")
        
        # Show version info
        if integration.current_version_uuid:
            print(f"   - Version UUID: {integration.current_version_uuid}")
        
        return result
        
    except Exception as e:
        print(f"❌ HF push test failed: {e}")
        return {"status": "failed", "error": str(e)}


async def main():
    """Run version creation and HF integration tests."""
    print("🚀 Latitude Version Management & HF Integration Test")
    print("=" * 60)
    
    # Test 1: Version Creation
    version_result = await test_version_creation()
    
    # Test 2: HF Prompts with Version
    if version_result.get("status") == "success":
        hf_result = await test_hf_prompts_with_version()
        
        print(f"\n🎉 Integration Test Complete!")
        print(f"   - Version created: {version_result.get('version_uuid', 'None')}")
        if hf_result and 'summary' in hf_result:
            print(f"   - Prompts pushed: {hf_result['summary']['successful_pushes']}")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Check your Latitude project for the new draft version")
        print(f"   2. Review the pushed prompts in the draft")
        print(f"   3. Publish the version when ready")
        print(f"   4. Test running prompts to get conversation UUIDs")
        
    else:
        print(f"\n⚠️ Version creation failed, cannot test HF integration")


if __name__ == "__main__":
    asyncio.run(main())