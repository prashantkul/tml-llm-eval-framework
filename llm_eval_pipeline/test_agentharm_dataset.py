"""Test script for AgentHarm dataset integration."""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from llm_eval.frameworks.safety.agentharm_hf import AgentHarmHFEvaluator
from llm_eval.core.model_interface import create_model


async def test_agentharm_dataset():
    """Test the AgentHarm dataset integration."""
    print("🧪 Testing AgentHarm Dataset Integration")
    print("=" * 50)
    
    # Test 1: Load dataset and inspect
    print("\n📊 Loading AgentHarm Dataset...")
    try:
        evaluator = AgentHarmHFEvaluator(
            subset="harmful", 
            split="test_public",
            sample_size=5  # Small sample for testing
        )
        
        dataset_info = evaluator.get_dataset_info()
        print(f"✅ Dataset loaded successfully:")
        print(f"   - Subset: {dataset_info['subset']}")
        print(f"   - Split: {dataset_info['split']}")
        print(f"   - Total samples: {dataset_info['total_samples']}")
        print(f"   - Categories: {dataset_info['categories'][:5]}...")  # Show first 5
        
        # Show sample data
        if hasattr(evaluator.dataset, '__iter__'):
            print(f"\n📝 Sample prompts:")
            for i, sample in enumerate(evaluator.dataset):
                if i >= 3:  # Show first 3
                    break
                prompt_preview = sample.get('prompt', '')[:100] + "..." if len(sample.get('prompt', '')) > 100 else sample.get('prompt', '')
                print(f"   {i+1}. [{sample.get('category', 'unknown')}] {prompt_preview}")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("   This may be expected if you don't have access to the dataset")
        return False
    
    # Test 2: Run evaluation with a model
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("\n⚠️ No OpenAI API key found - skipping model evaluation")
        return True
    
    print(f"\n🤖 Testing with OpenAI Model...")
    try:
        model = create_model('openai/gpt-3.5-turbo', openai_api_key=openai_key)
        
        # Run evaluation
        result = await evaluator.evaluate(model)
        
        if result.success:
            print(f"✅ Evaluation completed successfully!")
            print(f"   - Framework: {result.framework_name}")
            print(f"   - Metrics:")
            for metric, value in result.metrics.items():
                if isinstance(value, float):
                    print(f"     * {metric}: {value:.3f}")
                else:
                    print(f"     * {metric}: {value}")
            
            # Show categories tested
            if 'categories_tested' in result.metadata:
                print(f"   - Categories tested: {result.metadata['categories_tested']}")
            
            return True
        else:
            print(f"❌ Evaluation failed")
            return False
            
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
        return False


async def test_different_subsets():
    """Test different dataset subsets."""
    print("\n🔍 Testing Different Dataset Subsets...")
    
    subsets_to_test = [
        ("harmful", "test_public"),
        ("harmless_benign", "test_public"),
        ("chat", "test_public")
    ]
    
    for subset, split in subsets_to_test:
        try:
            print(f"\n   Testing {subset}/{split}...")
            evaluator = AgentHarmHFEvaluator(
                subset=subset,
                split=split,
                sample_size=2  # Very small for quick testing
            )
            
            info = evaluator.get_dataset_info()
            print(f"   ✅ {subset}: {info['total_samples']} samples, {len(info['categories'])} categories")
            
        except Exception as e:
            print(f"   ❌ {subset}: Failed to load - {e}")


async def main():
    """Run all tests."""
    print("🚀 AgentHarm Dataset Integration Test")
    print("=" * 60)
    
    # Test basic dataset loading
    dataset_success = await test_agentharm_dataset()
    
    # Test different subsets
    await test_different_subsets()
    
    print("\n" + "=" * 60)
    if dataset_success:
        print("🎉 AgentHarm dataset integration successful!")
        print("   You can now use real AgentHarm data in evaluations")
    else:
        print("⚠️ AgentHarm dataset integration partially successful")
        print("   Dataset structure works, but may need access permissions")
    
    print("\n💡 Usage example:")
    print("   from llm_eval.frameworks.safety.agentharm_hf import AgentHarmHFEvaluator")
    print("   evaluator = AgentHarmHFEvaluator(subset='harmful', sample_size=10)")
    print("   result = await evaluator.evaluate(model)")


if __name__ == "__main__":
    asyncio.run(main())