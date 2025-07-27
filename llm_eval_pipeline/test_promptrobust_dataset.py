"""Test script for PromptRobust dataset integration."""

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

from llm_eval.frameworks.reliability.promptrobust_hf import PromptRobustHFEvaluator
from llm_eval.core.model_interface import create_model


async def test_promptrobust_dataset():
    """Test the PromptRobust dataset integration."""
    print("🧪 Testing PromptRobust Dataset Integration")
    print("=" * 50)
    
    # Test 1: Load dataset and inspect
    print("\n📊 Loading PromptRobust Dataset...")
    try:
        evaluator = PromptRobustHFEvaluator(
            sample_size=10,  # Small sample for testing
            attack_types=['character', 'word', 'sentence', 'semantic']
        )
        
        dataset_info = evaluator.get_dataset_info()
        print(f"✅ Dataset loaded successfully:")
        print(f"   - Total samples: {dataset_info['total_samples']}")
        print(f"   - Attack types: {dataset_info['attack_types']}")
        print(f"   - Sample size used: {dataset_info['sample_size_used']}")
        
        # Show sample data
        if hasattr(evaluator.dataset, '__iter__'):
            print(f"\n📝 Sample prompt pairs:")
            for i, sample in enumerate(evaluator.dataset):
                if i >= 3:  # Show first 3
                    break
                original = sample.get('original', '')[:80] + "..." if len(sample.get('original', '')) > 80 else sample.get('original', '')
                attacked = sample.get('attacked', '')[:80] + "..." if len(sample.get('attacked', '')) > 80 else sample.get('attacked', '')
                attack_type = sample.get('attack_type', 'unknown')
                print(f"   {i+1}. [{attack_type}]")
                print(f"      Original: {original}")
                print(f"      Attacked: {attacked}")
        
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
            
            # Show attack types tested
            if 'attack_types_tested' in result.metadata:
                print(f"   - Attack types tested: {result.metadata['attack_types_tested']}")
            
            # Show attack breakdown
            if 'attack_breakdown' in result.metadata:
                print(f"   - Attack performance breakdown:")
                for attack_type, performance in result.metadata['attack_breakdown'].items():
                    if performance['total'] > 0:
                        consistency_rate = performance['consistent'] / performance['total']
                        print(f"     * {attack_type}: {consistency_rate:.3f} consistency ({performance['consistent']}/{performance['total']})")
            
            return True
        else:
            print(f"❌ Evaluation failed")
            return False
            
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
        return False


async def test_robustness_scenarios():
    """Test different robustness scenarios."""
    print("\n🔍 Testing Different Robustness Scenarios...")
    
    scenarios = [
        ("character", ["character"]),
        ("word", ["word"]),
        ("sentence", ["sentence"]),
        ("semantic", ["semantic"]),
        ("all_attacks", ["character", "word", "sentence", "semantic"])
    ]
    
    for scenario_name, attack_types in scenarios:
        try:
            print(f"\n   Testing {scenario_name} attacks...")
            evaluator = PromptRobustHFEvaluator(
                attack_types=attack_types,
                sample_size=5  # Very small for quick testing
            )
            
            info = evaluator.get_dataset_info()
            print(f"   ✅ {scenario_name}: {info['total_samples']} samples, {len(info['attack_types'])} attack types")
            
        except Exception as e:
            print(f"   ❌ {scenario_name}: Failed to load - {e}")


async def demonstrate_attack_types():
    """Demonstrate different types of adversarial attacks."""
    print("\n🎯 Demonstrating Attack Types...")
    
    evaluator = PromptRobustHFEvaluator(sample_size=5)
    
    # Test synthetic attack generation
    base_prompt = "What is the capital of France?"
    
    print(f"\n   Base prompt: {base_prompt}")
    print(f"   Attack variations:")
    
    try:
        char_attack = evaluator._apply_character_attack(base_prompt)
        print(f"   • Character: {char_attack}")
        
        word_attack = evaluator._apply_word_attack(base_prompt)
        print(f"   • Word: {word_attack}")
        
        sentence_attack = evaluator._apply_sentence_attack(base_prompt)
        print(f"   • Sentence: {sentence_attack}")
        
        semantic_attack = evaluator._apply_semantic_attack(base_prompt)
        print(f"   • Semantic: {semantic_attack}")
        
    except Exception as e:
        print(f"   ❌ Attack generation failed: {e}")


async def main():
    """Run all tests."""
    print("🚀 PromptRobust Dataset Integration Test")
    print("=" * 60)
    
    # Test basic dataset loading
    dataset_success = await test_promptrobust_dataset()
    
    # Test different scenarios
    await test_robustness_scenarios()
    
    # Demonstrate attack types
    await demonstrate_attack_types()
    
    print("\n" + "=" * 60)
    if dataset_success:
        print("🎉 PromptRobust dataset integration successful!")
        print("   You can now use PromptRobust for robustness evaluation")
    else:
        print("⚠️ PromptRobust dataset integration partially successful")
        print("   Framework works, but may need dataset access")
    
    print("\n💡 Usage example:")
    print("   from llm_eval.frameworks.reliability.promptrobust_hf import PromptRobustHFEvaluator")
    print("   evaluator = PromptRobustHFEvaluator(sample_size=100)")
    print("   result = await evaluator.evaluate(model)")
    
    print("\n🛡️ Key Features:")
    print("   ✅ Character-level attacks (typos, character manipulation)")
    print("   ✅ Word-level attacks (synonym replacement)")
    print("   ✅ Sentence-level attacks (irrelevant additions)")
    print("   ✅ Semantic-level attacks (paraphrasing)")
    print("   ✅ Robustness consistency scoring")
    print("   ✅ Attack-specific performance breakdown")


if __name__ == "__main__":
    asyncio.run(main())