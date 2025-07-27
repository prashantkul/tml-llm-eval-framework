"""Test script demonstrating real research dataset integration."""

import asyncio
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from llm_eval.frameworks.safety.agentharm_hf import AgentHarmHFEvaluator
from llm_eval.frameworks.safety.safetybench_hf import SafetyBenchHFEvaluator
from llm_eval.core.model_interface import create_model


async def test_agentharm_evaluation():
    """Test evaluation using AgentHarm dataset."""
    print("🔬 AgentHarm Dataset Evaluation")
    print("-" * 40)
    
    # Create evaluator with research dataset
    evaluator = AgentHarmHFEvaluator(
        subset="harmful",
        split="test_public", 
        sample_size=5
    )
    
    # Show dataset info
    info = evaluator.get_dataset_info()
    print(f"📊 Dataset: {info['subset']}/{info['split']}")
    print(f"📝 Samples: {info['total_samples']}")
    print(f"🏷️ Categories: {info['categories'][:3]}...")
    
    # Get API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("⚠️ No OpenAI API key - using mock evaluation")
        return {"framework": "agentharm_hf", "status": "skipped"}
    
    # Create model and evaluate
    model = create_model('openai/gpt-3.5-turbo', openai_api_key=openai_key)
    
    print("⚡ Running evaluation...")
    result = await evaluator.evaluate(model)
    
    if result.success:
        print(f"✅ Evaluation completed!")
        key_metrics = {
            "harm_score": result.metrics.get("harm_score", 0),
            "refusal_rate": result.metrics.get("refusal_rate", 0),
            "total_prompts": result.metrics.get("total_prompts", 0)
        }
        print(f"📈 Key metrics: {key_metrics}")
        return {"framework": "agentharm_hf", "status": "success", "metrics": key_metrics}
    else:
        print(f"❌ Evaluation failed")
        return {"framework": "agentharm_hf", "status": "failed"}


async def test_safetybench_evaluation():
    """Test evaluation using SafetyBench dataset."""
    print("\\n📚 SafetyBench Dataset Evaluation")
    print("-" * 40)
    
    # Create evaluator with research dataset
    evaluator = SafetyBenchHFEvaluator(
        language="en",
        subset="test",
        sample_size=5
    )
    
    # Show dataset info
    info = evaluator.get_dataset_info()
    print(f"📊 Dataset: {info['language']}_{info['subset']}")
    print(f"📝 Samples: {info['total_samples']}")
    print(f"🏷️ Categories: {info['categories'][:3]}...")
    
    # Get API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("⚠️ No OpenAI API key - using mock evaluation")
        return {"framework": "safetybench_hf", "status": "skipped"}
    
    # Create model and evaluate
    model = create_model('openai/gpt-3.5-turbo', openai_api_key=openai_key)
    
    print("⚡ Running evaluation...")
    result = await evaluator.evaluate(model)
    
    if result.success:
        print(f"✅ Evaluation completed!")
        key_metrics = {
            "overall_accuracy": result.metrics.get("overall_accuracy", 0),
            "safety_compliance_rate": result.metrics.get("safety_compliance_rate", 0),
            "total_questions": result.metrics.get("total_questions", 0)
        }
        print(f"📈 Key metrics: {key_metrics}")
        return {"framework": "safetybench_hf", "status": "success", "metrics": key_metrics}
    else:
        print(f"❌ Evaluation failed")
        return {"framework": "safetybench_hf", "status": "failed"}


async def compare_datasets():
    """Compare results from different research datasets."""
    print("\\n⚖️ Research Dataset Comparison")
    print("=" * 50)
    
    # Run both evaluations
    agentharm_result = await test_agentharm_evaluation()
    safetybench_result = await test_safetybench_evaluation()
    
    # Summary comparison
    print("\\n📊 Dataset Comparison Summary:")
    print("-" * 30)
    
    datasets_info = [
        {
            "name": "AgentHarm",
            "purpose": "Agent harmfulness benchmark",
            "type": "Open-ended prompts",
            "paper": "arXiv:2410.09024",
            "result": agentharm_result
        },
        {
            "name": "SafetyBench", 
            "purpose": "LLM safety evaluation",
            "type": "Multiple-choice questions",
            "paper": "SafetyBench research",
            "result": safetybench_result
        }
    ]
    
    for dataset in datasets_info:
        print(f"\\n🔬 {dataset['name']}:")
        print(f"   Purpose: {dataset['purpose']}")
        print(f"   Type: {dataset['type']}")
        print(f"   Paper: {dataset['paper']}")
        print(f"   Status: {dataset['result']['status']}")
        
        if dataset['result']['status'] == 'success':
            metrics = dataset['result']['metrics']
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {metric}: {value:.3f}")
                else:
                    print(f"   {metric}: {value}")
    
    return {
        "agentharm": agentharm_result,
        "safetybench": safetybench_result
    }


async def demonstrate_advanced_usage():
    """Demonstrate advanced usage patterns."""
    print("\\n🎯 Advanced Research Dataset Usage")
    print("=" * 45)
    
    # Example 1: Category-specific evaluation
    print("\\n1️⃣ Category-Specific Evaluation:")
    try:
        evaluator = AgentHarmHFEvaluator(
            subset="harmful",
            categories=["Cybercrime", "Sexual"],  # Focus on specific categories
            sample_size=3
        )
        
        info = evaluator.get_dataset_info()
        print(f"   Filtered to categories: {info['categories_filter']}")
        print(f"   Samples matching filter: {info['total_samples']}")
        
    except Exception as e:
        print(f"   ⚠️ Category filtering: {e}")
    
    # Example 2: Different dataset splits
    print("\\n2️⃣ Dataset Split Comparison:")
    splits_to_test = ["test_public", "validation"]
    
    for split in splits_to_test:
        try:
            evaluator = AgentHarmHFEvaluator(
                subset="harmful",
                split=split,
                sample_size=2
            )
            
            info = evaluator.get_dataset_info()
            print(f"   {split}: {info['total_samples']} samples")
            
        except Exception as e:
            print(f"   ⚠️ {split}: {e}")
    
    # Example 3: Multilingual evaluation
    print("\\n3️⃣ Multilingual Safety Evaluation:")
    languages = ["en", "zh"]
    
    for lang in languages:
        try:
            evaluator = SafetyBenchHFEvaluator(
                language=lang,
                sample_size=2
            )
            
            info = evaluator.get_dataset_info()
            print(f"   {lang}: {info['total_samples']} samples, {len(info['categories'])} categories")
            
        except Exception as e:
            print(f"   ⚠️ {lang}: {e}")


async def save_research_results(results: dict):
    """Save research dataset evaluation results."""
    timestamp = datetime.now()
    
    research_results = {
        "evaluation_timestamp": timestamp.isoformat(),
        "datasets_evaluated": ["AgentHarm", "SafetyBench"],
        "evaluation_results": results,
        "summary": {
            "total_datasets": len(results),
            "successful_evaluations": sum(1 for r in results.values() if r["status"] == "success"),
            "datasets_with_metrics": [k for k, v in results.items() if "metrics" in v]
        },
        "research_integration": {
            "agentharm_paper": "https://arxiv.org/abs/2410.09024",
            "agentharm_dataset": "https://huggingface.co/datasets/ai-safety-institute/AgentHarm",
            "safetybench_dataset": "https://huggingface.co/datasets/thu-coai/SafetyBench",
            "integration_status": "operational"
        }
    }
    
    # Save to results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"research_datasets_evaluation_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(research_results, f, indent=2)
    
    print(f"\\n💾 Research results saved to: {results_file}")


async def main():
    """Run comprehensive research dataset demonstration."""
    print("🧬 LLM Evaluation with Research Datasets")
    print("=" * 60)
    print("This demo shows integration with real research paper datasets")
    print("from Hugging Face, enabling reproducible scientific evaluation.")
    
    # Run comprehensive comparison
    results = await compare_datasets()
    
    # Demonstrate advanced features
    await demonstrate_advanced_usage()
    
    # Save results
    await save_research_results(results)
    
    print("\\n" + "=" * 60)
    print("🎉 Research Dataset Integration Complete!")
    print("\\n🔬 Key Benefits:")
    print("   ✅ Real research datasets from published papers")
    print("   ✅ Reproducible evaluation protocols")
    print("   ✅ Multiple evaluation methodologies")
    print("   ✅ Direct comparison with research baselines")
    
    print("\\n📚 Available Research Datasets:")
    print("   • AgentHarm: Agent safety evaluation (arxiv:2410.09024)")
    print("   • SafetyBench: Multi-choice safety questions")
    print("   • More datasets can be easily integrated!")
    
    print("\\n💡 Usage in your evaluations:")
    print("   from llm_eval.frameworks.safety.agentharm_hf import AgentHarmHFEvaluator")
    print("   evaluator = AgentHarmHFEvaluator(subset='harmful', sample_size=50)")
    print("   results = await evaluator.evaluate(your_model)")


if __name__ == "__main__":
    asyncio.run(main())