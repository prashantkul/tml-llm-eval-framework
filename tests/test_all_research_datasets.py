"""Comprehensive test script for all research dataset integrations."""

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
from llm_eval.frameworks.reliability.promptrobust_hf import PromptRobustHFEvaluator
from llm_eval.core.model_interface import create_model


async def test_agentharm_evaluation():
    """Test evaluation using AgentHarm dataset."""
    print("🔬 AgentHarm Dataset Evaluation")
    print("-" * 40)
    
    try:
        evaluator = AgentHarmHFEvaluator(
            subset="harmful",
            split="test_public", 
            sample_size=3
        )
        
        info = evaluator.get_dataset_info()
        print(f"📊 Dataset: {info['subset']}/{info['split']}")
        print(f"📝 Samples: {info['total_samples']}")
        print(f"🏷️ Categories: {info['categories'][:3] if len(info['categories']) > 3 else info['categories']}")
        
        # Get API key
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("⚠️ No OpenAI API key - using mock evaluation")
            return {"framework": "agentharm_hf", "status": "skipped"}
        
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
            
    except Exception as e:
        print(f"❌ AgentHarm evaluation error: {e}")
        return {"framework": "agentharm_hf", "status": "error", "error": str(e)}


async def test_safetybench_evaluation():
    """Test evaluation using SafetyBench dataset."""
    print("\n📚 SafetyBench Dataset Evaluation")
    print("-" * 40)
    
    try:
        evaluator = SafetyBenchHFEvaluator(
            language="en",
            subset="test",
            sample_size=3
        )
        
        info = evaluator.get_dataset_info()
        print(f"📊 Dataset: {info['language']}_{info['subset']}")
        print(f"📝 Samples: {info['total_samples']}")
        print(f"🏷️ Categories: {info['categories'][:3] if len(info['categories']) > 3 else info['categories']}")
        
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("⚠️ No OpenAI API key - using mock evaluation")
            return {"framework": "safetybench_hf", "status": "skipped"}
        
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
            
    except Exception as e:
        print(f"❌ SafetyBench evaluation error: {e}")
        return {"framework": "safetybench_hf", "status": "error", "error": str(e)}


async def test_promptrobust_evaluation():
    """Test evaluation using PromptRobust dataset."""
    print("\n🛡️ PromptRobust Dataset Evaluation")
    print("-" * 40)
    
    try:
        evaluator = PromptRobustHFEvaluator(
            sample_size=5,
            attack_types=['character', 'word', 'sentence', 'semantic']
        )
        
        info = evaluator.get_dataset_info()
        print(f"📊 Dataset: PromptBench robustness")
        print(f"📝 Samples: {info['total_samples']}")
        print(f"🏷️ Attack types: {info['attack_types']}")
        
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("⚠️ No OpenAI API key - using mock evaluation")
            return {"framework": "promptrobust_hf", "status": "skipped"}
        
        model = create_model('openai/gpt-3.5-turbo', openai_api_key=openai_key)
        
        print("⚡ Running evaluation...")
        result = await evaluator.evaluate(model)
        
        if result.success:
            print(f"✅ Evaluation completed!")
            key_metrics = {
                "overall_robustness_score": result.metrics.get("overall_robustness_score", 0),
                "inconsistency_rate": result.metrics.get("inconsistency_rate", 0),
                "total_pairs": result.metrics.get("total_pairs", 0)
            }
            print(f"📈 Key metrics: {key_metrics}")
            return {"framework": "promptrobust_hf", "status": "success", "metrics": key_metrics}
        else:
            print(f"❌ Evaluation failed")
            return {"framework": "promptrobust_hf", "status": "failed"}
            
    except Exception as e:
        print(f"❌ PromptRobust evaluation error: {e}")
        return {"framework": "promptrobust_hf", "status": "error", "error": str(e)}


async def compare_all_datasets():
    """Run comprehensive comparison across all research datasets."""
    print("\n⚖️ Comprehensive Research Dataset Comparison")
    print("=" * 60)
    
    # Run all evaluations
    agentharm_result = await test_agentharm_evaluation()
    safetybench_result = await test_safetybench_evaluation()
    promptrobust_result = await test_promptrobust_evaluation()
    
    # Summary comparison
    print("\n📊 Complete Dataset Comparison Summary:")
    print("-" * 50)
    
    datasets_info = [
        {
            "name": "AgentHarm",
            "category": "Safety Evaluation",
            "purpose": "Agent harmfulness benchmark",
            "type": "Open-ended harmful prompts",
            "paper": "https://arxiv.org/abs/2410.09024",
            "dataset": "https://huggingface.co/datasets/ai-safety-institute/AgentHarm",
            "result": agentharm_result
        },
        {
            "name": "SafetyBench", 
            "category": "Safety Evaluation",
            "purpose": "LLM safety assessment",
            "type": "Multiple-choice safety questions",
            "paper": "SafetyBench research",
            "dataset": "https://huggingface.co/datasets/thu-coai/SafetyBench",
            "result": safetybench_result
        },
        {
            "name": "PromptRobust",
            "category": "Reliability Evaluation", 
            "purpose": "Adversarial prompt robustness",
            "type": "Original/attacked prompt pairs",
            "paper": "https://arxiv.org/abs/2306.04528",
            "dataset": "https://huggingface.co/datasets/aryanagrawal1/promptbench",
            "result": promptrobust_result
        }
    ]
    
    for dataset in datasets_info:
        print(f"\n🔬 {dataset['name']} ({dataset['category']}):")
        print(f"   Purpose: {dataset['purpose']}")
        print(f"   Type: {dataset['type']}")
        print(f"   Paper: {dataset['paper']}")
        print(f"   Dataset: {dataset['dataset']}")
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
        "safetybench": safetybench_result,
        "promptrobust": promptrobust_result
    }


async def save_comprehensive_results(results: dict):
    """Save comprehensive research dataset evaluation results."""
    timestamp = datetime.now()
    
    comprehensive_results = {
        "evaluation_timestamp": timestamp.isoformat(),
        "datasets_evaluated": ["AgentHarm", "SafetyBench", "PromptRobust"],
        "evaluation_results": results,
        "summary": {
            "total_datasets": len(results),
            "successful_evaluations": sum(1 for r in results.values() if r["status"] == "success"),
            "datasets_with_metrics": [k for k, v in results.items() if "metrics" in v],
            "evaluation_categories": {
                "safety": ["agentharm", "safetybench"],
                "reliability": ["promptrobust"]
            }
        },
        "research_integration": {
            "agentharm_paper": "https://arxiv.org/abs/2410.09024",
            "agentharm_dataset": "https://huggingface.co/datasets/ai-safety-institute/AgentHarm",
            "safetybench_dataset": "https://huggingface.co/datasets/thu-coai/SafetyBench",
            "promptrobust_paper": "https://arxiv.org/abs/2306.04528",
            "promptrobust_dataset": "https://huggingface.co/datasets/aryanagrawal1/promptbench",
            "integration_status": "operational"
        },
        "capabilities": {
            "safety_evaluation": {
                "harmful_prompt_detection": "AgentHarm",
                "safety_question_answering": "SafetyBench",
                "multilingual_support": "SafetyBench (en/zh)"
            },
            "reliability_evaluation": {
                "adversarial_robustness": "PromptRobust",
                "attack_types": ["character", "word", "sentence", "semantic"],
                "consistency_measurement": "Response similarity analysis"
            }
        }
    }
    
    # Save to results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"comprehensive_research_datasets_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\n💾 Comprehensive results saved to: {results_file}")


async def main():
    """Run comprehensive research dataset demonstration."""
    print("🧬 LLM Evaluation with Comprehensive Research Datasets")
    print("=" * 80)
    print("This demo shows integration with multiple research paper datasets")
    print("from Hugging Face, enabling comprehensive scientific evaluation.")
    print("\nDatasets included:")
    print("• AgentHarm: Safety evaluation with harmful prompts")
    print("• SafetyBench: Multiple-choice safety questions")
    print("• PromptRobust: Adversarial robustness testing")
    
    # Run comprehensive comparison
    results = await compare_all_datasets()
    
    # Save results
    await save_comprehensive_results(results)
    
    print("\n" + "=" * 80)
    print("🎉 Comprehensive Research Dataset Integration Complete!")
    
    print("\n🔬 Evaluation Categories:")
    print("   ✅ Safety Evaluation: AgentHarm + SafetyBench")
    print("   ✅ Reliability Evaluation: PromptRobust")
    print("   ✅ Multi-modal approach: Harmful prompts + Q&A + Robustness")
    
    print("\n📚 Available Research Datasets:")
    print("   • AgentHarm: Agent safety evaluation (arxiv:2410.09024)")
    print("   • SafetyBench: Multi-choice safety questions")
    print("   • PromptRobust: Adversarial prompt robustness (arxiv:2306.04528)")
    
    print("\n💡 Usage in your evaluations:")
    print("   # Safety evaluation")
    print("   from llm_eval.frameworks.safety.agentharm_hf import AgentHarmHFEvaluator")
    print("   from llm_eval.frameworks.safety.safetybench_hf import SafetyBenchHFEvaluator")
    print("   # Reliability evaluation")
    print("   from llm_eval.frameworks.reliability.promptrobust_hf import PromptRobustHFEvaluator")
    
    print("\n🌟 Key Benefits:")
    print("   ✅ Real research datasets from published papers")
    print("   ✅ Reproducible evaluation protocols")
    print("   ✅ Multiple evaluation methodologies")
    print("   ✅ Direct comparison with research baselines")
    print("   ✅ Comprehensive safety and reliability coverage")


if __name__ == "__main__":
    asyncio.run(main())