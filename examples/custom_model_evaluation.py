"""Example of evaluating custom models and comparing different providers."""

import asyncio
import os
from pathlib import Path

from llm_eval import EvaluationOrchestrator, EvaluationConfig, create_model


async def evaluate_single_model(model_id: str, config: EvaluationConfig, **api_keys) -> dict:
    """Evaluate a single model and return results."""
    print(f"\n🤖 Evaluating: {model_id}")
    
    try:
        # Create model instance
        model = create_model(model_id, **api_keys)
        
        # Test connectivity
        test_response = await model.generate("Test connectivity", max_tokens=10)
        print(f"   ✅ Connectivity test passed")
        
        # Run evaluation
        orchestrator = EvaluationOrchestrator(config)
        results = await orchestrator.run_evaluation(model)
        
        print(f"   📊 Overall Score: {results.overall_score:.3f}")
        print(f"   🛡️ Safety Score: {sum(results.safety_metrics.values())/len(results.safety_metrics) if results.safety_metrics else 0:.3f}")
        print(f"   🔒 Security Score: {sum(results.security_metrics.values())/len(results.security_metrics) if results.security_metrics else 0:.3f}")
        print(f"   🔧 Reliability Score: {sum(results.reliability_metrics.values())/len(results.reliability_metrics) if results.reliability_metrics else 0:.3f}")
        print(f"   ⏱️ Execution Time: {results.total_execution_time:.2f}s")
        
        return {
            'model_id': model_id,
            'results': results,
            'success': True
        }
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return {
            'model_id': model_id,
            'results': None,
            'success': False,
            'error': str(e)
        }


async def main():
    """Compare multiple models across different providers."""
    print("🔍 LLM Evaluation Pipeline - Multi-Model Comparison")
    print("=" * 60)
    
    # Load configuration (using medium risk for faster evaluation)
    config_path = Path(__file__).parent.parent / "configs" / "medium_risk.yaml"
    config = EvaluationConfig.from_yaml(str(config_path))
    
    # Adjust for comparison (smaller sample for faster execution)
    config.sample_size = 30
    config.max_concurrent_requests = 2
    
    print(f"📝 Configuration: {config.risk_level.value} risk, sample size: {config.sample_size}")
    
    # Define models to compare
    models_to_evaluate = [
        # Groq models (open source)
        "groq/llama-3.2-11b-text-preview",
        "groq/mixtral-8x7b-32768",
    ]
    
    # Add OpenAI and Anthropic models if API keys are available
    if os.getenv('OPENAI_API_KEY'):
        models_to_evaluate.append("openai/gpt-3.5-turbo")
        print("   🔑 OpenAI API key found - including GPT-3.5-turbo")
    else:
        print("   ⚠️ OpenAI API key not found - skipping OpenAI models")
        print("      Set OPENAI_API_KEY to include OpenAI models")
    
    if os.getenv('ANTHROPIC_API_KEY'):
        models_to_evaluate.append("anthropic/claude-3-haiku-20240307")
        print("   🔑 Anthropic API key found - including Claude-3-Haiku")
    else:
        print("   ⚠️ Anthropic API key not found - skipping Anthropic models")
        print("      Set ANTHROPIC_API_KEY to include Anthropic models")
    
    if not os.getenv('GROQ_API_KEY'):
        print("❌ GROQ_API_KEY not found - this is required for the basic models")
        print("   Set GROQ_API_KEY to run this example")
        return
    
    # Prepare API keys
    api_keys = {
        'groq_api_key': os.getenv('GROQ_API_KEY'),
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY')
    }
    
    print(f"\n🎯 Models to evaluate: {len(models_to_evaluate)}")
    for model in models_to_evaluate:
        print(f"   • {model}")
    
    # Evaluate all models
    print(f"\n🏃 Starting evaluation of {len(models_to_evaluate)} models...")
    evaluation_results = []
    
    for model_id in models_to_evaluate:
        result = await evaluate_single_model(model_id, config, **api_keys)
        evaluation_results.append(result)
    
    # Analyze and compare results
    successful_results = [r for r in evaluation_results if r['success']]
    failed_results = [r for r in evaluation_results if not r['success']]
    
    print(f"\n" + "=" * 60)
    print(f"📊 COMPARISON RESULTS")
    print(f"=" * 60)
    print(f"Successful evaluations: {len(successful_results)}/{len(evaluation_results)}")
    
    if failed_results:
        print(f"\n❌ Failed evaluations:")
        for result in failed_results:
            print(f"   {result['model_id']}: {result['error']}")
    
    if successful_results:
        print(f"\n🏆 MODEL RANKINGS:")
        
        # Sort by overall score
        successful_results.sort(key=lambda x: x['results'].overall_score, reverse=True)
        
        print(f"\n{'Rank':<4} {'Model':<35} {'Overall':<8} {'Safety':<8} {'Security':<8} {'Reliability':<10} {'Time':<6}")
        print("-" * 85)
        
        for i, result in enumerate(successful_results, 1):
            results = result['results']
            safety_avg = sum(results.safety_metrics.values())/len(results.safety_metrics) if results.safety_metrics else 0
            security_avg = sum(results.security_metrics.values())/len(results.security_metrics) if results.security_metrics else 0
            reliability_avg = sum(results.reliability_metrics.values())/len(results.reliability_metrics) if results.reliability_metrics else 0
            
            print(f"{i:<4} {result['model_id']:<35} {results.overall_score:<8.3f} {safety_avg:<8.3f} {security_avg:<8.3f} {reliability_avg:<10.3f} {results.total_execution_time:<6.1f}s")
        
        # Detailed comparison
        print(f"\n📈 DETAILED ANALYSIS:")
        
        best_overall = successful_results[0]
        print(f"   🥇 Best Overall: {best_overall['model_id']} (score: {best_overall['results'].overall_score:.3f})")
        
        # Find best in each category
        best_safety = max(successful_results, key=lambda x: sum(x['results'].safety_metrics.values())/len(x['results'].safety_metrics) if x['results'].safety_metrics else 0)
        best_security = max(successful_results, key=lambda x: sum(x['results'].security_metrics.values())/len(x['results'].security_metrics) if x['results'].security_metrics else 0)
        best_reliability = max(successful_results, key=lambda x: sum(x['results'].reliability_metrics.values())/len(x['results'].reliability_metrics) if x['results'].reliability_metrics else 0)
        fastest = min(successful_results, key=lambda x: x['results'].total_execution_time)
        
        print(f"   🛡️ Best Safety: {best_safety['model_id']}")
        print(f"   🔒 Best Security: {best_security['model_id']}")
        print(f"   🔧 Best Reliability: {best_reliability['model_id']}")
        print(f"   ⚡ Fastest: {fastest['model_id']} ({fastest['results'].total_execution_time:.1f}s)")
        
        # Risk compliance analysis
        compliant_models = [r for r in successful_results if r['results'].risk_compliance]
        print(f"\n✅ Risk Compliant Models: {len(compliant_models)}/{len(successful_results)}")
        for result in compliant_models:
            print(f"   • {result['model_id']}")
        
        # Save comparison results
        print(f"\n💾 Saving comparison results...")
        
        # Save individual results
        for result in successful_results:
            filename = f"comparison_{result['model_id'].replace('/', '_')}.json"
            result['results'].to_json(filename)
            print(f"   📄 {filename}")
        
        # Create comparison report
        from llm_eval.core.results import EvaluationResults
        
        # Generate comparison summary
        comparison_summary = {
            'timestamp': successful_results[0]['results'].timestamp.isoformat(),
            'config': config.__dict__,
            'models_evaluated': len(successful_results),
            'rankings': [
                {
                    'rank': i+1,
                    'model_id': result['model_id'],
                    'overall_score': result['results'].overall_score,
                    'risk_compliant': result['results'].risk_compliance,
                    'execution_time': result['results'].total_execution_time
                }
                for i, result in enumerate(successful_results)
            ]
        }
        
        import json
        with open('model_comparison_summary.json', 'w') as f:
            json.dump(comparison_summary, f, indent=2)
        print(f"   📊 model_comparison_summary.json")
        
        print(f"\n🎉 Multi-model evaluation completed!")
        print(f"   Use the results to choose the best model for your specific needs.")
        print(f"   Consider overall score, risk compliance, and execution time.")


if __name__ == "__main__":
    asyncio.run(main())