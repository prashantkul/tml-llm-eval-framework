"""Batch evaluation example for processing multiple models systematically."""

import asyncio
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from llm_eval import EvaluationOrchestrator, EvaluationConfig, create_model


class BatchEvaluator:
    """Handles batch evaluation of multiple models."""
    
    def __init__(self, config: EvaluationConfig, output_dir: str = "batch_results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # API keys
        self.api_keys = {
            'groq_api_key': os.getenv('GROQ_API_KEY'),
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY')
        }
    
    async def evaluate_model_batch(self, model_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate a batch of models."""
        results = []
        
        print(f"🚀 Starting batch evaluation of {len(model_configs)} models...")
        
        for i, model_config in enumerate(model_configs, 1):
            model_id = model_config['model_id']
            print(f"\n[{i}/{len(model_configs)}] Evaluating {model_id}...")
            
            try:
                # Create model
                model = create_model(model_id, **self.api_keys)
                
                # Test connectivity
                await model.generate("Test", max_tokens=5)
                
                # Create orchestrator
                orchestrator = EvaluationOrchestrator(self.config)
                
                # Run evaluation with custom frameworks if specified
                if 'frameworks' in model_config:
                    frameworks = model_config['frameworks']
                    evaluation_results = await orchestrator.run_framework_subset(model, frameworks)
                else:
                    evaluation_results = await orchestrator.run_evaluation(model)
                
                # Save individual results
                filename = f"{model_id.replace('/', '_')}_results.json"
                filepath = self.output_dir / filename
                evaluation_results.to_json(str(filepath))
                
                result_summary = {
                    'model_id': model_id,
                    'status': 'success',
                    'overall_score': evaluation_results.overall_score,
                    'risk_compliant': evaluation_results.risk_compliance,
                    'execution_time': evaluation_results.total_execution_time,
                    'safety_score': self._calculate_category_average(evaluation_results.safety_metrics),
                    'security_score': self._calculate_category_average(evaluation_results.security_metrics),
                    'reliability_score': self._calculate_category_average(evaluation_results.reliability_metrics),
                    'results_file': str(filepath),
                    'timestamp': evaluation_results.timestamp.isoformat()
                }
                
                print(f"   ✅ Success - Score: {evaluation_results.overall_score:.3f}")
                
            except Exception as e:
                result_summary = {
                    'model_id': model_id,
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                print(f"   ❌ Failed: {e}")
            
            results.append(result_summary)
        
        return results
    
    def _calculate_category_average(self, metrics: Dict[str, float]) -> float:
        """Calculate average score for a category."""
        if not metrics:
            return 0.0
        return sum(metrics.values()) / len(metrics)
    
    def generate_batch_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive batch evaluation report."""
        successful_results = [r for r in results if r['status'] == 'success']
        failed_results = [r for r in results if r['status'] == 'failed']
        
        report = []
        report.append("# Batch Evaluation Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Configuration: {self.config.risk_level.value} risk")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- Total models evaluated: {len(results)}")
        report.append(f"- Successful evaluations: {len(successful_results)}")
        report.append(f"- Failed evaluations: {len(failed_results)}")
        report.append(f"- Success rate: {len(successful_results)/len(results)*100:.1f}%")
        report.append("")
        
        if successful_results:
            # Rankings
            successful_results.sort(key=lambda x: x['overall_score'], reverse=True)
            
            report.append("## Model Rankings")
            report.append("| Rank | Model | Overall Score | Safety | Security | Reliability | Risk Compliant | Time (s) |")
            report.append("|------|-------|---------------|--------|----------|-------------|----------------|----------|")
            
            for i, result in enumerate(successful_results, 1):
                compliance = "✅" if result['risk_compliant'] else "❌"
                report.append(
                    f"| {i} | {result['model_id']} | {result['overall_score']:.3f} | "
                    f"{result['safety_score']:.3f} | {result['security_score']:.3f} | "
                    f"{result['reliability_score']:.3f} | {compliance} | {result['execution_time']:.1f} |"
                )
            
            report.append("")
            
            # Best performers
            report.append("## Best Performers")
            best_overall = max(successful_results, key=lambda x: x['overall_score'])
            best_safety = max(successful_results, key=lambda x: x['safety_score'])
            best_security = max(successful_results, key=lambda x: x['security_score'])
            best_reliability = max(successful_results, key=lambda x: x['reliability_score'])
            fastest = min(successful_results, key=lambda x: x['execution_time'])
            
            report.append(f"- **Best Overall**: {best_overall['model_id']} ({best_overall['overall_score']:.3f})")
            report.append(f"- **Best Safety**: {best_safety['model_id']} ({best_safety['safety_score']:.3f})")
            report.append(f"- **Best Security**: {best_security['model_id']} ({best_security['security_score']:.3f})")
            report.append(f"- **Best Reliability**: {best_reliability['model_id']} ({best_reliability['reliability_score']:.3f})")
            report.append(f"- **Fastest**: {fastest['model_id']} ({fastest['execution_time']:.1f}s)")
            report.append("")
            
            # Risk compliance
            compliant_models = [r for r in successful_results if r['risk_compliant']]
            report.append("## Risk Compliance")
            report.append(f"Models compliant with {self.config.risk_level.value} risk requirements: {len(compliant_models)}/{len(successful_results)}")
            if compliant_models:
                for model in compliant_models:
                    report.append(f"- {model['model_id']}")
            report.append("")
        
        if failed_results:
            report.append("## Failed Evaluations")
            for result in failed_results:
                report.append(f"- **{result['model_id']}**: {result['error']}")
            report.append("")
        
        return "\n".join(report)


async def main():
    """Run batch evaluation example."""
    print("📋 LLM Evaluation Pipeline - Batch Evaluation")
    print("=" * 50)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "medium_risk.yaml"
    config = EvaluationConfig.from_yaml(str(config_path))
    
    # Adjust for batch processing
    config.sample_size = 25  # Smaller for faster batch processing
    config.max_concurrent_requests = 2
    
    print(f"📝 Configuration: {config.risk_level.value} risk, sample size: {config.sample_size}")
    
    # Define model batch
    model_batch = [
        {
            'model_id': 'groq/llama-3.2-11b-text-preview',
            'frameworks': {
                'safety': ['agentharm'],
                'security': ['houyi'],
                'reliability': ['selfprompt']
            }
        },
        {
            'model_id': 'groq/mixtral-8x7b-32768',
            # Full evaluation (no frameworks specified)
        }
    ]
    
    # Add OpenAI model if API key available
    if os.getenv('OPENAI_API_KEY'):
        model_batch.append({
            'model_id': 'openai/gpt-3.5-turbo',
            'frameworks': {
                'safety': ['agent_safetybench'],
                'security': ['cia_attacks']
            }
        })
        print("   🔑 Including OpenAI model in batch")
    
    # Add Anthropic model if API key available
    if os.getenv('ANTHROPIC_API_KEY'):
        model_batch.append({
            'model_id': 'anthropic/claude-3-haiku-20240307'
        })
        print("   🔑 Including Anthropic model in batch")
    
    if not os.getenv('GROQ_API_KEY'):
        print("❌ GROQ_API_KEY required for batch evaluation")
        return
    
    print(f"\n🎯 Batch size: {len(model_batch)} models")
    
    # Create batch evaluator
    evaluator = BatchEvaluator(config)
    
    # Run batch evaluation
    try:
        results = await evaluator.evaluate_model_batch(model_batch)
        
        # Generate and save batch report
        report = evaluator.generate_batch_report(results)
        
        report_file = evaluator.output_dir / "batch_evaluation_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save results summary
        summary_file = evaluator.output_dir / "batch_results_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n" + "=" * 50)
        print("📊 BATCH EVALUATION COMPLETED")
        print("=" * 50)
        
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        print(f"✅ Successful: {len(successful)}/{len(results)}")
        print(f"❌ Failed: {len(failed)}/{len(results)}")
        
        if successful:
            best = max(successful, key=lambda x: x['overall_score'])
            print(f"🏆 Best performer: {best['model_id']} (score: {best['overall_score']:.3f})")
        
        print(f"\n📁 Results saved to: {evaluator.output_dir}")
        print(f"📄 Report: {report_file}")
        print(f"📊 Summary: {summary_file}")
        
        # Print the report to console
        print(f"\n{report}")
        
    except Exception as e:
        print(f"❌ Batch evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())