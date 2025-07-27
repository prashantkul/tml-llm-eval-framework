"""Basic evaluation example using the LLM evaluation pipeline."""

import asyncio
import os
from pathlib import Path

from llm_eval import EvaluationOrchestrator, EvaluationConfig, create_model


async def main():
    """Run a basic evaluation example."""
    print("🚀 LLM Evaluation Pipeline - Basic Example")
    print("=" * 50)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "medium_risk.yaml"
    config = EvaluationConfig.from_yaml(str(config_path))
    print(f"📝 Loaded configuration: {config.risk_level.value} risk")
    
    # Configure for quick demo (smaller sample size)
    config.sample_size = 20
    print(f"🔧 Configured for demo with sample size: {config.sample_size}")
    
    # Create model instance (using environment variable for API key)
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ Please set GROQ_API_KEY environment variable")
        print("   export GROQ_API_KEY='your_api_key'")
        return
    
    try:
        model = create_model('groq/llama-3.2-11b-text-preview', groq_api_key=api_key)
        print(f"🤖 Created model: {model.get_model_info()}")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return
    
    # Create orchestrator
    orchestrator = EvaluationOrchestrator(config)
    available_frameworks = orchestrator.get_available_frameworks()
    print(f"🔍 Available frameworks: {sum(len(f) for f in available_frameworks.values())}")
    
    # Run evaluation
    print("\n🏃 Starting evaluation...")
    try:
        results = await orchestrator.run_evaluation(model)
        
        # Print results summary
        print("\n" + "=" * 50)
        print("📊 EVALUATION RESULTS")
        print("=" * 50)
        print(f"Model: {results.model_name}")
        print(f"Overall Score: {results.overall_score:.3f}")
        print(f"Risk Compliance: {'✅ PASS' if results.risk_compliance else '❌ FAIL'}")
        print(f"Execution Time: {results.total_execution_time:.2f}s")
        
        # Safety metrics
        if results.safety_metrics:
            print(f"\n🛡️ Safety Metrics:")
            for metric, value in results.safety_metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Security metrics
        if results.security_metrics:
            print(f"\n🔒 Security Metrics:")
            for metric, value in results.security_metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Reliability metrics
        if results.reliability_metrics:
            print(f"\n🔧 Reliability Metrics:")
            for metric, value in results.reliability_metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Framework failures
        if results.framework_failures:
            print(f"\n⚠️ Framework Failures:")
            for failure in results.framework_failures:
                print(f"  ❌ {failure}")
        
        # Save results
        output_file = "basic_evaluation_results.json"
        results.to_json(output_file)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Generate report
        report_file = "basic_evaluation_report.html"
        results.save_report(report_file, format="html")
        print(f"📄 HTML report saved to: {report_file}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())