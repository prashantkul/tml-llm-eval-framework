"""Specific example for evaluating Llama-4-Scout via Groq."""

import asyncio
import os
from pathlib import Path

from llm_eval import EvaluationOrchestrator, EvaluationConfig, GroqModel


async def main():
    """Comprehensive evaluation of Llama-4-Scout model."""
    print("🦙 LLM Evaluation Pipeline - Llama-4-Scout Evaluation")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ Please set GROQ_API_KEY environment variable")
        print("   export GROQ_API_KEY='your_groq_api_key'")
        print("   Get your key from: https://console.groq.com/keys")
        return
    
    # Load high-risk configuration for comprehensive testing
    config_path = Path(__file__).parent.parent / "configs" / "high_risk.yaml"
    config = EvaluationConfig.from_yaml(str(config_path))
    
    # Adjust for demo (reduce sample size for faster execution)
    config.sample_size = 50
    config.max_concurrent_requests = 3  # Be conservative with Groq rate limits
    
    print(f"📝 Configuration: {config.risk_level.value} risk")
    print(f"🔢 Sample size: {config.sample_size}")
    print(f"⚡ Max concurrent requests: {config.max_concurrent_requests}")
    
    # Create Llama-4-Scout model instance
    model_name = "llama-4-scout-17b-16e-instruct"
    
    try:
        model = GroqModel(model_name, api_key)
        model_info = model.get_model_info()
        print(f"\n🤖 Model Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"❌ Failed to initialize Llama-4-Scout model: {e}")
        print("   Make sure the model name is correct and your API key is valid")
        return
    
    # Test basic model functionality
    print(f"\n🧪 Testing model connectivity...")
    try:
        test_response = await model.generate("Hello, this is a test. Please respond briefly.", max_tokens=50)
        print(f"✅ Model test successful: {test_response[:100]}...")
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return
    
    # Create orchestrator and show available frameworks
    orchestrator = EvaluationOrchestrator(config)
    frameworks = orchestrator.get_available_frameworks()
    
    print(f"\n🔍 Evaluation Frameworks:")
    for category, framework_list in frameworks.items():
        print(f"  {category.upper()}: {', '.join(framework_list)}")
    
    # Run comprehensive evaluation
    print(f"\n🏃 Starting comprehensive evaluation of {model_name}...")
    print("   This may take several minutes...")
    
    try:
        results = await orchestrator.run_evaluation(model)
        
        # Print comprehensive results
        print("\n" + "=" * 60)
        print("📊 LLAMA-4-SCOUT EVALUATION RESULTS")
        print("=" * 60)
        print(f"Model: {results.model_name}")
        print(f"Timestamp: {results.timestamp}")
        print(f"Overall Score: {results.overall_score:.3f}")
        print(f"Risk Compliance: {'✅ PASS' if results.risk_compliance else '❌ FAIL'}")
        print(f"Total Execution Time: {results.total_execution_time:.2f}s")
        
        # Detailed metrics breakdown
        print(f"\n🛡️ SAFETY ASSESSMENT:")
        print(f"   Frameworks run: {len(results.safety_results)}")
        for metric, value in results.safety_metrics.items():
            status = "✅" if value <= config.safety_thresholds.get(metric, 1.0) else "❌"
            print(f"   {metric}: {value:.4f} {status}")
        
        print(f"\n🔒 SECURITY ASSESSMENT:")
        print(f"   Frameworks run: {len(results.security_results)}")
        for metric, value in results.security_metrics.items():
            threshold = config.security_thresholds.get(metric, 0.0)
            if "resistance" in metric.lower():
                status = "✅" if value >= threshold else "❌"
            else:
                status = "✅" if value <= threshold else "❌"
            print(f"   {metric}: {value:.4f} {status}")
        
        print(f"\n🔧 RELIABILITY ASSESSMENT:")
        print(f"   Frameworks run: {len(results.reliability_results)}")
        for metric, value in results.reliability_metrics.items():
            threshold = config.reliability_thresholds.get(metric, 0.0)
            if "consistency" in metric.lower():
                status = "✅" if value >= threshold else "❌"
            else:
                status = "✅" if value <= threshold else "❌"
            print(f"   {metric}: {value:.4f} {status}")
        
        # Framework performance
        print(f"\n⚡ FRAMEWORK PERFORMANCE:")
        all_results = results.safety_results + results.security_results + results.reliability_results
        successful = sum(1 for r in all_results if r.success)
        total = len(all_results)
        print(f"   Successful frameworks: {successful}/{total}")
        print(f"   Average execution time: {sum(r.execution_time for r in all_results)/total:.2f}s")
        
        if results.framework_failures:
            print(f"\n⚠️ FRAMEWORK FAILURES:")
            for failure in results.framework_failures:
                print(f"   ❌ {failure}")
        
        # Risk assessment summary
        print(f"\n📋 RISK ASSESSMENT SUMMARY:")
        compliance_details = results.check_thresholds()
        passed = sum(1 for v in compliance_details.values() if v)
        total_checks = len(compliance_details)
        print(f"   Compliance checks passed: {passed}/{total_checks}")
        print(f"   Overall risk compliance: {'✅ SUITABLE' if results.risk_compliance else '❌ NOT SUITABLE'} for {config.risk_level.value} risk applications")
        
        # Save detailed results
        output_file = f"llama_scout_evaluation_{config.risk_level.value}_risk.json"
        results.to_json(output_file)
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # Generate comprehensive report
        report_file = f"llama_scout_report_{config.risk_level.value}_risk.html"
        results.save_report(report_file, format="html")
        print(f"📄 HTML report saved to: {report_file}")
        
        # Save text summary
        summary_file = f"llama_scout_summary_{config.risk_level.value}_risk.txt"
        results.save_report(summary_file, format="text")
        print(f"📄 Text summary saved to: {summary_file}")
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"   Review the reports to understand Llama-4-Scout's performance")
        print(f"   across safety, security, and reliability dimensions.")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())