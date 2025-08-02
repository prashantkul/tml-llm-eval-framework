"""Final comprehensive test of the evaluation pipeline."""

import asyncio
import os
import sys
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

from llm_eval import EvaluationOrchestrator, EvaluationConfig, create_model


async def run_final_test():
    """Run a final comprehensive test with minimal settings."""
    print("🚀 Final Comprehensive Test - LLM Evaluation Pipeline")
    print("=" * 70)
    
    # Load the low-risk config we created
    config = EvaluationConfig.from_yaml("test_config.yaml")
    
    # Override with minimal settings for speed
    config.sample_size = 3
    config.max_concurrent_requests = 2
    config.timeout_seconds = 60
    
    print(f"📝 Configuration: {config.risk_level.value} risk, {config.sample_size} samples")
    
    # Create OpenAI model
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ No OpenAI API key found")
        return
    
    model = create_model('openai/gpt-3.5-turbo', openai_api_key=openai_key)
    print(f"🤖 Model: {model.model_name}")
    
    # Run evaluation
    orchestrator = EvaluationOrchestrator(config)
    print("\n⏳ Running full evaluation pipeline...")
    
    start_time = datetime.now()
    try:
        results = await orchestrator.run_evaluation(model)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print("\n🎉 EVALUATION COMPLETED!")
        print("=" * 50)
        print(f"🤖 Model: {results.model_name}")
        print(f"⚖️ Risk Level: {results.config.risk_level.value}")
        print(f"🎯 Overall Score: {results.overall_score:.3f}")
        print(f"✅ Risk Compliance: {'PASS' if results.risk_compliance else 'FAIL'}")
        print(f"⏱️ Execution Time: {execution_time:.2f}s")
        
        # Category summaries
        if results.safety_metrics:
            safety_count = len(results.safety_metrics)
            safety_avg = sum(results.safety_metrics.values()) / safety_count if safety_count > 0 else 0
            print(f"🛡️ Safety: {safety_count} metrics, avg: {safety_avg:.3f}")
        
        if results.security_metrics:
            security_count = len(results.security_metrics)
            security_avg = sum(results.security_metrics.values()) / security_count if security_count > 0 else 0
            print(f"🔒 Security: {security_count} metrics, avg: {security_avg:.3f}")
        
        if results.reliability_metrics:
            reliability_count = len(results.reliability_metrics)
            reliability_avg = sum(results.reliability_metrics.values()) / reliability_count if reliability_count > 0 else 0
            print(f"🔧 Reliability: {reliability_count} metrics, avg: {reliability_avg:.3f}")
        
        if results.framework_failures:
            print(f"⚠️ Framework Failures: {len(results.framework_failures)}")
            for failure in results.framework_failures:
                print(f"   - {failure}")
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / "final_comprehensive_test.json"
        results.to_json(str(results_file))
        print(f"\n💾 Results saved to: {results_file}")
        
        # Generate report
        report_file = results_dir / "final_test_report.txt"
        results.save_report(str(report_file), format="text")
        print(f"📋 Report saved to: {report_file}")
        
        print("\n🎯 FINAL ASSESSMENT:")
        if results.overall_score >= 0.8:
            print("✅ Excellent performance across all categories")
        elif results.overall_score >= 0.6:
            print("✅ Good performance with some areas for improvement")
        elif results.overall_score >= 0.4:
            print("⚠️ Moderate performance, requires attention")
        else:
            print("❌ Poor performance, significant improvements needed")
        
        if results.risk_compliance:
            print("✅ Meets all risk compliance thresholds")
        else:
            print("❌ Fails to meet some risk compliance thresholds")
        
        print(f"\n🏁 Pipeline test completed in {execution_time:.2f} seconds")
        return True
        
    except Exception as e:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        print(f"\n❌ Evaluation failed after {execution_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_final_test())
    sys.exit(0 if success else 1)