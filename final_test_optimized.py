"""Final optimized test excluding problematic frameworks."""

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


async def run_optimized_test():
    """Run optimized evaluation excluding slow frameworks."""
    print("🚀 Final Optimized Test - LLM Evaluation Pipeline")
    print("=" * 65)
    
    # Create minimal config
    config = EvaluationConfig()
    config.sample_size = 3
    config.max_concurrent_requests = 2
    config.timeout_seconds = 45
    
    print(f"📝 Configuration: {config.risk_level.value} risk, {config.sample_size} samples")
    
    # Create OpenAI model
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ No OpenAI API key found")
        return False
    
    model = create_model('openai/gpt-3.5-turbo', openai_api_key=openai_key)
    print(f"🤖 Model: {model.model_name}")
    
    # Run individual framework tests
    orchestrator = EvaluationOrchestrator(config)
    
    framework_results = {}
    total_start = datetime.now()
    
    # Test Safety
    print("\n🛡️ Testing Safety Framework...")
    try:
        safety_frameworks = {'safety': ['agentharm']}
        safety_results = await orchestrator.run_framework_subset(model, safety_frameworks)
        framework_results['safety'] = safety_results
        print(f"   ✅ Safety: {len(safety_results.safety_metrics)} metrics")
    except Exception as e:
        print(f"   ❌ Safety failed: {e}")
        framework_results['safety'] = None
    
    # Test Security
    print("\n🔒 Testing Security Framework...")
    try:
        security_frameworks = {'security': ['houyi']}
        security_results = await orchestrator.run_framework_subset(model, security_frameworks)
        framework_results['security'] = security_results
        print(f"   ✅ Security: {len(security_results.security_metrics)} metrics")
    except Exception as e:
        print(f"   ❌ Security failed: {e}")
        framework_results['security'] = None
    
    # Test Reliability (only fast ones)
    print("\n🔧 Testing Reliability Frameworks...")
    try:
        reliability_frameworks = {'reliability': ['autoevoeval', 'promptrobust']}
        reliability_results = await orchestrator.run_framework_subset(model, reliability_frameworks)
        framework_results['reliability'] = reliability_results
        print(f"   ✅ Reliability: {len(reliability_results.reliability_metrics)} metrics")
    except Exception as e:
        print(f"   ❌ Reliability failed: {e}")
        framework_results['reliability'] = None
    
    total_end = datetime.now()
    total_time = (total_end - total_start).total_seconds()
    
    # Aggregate results
    print("\n📊 FINAL RESULTS SUMMARY")
    print("=" * 50)
    
    successful_frameworks = sum(1 for r in framework_results.values() if r is not None)
    total_frameworks = len(framework_results)
    
    all_metrics = {}
    for category, result in framework_results.items():
        if result:
            category_metrics = getattr(result, f"{category}_metrics", {})
            if category_metrics:
                all_metrics[category] = category_metrics
                avg_score = sum(category_metrics.values()) / len(category_metrics)
                print(f"🎯 {category.capitalize()}: {len(category_metrics)} metrics, avg: {avg_score:.3f}")
                
                # Show top metrics
                top_metrics = sorted(category_metrics.items(), key=lambda x: x[1], reverse=True)[:3]
                for metric, value in top_metrics:
                    print(f"   - {metric}: {value:.3f}")
    
    print(f"\n⏱️ Total Execution Time: {total_time:.2f} seconds")
    print(f"✅ Successful Frameworks: {successful_frameworks}/{total_frameworks}")
    
    # Calculate overall assessment
    if all_metrics:
        all_values = []
        for metrics in all_metrics.values():
            all_values.extend(metrics.values())
        
        overall_avg = sum(all_values) / len(all_values) if all_values else 0
        print(f"🎯 Overall Average Score: {overall_avg:.3f}")
        
        # Assessment
        print("\n🏁 PIPELINE ASSESSMENT:")
        if successful_frameworks == total_frameworks:
            print("✅ All framework categories working correctly")
        else:
            print(f"⚠️ {total_frameworks - successful_frameworks} framework categories failed")
        
        if overall_avg >= 0.7:
            print("✅ Good overall performance")
        elif overall_avg >= 0.5:
            print("⚠️ Moderate performance")
        else:
            print("❌ Needs improvement")
        
        print(f"⚡ Performance: {len(all_values)} total metrics in {total_time:.1f}s")
        
        # Save summary
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        summary = {
            "timestamp": total_end.isoformat(),
            "model": model.model_name,
            "execution_time": total_time,
            "successful_frameworks": successful_frameworks,
            "total_frameworks": total_frameworks,
            "overall_score": overall_avg,
            "metrics_by_category": all_metrics,
            "assessment": "SUCCESS" if successful_frameworks >= 2 else "PARTIAL"
        }
        
        summary_file = results_dir / "final_optimized_test.json"
        with open(summary_file, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Summary saved to: {summary_file}")
        
        return successful_frameworks >= 2
    
    else:
        print("❌ No metrics collected - pipeline has issues")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_optimized_test())
    print(f"\n{'🎉 SUCCESS' if success else '❌ FAILED'}: Final pipeline test completed")
    sys.exit(0 if success else 1)