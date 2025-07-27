"""Quick integration test focusing on working functionality."""

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

from llm_eval import EvaluationOrchestrator, EvaluationConfig, create_model


async def main():
    """Run quick tests and save results."""
    print("🚀 Quick Integration Test - LLM Evaluation Pipeline")
    print("=" * 60)
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    test_log = []
    
    # Test 1: Configuration Loading
    print("\n⚙️ Testing Configuration Loading...")
    try:
        config = EvaluationConfig.from_yaml("configs/low_risk.yaml")
        config.sample_size = 2  # Very small for quick testing
        config.max_concurrent_requests = 1
        
        test_log.append({
            "test": "config_loading",
            "status": "success",
            "details": f"Loaded {config.risk_level.value} risk config"
        })
        print("✅ Configuration loaded successfully")
    except Exception as e:
        test_log.append({
            "test": "config_loading", 
            "status": "failed",
            "error": str(e)
        })
        print(f"❌ Configuration failed: {e}")
        return
    
    # Test 2: Model Connectivity (OpenAI only)
    print("\n🧪 Testing OpenAI Model Connectivity...")
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        test_log.append({
            "test": "openai_connectivity",
            "status": "skipped", 
            "reason": "No API key"
        })
        print("⚠️ No OpenAI API key found")
        return
    
    try:
        model = create_model('openai/gpt-3.5-turbo', openai_api_key=openai_key)
        response = await model.generate("Say 'Hello Test'", max_tokens=5)
        test_log.append({
            "test": "openai_connectivity",
            "status": "success",
            "response": response
        })
        print(f"✅ OpenAI connectivity: {response}")
    except Exception as e:
        test_log.append({
            "test": "openai_connectivity",
            "status": "failed", 
            "error": str(e)
        })
        print(f"❌ OpenAI connectivity failed: {e}")
        return
    
    # Test 3: Single Framework Test (Safety only)
    print("\n🛡️ Testing Single Safety Framework...")
    try:
        orchestrator = EvaluationOrchestrator(config)
        safety_frameworks = {'safety': ['agentharm']}
        
        print("   Running safety evaluation...")
        results = await orchestrator.run_framework_subset(model, safety_frameworks)
        
        test_log.append({
            "test": "safety_framework",
            "status": "success",
            "metrics": dict(results.safety_metrics) if results.safety_metrics else {},
            "execution_time": results.total_execution_time
        })
        print(f"✅ Safety framework completed in {results.total_execution_time:.2f}s")
        print(f"   Metrics: {list(results.safety_metrics.keys()) if results.safety_metrics else 'None'}")
        
    except Exception as e:
        test_log.append({
            "test": "safety_framework",
            "status": "failed",
            "error": str(e)
        })
        print(f"❌ Safety framework failed: {e}")
    
    # Test 4: Groq Test (if key available)
    print("\n🦙 Testing Groq Connectivity...")
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        try:
            groq_model = create_model('groq/llama-3.3-70b-versatile', groq_api_key=groq_key)
            groq_response = await groq_model.generate("Say 'Hello Groq'", max_tokens=5)
            test_log.append({
                "test": "groq_connectivity",
                "status": "success",
                "response": groq_response
            })
            print(f"✅ Groq connectivity: {groq_response}")
        except Exception as e:
            test_log.append({
                "test": "groq_connectivity",
                "status": "failed",
                "error": str(e)
            })
            print(f"❌ Groq connectivity failed: {e}")
    else:
        test_log.append({
            "test": "groq_connectivity",
            "status": "skipped",
            "reason": "No API key"
        })
        print("⚠️ No Groq API key found")
    
    # Save test results
    timestamp = datetime.now().isoformat()
    test_results = {
        "timestamp": timestamp,
        "pipeline_version": "0.1.0",
        "test_summary": {
            "total_tests": len(test_log),
            "passed": len([t for t in test_log if t["status"] == "success"]),
            "failed": len([t for t in test_log if t["status"] == "failed"]),
            "skipped": len([t for t in test_log if t["status"] == "skipped"])
        },
        "detailed_results": test_log
    }
    
    # Save to results folder
    results_file = results_dir / f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Create summary report
    summary_file = results_dir / "test_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("LLM Evaluation Pipeline - Integration Test Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Date: {timestamp}\n")
        f.write(f"Pipeline Version: 0.1.0\n\n")
        
        f.write("Test Results:\n")
        f.write("-" * 20 + "\n")
        for test in test_log:
            status_emoji = {"success": "✅", "failed": "❌", "skipped": "⚠️"}[test["status"]]
            f.write(f"{status_emoji} {test['test']}: {test['status'].upper()}\n")
            if test["status"] == "failed":
                f.write(f"   Error: {test.get('error', 'Unknown')}\n")
            elif test["status"] == "success" and "execution_time" in test:
                f.write(f"   Time: {test['execution_time']:.2f}s\n")
        
        f.write(f"\nSummary: {test_results['test_summary']['passed']}/{test_results['test_summary']['total_tests']} tests passed\n")
    
    print(f"\n📊 Test Results Summary:")
    print(f"   Total Tests: {test_results['test_summary']['total_tests']}")
    print(f"   Passed: {test_results['test_summary']['passed']}")
    print(f"   Failed: {test_results['test_summary']['failed']}")
    print(f"   Skipped: {test_results['test_summary']['skipped']}")
    
    print(f"\n💾 Results saved to:")
    print(f"   Detailed: {results_file}")
    print(f"   Summary: {summary_file}")
    
    # Final evaluation recommendations
    print(f"\n🎯 Recommendations:")
    if test_results['test_summary']['passed'] >= 2:
        print("   ✅ Core pipeline functionality is working")
        print("   ✅ Ready for basic evaluations with OpenAI")
        if any(t["test"] == "groq_connectivity" and t["status"] == "failed" for t in test_log):
            print("   ⚠️ Check Groq API key if you need Groq models")
    else:
        print("   ❌ Pipeline needs debugging before use")
        print("   🔧 Check API keys and dependencies")


if __name__ == "__main__":
    asyncio.run(main())