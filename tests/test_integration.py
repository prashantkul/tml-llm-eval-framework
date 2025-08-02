"""Integration tests with real API calls to verify functionality."""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
    print("   Trying to use environment variables directly...")

from llm_eval import EvaluationOrchestrator, EvaluationConfig, create_model


async def test_model_connectivity():
    """Test basic model connectivity."""
    print("\n🧪 Testing Model Connectivity")
    print("=" * 40)
    
    test_results = {}
    
    # Test Groq
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        print("🔍 Testing Groq API...")
        try:
            model = create_model('groq/llama-3.3-70b-versatile', groq_api_key=groq_key)
            response = await model.generate("Hello, this is a test. Respond with 'Test successful'.", max_tokens=10)
            print(f"   ✅ Groq: {response.strip()}")
            test_results['groq'] = True
        except Exception as e:
            print(f"   ❌ Groq failed: {e}")
            test_results['groq'] = False
    else:
        print("   ⚠️ GROQ_API_KEY not found")
        test_results['groq'] = False
    
    # Test OpenAI
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("🔍 Testing OpenAI API...")
        try:
            model = create_model('openai/gpt-3.5-turbo', openai_api_key=openai_key)
            response = await model.generate("Hello, this is a test. Respond with 'Test successful'.", max_tokens=10)
            print(f"   ✅ OpenAI: {response.strip()}")
            test_results['openai'] = True
        except Exception as e:
            print(f"   ❌ OpenAI failed: {e}")
            test_results['openai'] = False
    else:
        print("   ⚠️ OPENAI_API_KEY not found")
        test_results['openai'] = False
    
    return test_results


async def test_framework_integration():
    """Test individual framework integration."""
    print("\n🔬 Testing Framework Integration")
    print("=" * 40)
    
    # Use a simple configuration for faster testing
    config = EvaluationConfig()
    config.sample_size = 5  # Very small for quick testing
    config.max_concurrent_requests = 2
    
    # Get API key for testing
    api_key = os.getenv('GROQ_API_KEY') or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No API keys available for testing")
        return False
    
    # Create model - prefer OpenAI since it's working
    if os.getenv('OPENAI_API_KEY'):
        model = create_model('openai/gpt-3.5-turbo', openai_api_key=os.getenv('OPENAI_API_KEY'))
        print("🤖 Using OpenAI model for framework testing")
    elif os.getenv('GROQ_API_KEY'):
        model = create_model('groq/llama-3.3-70b-versatile', groq_api_key=os.getenv('GROQ_API_KEY'))
        print("🤖 Using Groq model for framework testing")
    else:
        print("❌ No working API keys available for testing")
        return False
    
    # Test individual frameworks
    orchestrator = EvaluationOrchestrator(config)
    framework_results = {}
    
    # Test Safety Framework
    print("\n🛡️ Testing Safety Framework...")
    try:
        safety_frameworks = {'safety': ['agentharm']}
        safety_results = await orchestrator.run_framework_subset(model, safety_frameworks)
        if safety_results.safety_metrics:
            print(f"   ✅ Safety framework: {list(safety_results.safety_metrics.keys())}")
            framework_results['safety'] = True
        else:
            print("   ⚠️ Safety framework returned no metrics")
            framework_results['safety'] = False
    except Exception as e:
        print(f"   ❌ Safety framework failed: {e}")
        framework_results['safety'] = False
    
    # Test Security Framework  
    print("\n🔒 Testing Security Framework...")
    try:
        security_frameworks = {'security': ['houyi']}
        security_results = await orchestrator.run_framework_subset(model, security_frameworks)
        if security_results.security_metrics:
            print(f"   ✅ Security framework: {list(security_results.security_metrics.keys())}")
            framework_results['security'] = True
        else:
            print("   ⚠️ Security framework returned no metrics")
            framework_results['security'] = False
    except Exception as e:
        print(f"   ❌ Security framework failed: {e}")
        framework_results['security'] = False
    
    # Test Reliability Framework
    print("\n🔧 Testing Reliability Framework...")
    try:
        reliability_frameworks = {'reliability': ['selfprompt']}
        reliability_results = await orchestrator.run_framework_subset(model, reliability_frameworks)
        if reliability_results.reliability_metrics:
            print(f"   ✅ Reliability framework: {list(reliability_results.reliability_metrics.keys())}")
            framework_results['reliability'] = True
        else:
            print("   ⚠️ Reliability framework returned no metrics")
            framework_results['reliability'] = False
    except Exception as e:
        print(f"   ❌ Reliability framework failed: {e}")
        framework_results['reliability'] = False
    
    return framework_results


async def test_full_evaluation():
    """Test a complete but small evaluation."""
    print("\n🏃 Testing Full Evaluation Pipeline")
    print("=" * 40)
    
    # Load configuration with very small sample size
    config_path = Path(__file__).parent / "configs" / "low_risk.yaml"
    config = EvaluationConfig.from_yaml(str(config_path))
    config.sample_size = 3  # Very small for quick testing
    config.max_concurrent_requests = 1
    
    print(f"📝 Using {config.risk_level.value} risk configuration")
    print(f"🔢 Sample size: {config.sample_size}")
    
    # Use available API key - prefer OpenAI since it's working
    if os.getenv('OPENAI_API_KEY'):
        model = create_model('openai/gpt-3.5-turbo', openai_api_key=os.getenv('OPENAI_API_KEY'))
        print("🤖 Using OpenAI model for full evaluation")
    elif os.getenv('GROQ_API_KEY'):
        model = create_model('groq/llama-3.3-70b-versatile', groq_api_key=os.getenv('GROQ_API_KEY'))
        print("🤖 Using Groq model for full evaluation")
    else:
        print("❌ No API keys available for full evaluation")
        return False
    
    try:
        # Run full evaluation
        orchestrator = EvaluationOrchestrator(config)
        print("\n⏳ Running evaluation... (this may take a few minutes)")
        
        results = await orchestrator.run_evaluation(model)
        
        # Print results
        print("\n📊 EVALUATION RESULTS:")
        print(f"   Model: {results.model_name}")
        print(f"   Overall Score: {results.overall_score:.3f}")
        print(f"   Risk Compliance: {'✅ PASS' if results.risk_compliance else '❌ FAIL'}")
        print(f"   Execution Time: {results.total_execution_time:.2f}s")
        
        if results.safety_metrics:
            print(f"   Safety Metrics: {len(results.safety_metrics)} metrics")
        if results.security_metrics:
            print(f"   Security Metrics: {len(results.security_metrics)} metrics")
        if results.reliability_metrics:
            print(f"   Reliability Metrics: {len(results.reliability_metrics)} metrics")
        
        if results.framework_failures:
            print(f"   ⚠️ Framework Failures: {len(results.framework_failures)}")
        
        # Save test results
        results.to_json("integration_test_results.json")
        print("\n💾 Results saved to: integration_test_results.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Full evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_configuration_loading():
    """Test configuration loading."""
    print("\n⚙️ Testing Configuration Loading")
    print("=" * 40)
    
    try:
        # Test each configuration file
        config_files = [
            "configs/default.yaml",
            "configs/high_risk.yaml", 
            "configs/medium_risk.yaml",
            "configs/low_risk.yaml"
        ]
        
        for config_file in config_files:
            config_path = Path(__file__).parent / config_file
            if config_path.exists():
                config = EvaluationConfig.from_yaml(str(config_path))
                errors = config.validate()
                
                if not errors:
                    print(f"   ✅ {config_file}: {config.risk_level.value} risk")
                else:
                    print(f"   ❌ {config_file}: Validation errors: {errors}")
                    return False
            else:
                print(f"   ⚠️ {config_file}: File not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


async def main():
    """Run all integration tests."""
    print("🚀 LLM Evaluation Pipeline - Integration Tests")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("src/llm_eval").exists():
        print("❌ Please run this script from the llm_eval_pipeline directory")
        return
    
    test_results = {}
    
    # Test 1: Configuration Loading
    test_results['config'] = await test_configuration_loading()
    
    # Test 2: Model Connectivity
    connectivity_results = await test_model_connectivity()
    test_results['connectivity'] = any(connectivity_results.values())
    
    # Test 3: Framework Integration (only if we have connectivity)
    if test_results['connectivity']:
        framework_results = await test_framework_integration()
        test_results['frameworks'] = any(framework_results.values())
    else:
        print("\n⚠️ Skipping framework tests - no API connectivity")
        test_results['frameworks'] = False
    
    # Test 4: Full Evaluation (only if frameworks work)
    if test_results['frameworks']:
        test_results['full_evaluation'] = await test_full_evaluation()
    else:
        print("\n⚠️ Skipping full evaluation - framework integration failed")
        test_results['full_evaluation'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All integration tests passed! The pipeline is working correctly.")
    elif passed_tests > 0:
        print("⚠️ Some tests passed. Check the failures above for issues.")
    else:
        print("❌ All tests failed. Please check your configuration and API keys.")
    
    print("\n💡 Tips:")
    print("   - Make sure your .env file contains valid API keys")
    print("   - Check your internet connection")
    print("   - Verify API key permissions and rate limits")
    print("   - Run 'pip install -e .' to install the package in development mode")


if __name__ == "__main__":
    asyncio.run(main())