"""Quick integration test with optimized settings for faster execution."""

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


async def test_individual_frameworks():
    """Test each framework individually with minimal settings."""
    print("\n🔬 Testing Individual Frameworks")
    print("=" * 50)
    
    # Create minimal config
    config = EvaluationConfig()
    config.sample_size = 2  # Very small for speed
    config.max_concurrent_requests = 1
    
    # Use OpenAI since it's fast and reliable
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ No OpenAI API key found")
        return {}
    
    model = create_model('openai/gpt-3.5-turbo', openai_api_key=openai_key)
    orchestrator = EvaluationOrchestrator(config)
    
    test_results = {}
    
    # Test Safety Framework
    print("\n🛡️ Testing Safety Framework...")
    try:
        safety_frameworks = {'safety': ['agentharm']}
        safety_results = await orchestrator.run_framework_subset(model, safety_frameworks)
        if safety_results.safety_metrics:
            print(f"   ✅ Safety: {list(safety_results.safety_metrics.keys())}")
            test_results['safety'] = True
        else:
            print("   ⚠️ Safety returned no metrics")
            test_results['safety'] = False
    except Exception as e:
        print(f"   ❌ Safety failed: {e}")
        test_results['safety'] = False
    
    # Test Security Framework
    print("\n🔒 Testing Security Framework...")
    try:
        security_frameworks = {'security': ['houyi']}
        security_results = await orchestrator.run_framework_subset(model, security_frameworks)
        if security_results.security_metrics:
            print(f"   ✅ Security: {list(security_results.security_metrics.keys())}")
            test_results['security'] = True
        else:
            print("   ⚠️ Security returned no metrics")
            test_results['security'] = False
    except Exception as e:
        print(f"   ❌ Security failed: {e}")
        test_results['security'] = False
    
    # Test Reliability Framework (one at a time)
    print("\n🔧 Testing Reliability Frameworks...")
    
    # Test AutoEvoEval
    try:
        autoevo_frameworks = {'reliability': ['autoevoeval']}
        autoevo_results = await orchestrator.run_framework_subset(model, autoevo_frameworks)
        if autoevo_results.reliability_metrics:
            print(f"   ✅ AutoEvoEval: {list(autoevo_results.reliability_metrics.keys())}")
            test_results['autoevoeval'] = True
        else:
            print("   ⚠️ AutoEvoEval returned no metrics")
            test_results['autoevoeval'] = False
    except Exception as e:
        print(f"   ❌ AutoEvoEval failed: {e}")
        test_results['autoevoeval'] = False
    
    # Test PromptRobust
    try:
        prompt_frameworks = {'reliability': ['promptrobust']}
        prompt_results = await orchestrator.run_framework_subset(model, prompt_frameworks)
        if prompt_results.reliability_metrics:
            print(f"   ✅ PromptRobust: {list(prompt_results.reliability_metrics.keys())}")
            test_results['promptrobust'] = True
        else:
            print("   ⚠️ PromptRobust returned no metrics")
            test_results['promptrobust'] = False
    except Exception as e:
        print(f"   ❌ PromptRobust failed: {e}")
        test_results['promptrobust'] = False
    
    # Skip SelfPrompt for now since it was timing out
    print("   ⏭️ Skipping SelfPrompt (timeout issues)")
    test_results['selfprompt'] = False
    
    return test_results


async def test_configuration_validation():
    """Test all configuration files are valid."""
    print("\n⚙️ Testing Configuration Files")
    print("=" * 40)
    
    config_files = [
        "configs/default.yaml",
        "configs/high_risk.yaml", 
        "configs/medium_risk.yaml",
        "configs/low_risk.yaml"
    ]
    
    config_results = {}
    
    for config_file in config_files:
        config_path = Path(__file__).parent / config_file
        if config_path.exists():
            try:
                config = EvaluationConfig.from_yaml(str(config_path))
                errors = config.validate()
                
                if not errors:
                    print(f"   ✅ {config_file}: {config.risk_level.value} risk")
                    config_results[config_file] = True
                else:
                    print(f"   ❌ {config_file}: Validation errors: {errors}")
                    config_results[config_file] = False
            except Exception as e:
                print(f"   ❌ {config_file}: Failed to load - {e}")
                config_results[config_file] = False
        else:
            print(f"   ⚠️ {config_file}: File not found")
            config_results[config_file] = False
    
    return config_results


async def test_groq_integration():
    """Test Groq integration separately."""
    print("\n🦙 Testing Groq Integration")
    print("=" * 35)
    
    groq_key = os.getenv('GROQ_API_KEY')
    if not groq_key:
        print("   ⚠️ No Groq API key found")
        return False
    
    try:
        groq_model = create_model('groq/llama-3.3-70b-versatile', groq_api_key=groq_key)
        response = await groq_model.generate("Say 'Groq test successful'", max_tokens=10)
        print(f"   ✅ Groq connectivity: {response.strip()}")
        
        # Test with minimal safety framework
        config = EvaluationConfig()
        config.sample_size = 1
        config.max_concurrent_requests = 1
        
        orchestrator = EvaluationOrchestrator(config)
        safety_frameworks = {'safety': ['agentharm']}
        safety_results = await orchestrator.run_framework_subset(groq_model, safety_frameworks)
        
        if safety_results.safety_metrics:
            print(f"   ✅ Groq safety evaluation: {len(safety_results.safety_metrics)} metrics")
            return True
        else:
            print("   ⚠️ Groq safety evaluation returned no metrics")
            return False
    except Exception as e:
        print(f"   ❌ Groq failed: {e}")
        return False


async def main():
    """Run optimized integration tests."""
    print("🚀 LLM Evaluation Pipeline - Quick Integration Tests")
    print("=" * 65)
    
    # Check if we're in the right directory
    if not Path("src/llm_eval").exists():
        print("❌ Please run this script from the llm_eval_pipeline directory")
        return
    
    test_start_time = datetime.now()
    all_results = {}
    
    # Test 1: Configuration Validation
    config_results = await test_configuration_validation()
    all_results['config'] = all(config_results.values())
    
    # Test 2: Individual Framework Testing
    framework_results = await test_individual_frameworks()
    all_results['frameworks'] = framework_results
    
    # Test 3: Groq Integration
    groq_result = await test_groq_integration()
    all_results['groq'] = groq_result
    
    # Calculate summary
    test_end_time = datetime.now()
    execution_time = (test_end_time - test_start_time).total_seconds()
    
    # Generate summary
    print("\n" + "=" * 65)
    print("📋 QUICK INTEGRATION TEST SUMMARY")
    print("=" * 65)
    
    print(f"📊 Configuration Tests:")
    for config_file, result in config_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {config_file}: {status}")
    
    print(f"\n🔬 Framework Tests:")
    for framework, result in framework_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {framework}: {status}")
    
    print(f"\n🦙 Groq Integration: {'✅ PASS' if groq_result else '❌ FAIL'}")
    
    # Overall summary
    total_tests = len(config_results) + len(framework_results) + 1
    passed_tests = sum(config_results.values()) + sum(framework_results.values()) + (1 if groq_result else 0)
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    print(f"Execution Time: {execution_time:.2f}s")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    full_results = {
        "timestamp": test_end_time.isoformat(),
        "execution_time": execution_time,
        "config_tests": config_results,
        "framework_tests": framework_results,
        "groq_test": groq_result,
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests
        }
    }
    
    results_file = results_dir / f"quick_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    if passed_tests >= total_tests * 0.8:  # 80% pass rate
        print("🎉 Integration tests mostly successful!")
    else:
        print("⚠️ Several integration tests failed - check individual results")


if __name__ == "__main__":
    asyncio.run(main())