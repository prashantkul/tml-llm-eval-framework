"""
Latitude.so Integration Usage Example

This example demonstrates how to integrate our LLM evaluation framework
with Latitude's prompt engineering platform for comprehensive evaluation workflows.
"""

import asyncio
import os
from llm_eval.integrations.latitude import LatitudeIntegration, LatitudeConfig, LatitudePrompt
from llm_eval.frameworks.safety.agentharm_hf import AgentHarmHFEvaluator
from llm_eval.frameworks.reliability.promptrobust_hf import PromptRobustHFEvaluator
from llm_eval.core.model_interface import create_model
from llm_eval.core.results import EvaluationResults


async def basic_latitude_integration():
    """Basic example of Latitude integration."""
    
    # 1. Configure Latitude integration
    config = LatitudeConfig(
        api_key=os.getenv('LATITUDE_API_KEY', 'your-latitude-api-key'),
        project_id=123,  # Optional: your project ID
        timeout=30
    )
    
    # 2. Create integration instance
    integration = LatitudeIntegration(config)
    
    # 3. Test connectivity
    health_status = await integration.health_check()
    print(f"Latitude connectivity: {health_status['status']}")
    
    if health_status['status'] != 'healthy':
        print("❌ Cannot connect to Latitude")
        return
    
    # 4. Sync datasets (pull from Latitude)
    sync_result = await integration.sync_datasets()
    print(f"📊 Found {len(sync_result.get('datasets', []))} datasets in Latitude")
    
    return integration


async def comprehensive_evaluation_workflow():
    """Example of running evaluation and pushing results to Latitude."""
    
    # Setup
    config = LatitudeConfig(api_key=os.getenv('LATITUDE_API_KEY'))
    integration = LatitudeIntegration(config)
    model = create_model('openai/gpt-3.5-turbo', openai_api_key=os.getenv('OPENAI_API_KEY'))
    
    # 1. Run comprehensive evaluation with our framework
    print("🔬 Running comprehensive LLM evaluation...")
    
    results = EvaluationResults()
    
    # Safety evaluation
    safety_evaluator = AgentHarmHFEvaluator(sample_size=3)
    safety_result = await safety_evaluator.evaluate(model)
    results.safety_results['agentharm_hf'] = safety_result
    
    # Reliability evaluation  
    reliability_evaluator = PromptRobustHFEvaluator(sample_size=3)
    reliability_result = await reliability_evaluator.evaluate(model)
    results.reliability_results['promptrobust_hf'] = reliability_result
    
    print("✅ Evaluation completed")
    
    # 2. Push results to Latitude
    print("📤 Pushing results to Latitude...")
    
    push_result = await integration.push_framework_results(
        results=results,
        prompt_path="evaluation/comprehensive-safety-reliability",
        conversation_uuid="eval_comprehensive_001"
    )
    
    print(f"✅ Pushed {push_result['summary']['successful_pushes']} evaluations to Latitude")
    
    return push_result


async def prompt_evaluation_workflow():
    """Example of evaluating Latitude prompts with our framework."""
    
    config = LatitudeConfig(api_key=os.getenv('LATITUDE_API_KEY'))
    integration = LatitudeIntegration(config)
    model = create_model('openai/gpt-3.5-turbo', openai_api_key=os.getenv('OPENAI_API_KEY'))
    
    # List of Latitude prompts to evaluate
    prompt_paths = [
        "chatbot/customer-service",
        "content/blog-writer", 
        "analysis/sentiment-analyzer"
    ]
    
    print("🎯 Evaluating Latitude prompts with our framework...")
    
    # Pull prompts from Latitude and evaluate
    evaluation_result = await integration.pull_and_evaluate_prompts(
        prompt_paths=prompt_paths,
        model=model,
        evaluation_frameworks=['safety', 'reliability']
    )
    
    print(f"✅ Evaluated {evaluation_result['summary']['successful_evaluations']} prompts")
    
    # Results are automatically available in Latitude dashboard
    return evaluation_result


async def continuous_monitoring_setup():
    """Example of setting up continuous monitoring workflow."""
    
    config = LatitudeConfig(api_key=os.getenv('LATITUDE_API_KEY'))
    integration = LatitudeIntegration(config)
    
    print("🔄 Setting up continuous monitoring...")
    
    # This would typically be run on a schedule (e.g., daily)
    while True:
        try:
            # 1. Check Latitude for new prompts/datasets
            sync_result = await integration.sync_datasets()
            
            # 2. Run evaluations on new/updated prompts
            # (Implementation would depend on your specific needs)
            
            # 3. Push updated evaluation results
            # (Based on detected changes)
            
            print(f"📊 Monitoring cycle completed at {sync_result['sync_timestamp']}")
            
            # In real implementation, you'd break after one cycle
            # Here we break to avoid infinite loop in example
            break
            
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
            break


async def custom_evaluation_mapping():
    """Example of custom evaluation result mapping to Latitude."""
    
    config = LatitudeConfig(api_key=os.getenv('LATITUDE_API_KEY'))
    integration = LatitudeIntegration(config)
    
    # Custom mapping for specific business needs
    custom_mappings = {
        "safety_score": "company_safety_metric",
        "reliability_index": "uptime_reliability", 
        "security_rating": "compliance_score"
    }
    
    # You can extend the integration class for custom mappings
    # This is a simplified example
    print("🎨 Custom evaluation mapping configured")
    
    return custom_mappings


async def main():
    """Run all Latitude integration examples."""
    print("🌍 Latitude.so Integration Examples")
    print("=" * 50)
    
    try:
        # Basic integration
        print("\n1️⃣ Basic Integration Test")
        await basic_latitude_integration()
        
        # Comprehensive workflow (requires API keys)
        if os.getenv('LATITUDE_API_KEY') and os.getenv('OPENAI_API_KEY'):
            print("\n2️⃣ Comprehensive Evaluation Workflow")
            await comprehensive_evaluation_workflow()
            
            print("\n3️⃣ Prompt Evaluation Workflow")
            await prompt_evaluation_workflow()
        else:
            print("\n⚠️ Skipping advanced examples - need LATITUDE_API_KEY and OPENAI_API_KEY")
        
        # Monitoring setup
        print("\n4️⃣ Continuous Monitoring Setup")
        await continuous_monitoring_setup()
        
        # Custom mapping
        print("\n5️⃣ Custom Evaluation Mapping")
        await custom_evaluation_mapping()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")


if __name__ == "__main__":
    # Set up environment variables first:
    # export LATITUDE_API_KEY="your-latitude-api-key"
    # export OPENAI_API_KEY="your-openai-api-key"
    
    asyncio.run(main())