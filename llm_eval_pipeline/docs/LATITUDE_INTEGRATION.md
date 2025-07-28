# Latitude.so Integration

This document describes the integration between our LLM Evaluation Framework and [Latitude.so](https://latitude.so), an open-source prompt engineering platform.

## Overview

The Latitude integration provides bidirectional sync between our comprehensive evaluation framework and Latitude's prompt management platform, enabling:

- **Push Evaluation Results**: Send safety, security, and reliability metrics to Latitude
- **Pull Prompt Configurations**: Evaluate Latitude-managed prompts with our framework
- **Dataset Synchronization**: Sync evaluation datasets between platforms
- **Automated Workflows**: Set up continuous evaluation and monitoring

## Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  LLM Eval Framework │ ←→ │  Latitude Bridge    │ ←→ │    Latitude.so      │
│                     │    │                     │    │                     │
│ • AgentHarm Safety  │    │ • Format Conversion │    │ • Prompt Manager    │
│ • SafetyBench QA    │    │ • Data Sync         │    │ • Evaluation System │
│ • PromptRobust      │    │ • Result Push       │    │ • Dataset Storage   │
│ • Security Tests    │    │ • Dataset Pull      │    │ • API Gateway       │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## Quick Start

### 1. Installation

```bash
# The integration is included in the main package
pip install -e .

# Additional dependency for HTTP client
pip install aiohttp
```

### 2. Configuration

```python
from llm_eval.integrations.latitude import LatitudeIntegration, LatitudeConfig

# Basic configuration
config = LatitudeConfig(
    api_key="your-latitude-api-key",
    project_id=123,  # Optional
    timeout=30
)

# Create integration instance
integration = LatitudeIntegration(config)
```

### 3. Environment Setup

```bash
# Required
export LATITUDE_API_KEY="lat_your_api_key_here"

# Optional
export LATITUDE_PROJECT_ID="123"
export LATITUDE_VERSION_UUID="version-uuid"

# For running evaluations
export OPENAI_API_KEY="sk-your_openai_key_here"
```

## Core Features

### Push Evaluation Results

Send comprehensive evaluation results from our framework to Latitude:

```python
from llm_eval.core.results import EvaluationResults
from llm_eval.frameworks.safety.agentharm_hf import AgentHarmHFEvaluator

# Run evaluation
evaluator = AgentHarmHFEvaluator(sample_size=10)
safety_result = await evaluator.evaluate(model)

# Create results object
results = EvaluationResults()
results.safety_results['agentharm_hf'] = safety_result

# Push to Latitude
push_result = await integration.push_framework_results(
    results=results,
    prompt_path="safety/harmful-content-detection",
    conversation_uuid="eval_001"
)
```

### Pull and Evaluate Latitude Prompts

Evaluate prompts managed in Latitude with our comprehensive framework:

```python
# Evaluate specific prompts
evaluation_result = await integration.pull_and_evaluate_prompts(
    prompt_paths=[
        "chatbot/customer-service",
        "content/blog-writer"
    ],
    model=model,
    evaluation_frameworks=['safety', 'reliability']
)
```

### Dataset Synchronization

Sync datasets and evaluation results between platforms:

```python
# Pull datasets and evaluations from Latitude
sync_result = await integration.sync_datasets()

print(f"Found {len(sync_result['datasets'])} datasets")
print(f"Found {len(sync_result['evaluations'])} evaluations")
```

### Health Monitoring

Check connectivity and system status:

```python
# Health check
health_status = await integration.health_check()

if health_status['status'] == 'healthy':
    print("✅ Latitude integration is working")
else:
    print(f"❌ Integration issue: {health_status['error']}")
```

## Configuration Options

### LatitudeConfig

```python
@dataclass
class LatitudeConfig:
    api_key: str                    # Required: Latitude API key
    project_id: Optional[int]       # Optional: Project ID for scoping
    version_uuid: Optional[str]     # Optional: Specific version UUID
    base_url: str                   # API base URL (default: gateway.latitude.so)
    timeout: int                    # Request timeout in seconds (default: 30)
```

### YAML Configuration

Use the provided configuration template:

```yaml
# configs/latitude_integration_example.yaml
latitude:
  api_key: "${LATITUDE_API_KEY}"
  project_id: null
  base_url: "https://gateway.latitude.so"
  timeout: 30
  
  integration:
    auto_push_results: true
    auto_sync_datasets: false
    batch_size: 50
    
    push_metrics:
      safety: true
      security: true
      reliability: true
```

## Evaluation Result Mapping

Our framework results are automatically mapped to Latitude evaluations:

| Framework Category | Our Metrics | Latitude Evaluation UUID |
|-------------------|-------------|-------------------------|
| **Safety** | `harm_score` | `safety_agentharm_harm_score` |
| | `refusal_rate` | `safety_agentharm_refusal_rate` |
| | `safety_compliance_rate` | `safety_safetybench_compliance` |
| **Security** | `attack_success_rate` | `security_houyi_attack_success` |
| | `vulnerability_score` | `security_cia_vulnerability` |
| **Reliability** | `robustness_score` | `reliability_promptrobust_robustness` |
| | `consistency_rate` | `reliability_promptrobust_consistency` |

## Error Handling

The integration includes comprehensive error handling:

```python
from llm_eval.integrations.latitude import LatitudeAPIError

try:
    result = await integration.push_framework_results(results, prompt_path)
except LatitudeAPIError as e:
    print(f"API Error: {e}")
    print(f"Status Code: {e.status_code}")
    print(f"Response: {e.response_data}")
```

## Common Workflows

### 1. Development Workflow

```python
# 1. Develop prompts in Latitude
# 2. Pull and evaluate with our framework
evaluation_result = await integration.pull_and_evaluate_prompts([prompt_path], model)

# 3. Review comprehensive safety/security/reliability metrics
# 4. Iterate on prompt design in Latitude
```

### 2. Production Monitoring

```python
# Continuous monitoring setup
async def monitor_production_prompts():
    while True:
        # Pull latest prompt performance data
        sync_result = await integration.sync_datasets()
        
        # Run evaluations on critical prompts
        critical_prompts = ["prod/user-facing", "prod/content-generation"]
        eval_result = await integration.pull_and_evaluate_prompts(critical_prompts, model)
        
        # Push updated metrics to Latitude dashboard
        await integration.push_framework_results(results, "prod/monitoring")
        
        # Sleep until next monitoring cycle
        await asyncio.sleep(3600)  # Every hour
```

### 3. Batch Evaluation

```python
# Evaluate multiple prompt variants
prompt_variants = [
    "experiments/variant-a",
    "experiments/variant-b", 
    "experiments/variant-c"
]

# Run comprehensive evaluation on all variants
for prompt_path in prompt_variants:
    eval_result = await integration.pull_and_evaluate_prompts([prompt_path], model)
    
    # Results automatically appear in Latitude for comparison
```

## API Reference

### LatitudeIntegration

Main integration class providing high-level workflow methods.

#### Methods

- `push_framework_results(results, prompt_path, conversation_uuid)` - Push evaluation results to Latitude
- `pull_and_evaluate_prompts(prompt_paths, model, frameworks)` - Pull prompts and evaluate with our framework
- `sync_datasets()` - Synchronize datasets between platforms
- `health_check()` - Check connectivity and authentication

### LatitudeClient

Low-level API client for direct Latitude API interaction.

#### Methods

- `run_prompt(prompt)` - Execute a prompt in Latitude
- `push_evaluation(evaluation)` - Push single evaluation result
- `create_log(log_data)` - Create log entry
- `get_datasets()` - Retrieve available datasets
- `get_evaluations(dataset_id)` - Retrieve evaluation results

## Testing

Run the integration test suite:

```bash
# Basic functionality test (no API key required)
python test_latitude_integration.py

# Full integration test (requires LATITUDE_API_KEY)
LATITUDE_API_KEY=your_key python test_latitude_integration.py
```

Run specific examples:

```bash
# Run usage examples
python examples/latitude_integration_example.py
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```
   Error: Invalid API key
   Solution: Check LATITUDE_API_KEY environment variable
   ```

2. **Timeout Errors**
   ```
   Error: Request timeout
   Solution: Increase timeout in LatitudeConfig or check network connectivity
   ```

3. **Project Access Errors**
   ```
   Error: Project not found
   Solution: Verify project_id or remove for default project access
   ```

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.getLogger('llm_eval.integrations.latitude').setLevel(logging.DEBUG)
```

## Limitations

- Requires active Latitude.so account and API access
- HTTP client dependency (aiohttp)
- Async-only interface (no synchronous methods)
- Evaluation result conversion may lose some metadata detail

## Future Enhancements

- [ ] Real-time webhook integration
- [ ] Custom evaluation metric mappings
- [ ] Batch evaluation optimizations
- [ ] Synchronous API interface
- [ ] Enhanced error recovery and retry logic
- [ ] Streaming evaluation results
- [ ] Integration with Latitude's A/B testing features

## Support

For integration issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Review [Latitude.so documentation](https://docs.latitude.so)
3. Run the test suite for diagnostic information
4. Check logs with debug mode enabled

## Contributing

To contribute to the Latitude integration:

1. Follow the existing async patterns
2. Add comprehensive error handling
3. Include test coverage for new features
4. Update this documentation
5. Test with real Latitude.so account when possible