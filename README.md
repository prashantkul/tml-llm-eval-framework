# LLM Comprehensive Evaluation Pipeline

An LLM evaluation pipeline that integrates multiple safety, security, and reliability frameworks. This modular, extensible system can evaluate any LLM across comprehensive metrics with risk-adjusted thresholds.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, Groq, and local models
- **Comprehensive Frameworks**: Safety (AgentHarm, Agent-SafetyBench), Security (HouYi, CIA Attacks), Reliability (AutoEvoEval, PromptRobust, SelfPrompt)
- **Risk-Based Configuration**: High, medium, and low risk profiles with appropriate thresholds
- **Async Evaluation**: Concurrent execution with rate limiting and retry logic
- **Rich Reporting**: JSON, HTML, Markdown reports with visualizations
- **CLI Interface**: Easy command-line usage with comprehensive options

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm_eval_pipeline

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Basic Usage

```bash
# Evaluate Llama-4-Scout via Groq
llm-eval evaluate \
    --model groq/llama-4-scout-17b-16e-instruct \
    --config configs/medium_risk.yaml \
    --output results.json \
    --groq-api-key your_groq_api_key

# Generate HTML report
llm-eval report --results results.json --format html --output report.html

# Test model connection
llm-eval test-model --model groq/llama-4-scout-17b-16e-instruct --groq-api-key your_key
```

## Configuration

### Risk Levels

The pipeline supports three risk levels with pre-configured thresholds:

- **High Risk** (`configs/high_risk.yaml`): Medical, Legal, Financial applications
- **Medium Risk** (`configs/medium_risk.yaml`): Education, Customer Service
- **Low Risk** (`configs/low_risk.yaml`): Entertainment, Creative applications

### Supported Models

```bash
# OpenAI models
--model openai/gpt-4
--model openai/gpt-3.5-turbo

# Anthropic models  
--model anthropic/claude-3-opus-20240229
--model anthropic/claude-3-sonnet-20240229

# Groq models (open source)
--model groq/llama-4-scout-17b-16e-instruct
--model groq/llama-3.2-90b-text-preview
--model groq/mixtral-8x7b-32768

# Local models
--model local/microsoft/DialoGPT-medium
```

## Research Foundation

This evaluation pipeline implements methodologies from multiple research papers and frameworks:

### Safety Research
- **AgentHarm**: [arxiv:2410.09024](https://arxiv.org/abs/2410.09024) - Evaluating Cybersecurity Skills and Jailbreaking in LLM Agents
- **Agent-SafetyBench**: Comprehensive safety assessment framework for LLM agents
- **SafetyBench**: Multi-dimensional safety evaluation framework

### Security Research  
- **HouYi**: Attack framework testing 8 different security vulnerabilities
- **CIA Attacks**: Compositional Instruction Attacks using T-CIA and W-CIA transformations

### Reliability Research
- **AutoEvoEval**: 22 atomic evolution operations testing robustness
- **PromptRobust**: [arxiv:2306.04528](https://arxiv.org/abs/2306.04528) - 4-level attack taxonomy (character, word, sentence, semantic)
- **SelfPrompt**: Self-consistency evaluation across multiple dimensions

### Integration Platform
- **Latitude.so**: Bidirectional integration for prompt management and evaluation workflows
- **HuggingFace Datasets**: Direct integration with research datasets

## Framework Overview

### Safety Frameworks

- **AgentHarm**: Evaluates response to harmful prompts across 10 categories
- **Agent-SafetyBench**: Comprehensive safety assessment with multiple failure modes

### Security Frameworks

- **HouYi**: Attack framework testing 8 different security vulnerabilities
- **CIA Attacks**: Compositional Instruction Attacks using T-CIA and W-CIA transformations

### Reliability Frameworks

- **AutoEvoEval**: 22 atomic evolution operations testing robustness
- **PromptRobust**: 4-level attack taxonomy (character, word, sentence, semantic)
- **SelfPrompt**: Self-consistency evaluation across multiple dimensions

## API Usage

```python
from llm_eval import EvaluationOrchestrator, EvaluationConfig, GroqModel

# Configure evaluation
config = EvaluationConfig.from_yaml('configs/medium_risk.yaml')

# Create model instance
model = GroqModel('llama-4-scout-17b-16e-instruct', api_key='your_key')

# Run evaluation
orchestrator = EvaluationOrchestrator(config)
results = await orchestrator.run_evaluation(model)

# Check results
print(f"Overall Score: {results.overall_score}")
print(f"Risk Compliance: {results.risk_compliance}")
```

## Advanced Usage

### Custom Framework Selection

```bash
llm-eval evaluate \
    --model groq/llama-4-scout-17b-16e-instruct \
    --frameworks agentharm,houyi,selfprompt \
    --output subset_results.json
```

### Model Comparison

```bash
# Evaluate multiple models
llm-eval evaluate --model groq/llama-4-scout-17b-16e-instruct --output scout_results.json
llm-eval evaluate --model groq/llama-3.2-90b-text-preview --output llama_results.json

# Compare results
llm-eval compare --results scout_results.json --results llama_results.json --output comparison.txt
```

### Custom Configuration

```bash
# Create custom config
llm-eval create-config \
    --risk-level high \
    --output my_config.yaml \
    --enable-safety \
    --disable-security

# Use custom config
llm-eval evaluate --model your_model --config my_config.yaml
```

## Environment Variables

Set API keys as environment variables:

```bash
export GROQ_API_KEY="your_groq_api_key"
export OPENAI_API_KEY="your_openai_api_key" 
export ANTHROPIC_API_KEY="your_anthropic_api_key"
```

## Results Format

Evaluation results include:

- **Overall Metrics**: Aggregated safety, security, reliability scores
- **Framework Results**: Detailed results from each evaluation framework
- **Risk Compliance**: Pass/fail status against risk-adjusted thresholds
- **Execution Metadata**: Timing, failures, and framework status

Example result structure:
```json
{
    "model_name": "llama-4-scout-17b-16e-instruct",
    "timestamp": "2024-01-15T10:30:00",
    "overall_score": 0.85,
    "risk_compliance": true,
    "safety_metrics": {...},
    "security_metrics": {...},
    "reliability_metrics": {...}
}
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=llm_eval

# Format code
black src/
flake8 src/

# Type checking
mypy src/
```

### Adding New Frameworks

1. Create evaluator class in appropriate framework directory
2. Implement `evaluate()` method returning `Dict[str, float]`
3. Add to orchestrator framework initialization
4. Update configuration schema

### Running Tests

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/test_safety/
pytest tests/test_security/
pytest tests/test_reliability/

# Run with verbose output
pytest -v

# Run async tests
pytest tests/test_async/
```

## Performance Considerations

- **Concurrency**: Adjust `max_concurrent_requests` based on API rate limits
- **Sample Size**: Balance between statistical significance and evaluation time
- **Framework Selection**: Use subset evaluation for faster results
- **Risk Level**: Lower risk configurations run faster with relaxed testing

## Monitoring and Logging

The pipeline includes comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Enable debug logging
logging.getLogger('llm_eval').setLevel(logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this evaluation pipeline in your research, please cite:

```bibtex
@software{llm_eval_pipeline,
  title={LLM Comprehensive Evaluation Pipeline},
  author={LLM Evaluation Team},
  year={2024},
  url={https://github.com/example/llm-eval-pipeline}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/example/llm-eval-pipeline/issues)
- **Documentation**: [ReadTheDocs](https://llm-eval-pipeline.readthedocs.io/)
- **Discussions**: [GitHub Discussions](https://github.com/example/llm-eval-pipeline/discussions)