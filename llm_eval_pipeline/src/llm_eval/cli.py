"""Command-line interface for LLM evaluation pipeline."""

import click
import asyncio
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

from .core.config import EvaluationConfig, RiskLevel
from .core.model_interface import create_model
from .core.orchestrator import EvaluationOrchestrator
from .core.results import EvaluationResults

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """LLM Comprehensive Evaluation Pipeline"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option('--model', '-m', required=True, 
              help='Model identifier (e.g., groq/llama-3.2-90b-text-preview, openai/gpt-4)')
@click.option('--config', '-c', default='configs/default.yaml', 
              help='Configuration file path')
@click.option('--output', '-o', default='results.json', 
              help='Output file for results')
@click.option('--datasets', '-d', multiple=True, 
              help='Evaluation datasets (can be specified multiple times)')
@click.option('--groq-api-key', envvar='GROQ_API_KEY', 
              help='Groq API key (or set GROQ_API_KEY env var)')
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', 
              help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--anthropic-api-key', envvar='ANTHROPIC_API_KEY', 
              help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
@click.option('--frameworks', 
              help='Comma-separated list of specific frameworks to run (e.g., agentharm,houyi)')
@click.option('--risk-level', type=click.Choice(['high', 'medium', 'low']), 
              help='Override risk level from config')
@click.option('--sample-size', type=int, 
              help='Override sample size from config')
@click.option('--timeout', type=int, 
              help='Override timeout in seconds')
def evaluate(model, config, output, datasets, groq_api_key, openai_api_key, 
             anthropic_api_key, frameworks, risk_level, sample_size, timeout):
    """Run comprehensive LLM evaluation"""
    asyncio.run(_run_evaluation(
        model, config, output, datasets, groq_api_key, openai_api_key, 
        anthropic_api_key, frameworks, risk_level, sample_size, timeout
    ))


async def _run_evaluation(model_id, config_path, output_path, datasets, 
                         groq_api_key, openai_api_key, anthropic_api_key, 
                         frameworks, risk_level, sample_size, timeout):
    """Async function to run evaluation."""
    try:
        # Load configuration
        if Path(config_path).exists():
            evaluation_config = EvaluationConfig.from_yaml(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            evaluation_config = EvaluationConfig()
            logger.info("Using default configuration")
        
        # Override config parameters if provided
        if risk_level:
            evaluation_config.risk_level = RiskLevel(risk_level)
        if sample_size:
            evaluation_config.sample_size = sample_size
        if timeout:
            evaluation_config.timeout_seconds = timeout
        
        # Validate configuration
        config_errors = evaluation_config.validate()
        if config_errors:
            for error in config_errors:
                click.echo(f"Configuration error: {error}", err=True)
            sys.exit(1)
        
        # Create model instance
        model_kwargs = {}
        if groq_api_key:
            model_kwargs['groq_api_key'] = groq_api_key
        if openai_api_key:
            model_kwargs['openai_api_key'] = openai_api_key
        if anthropic_api_key:
            model_kwargs['anthropic_api_key'] = anthropic_api_key
        
        try:
            model_instance = create_model(model_id, **model_kwargs)
            logger.info(f"Created model instance: {model_instance.get_model_info()}")
        except Exception as e:
            click.echo(f"Failed to create model instance: {e}", err=True)
            sys.exit(1)
        
        # Create orchestrator
        orchestrator = EvaluationOrchestrator(evaluation_config)
        
        # Parse specific frameworks if provided
        if frameworks:
            framework_list = [f.strip() for f in frameworks.split(',')]
            available_frameworks = orchestrator.get_available_frameworks()
            
            # Map frameworks to categories
            framework_mapping = {}
            for category, framework_names in available_frameworks.items():
                for framework_name in framework_names:
                    if framework_name in framework_list:
                        if category not in framework_mapping:
                            framework_mapping[category] = []
                        framework_mapping[category].append(framework_name)
            
            click.echo(f"Running subset evaluation with frameworks: {framework_mapping}")
            results = await orchestrator.run_framework_subset(
                model_instance, framework_mapping, list(datasets) if datasets else None
            )
        else:
            # Run full evaluation
            click.echo("Running comprehensive evaluation...")
            results = await orchestrator.run_evaluation(
                model_instance, list(datasets) if datasets else None
            )
        
        # Save results
        results.to_json(output_path)
        click.echo(f"Results saved to {output_path}")
        
        # Print summary
        _print_evaluation_summary(results)
        
    except Exception as e:
        logger.exception("Evaluation failed")
        click.echo(f"Evaluation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--results', '-r', required=True, help='Results JSON file')
@click.option('--format', '-f', type=click.Choice(['text', 'html', 'markdown']), 
              default='text', help='Report format')
@click.option('--output', '-o', help='Output file (default: print to stdout)')
def report(results, format, output):
    """Generate evaluation report from results"""
    try:
        # Load results
        with open(results, 'r') as f:
            results_data = json.load(f)
        
        # Reconstruct results object
        evaluation_results = EvaluationResults.from_json(json.dumps(results_data))
        
        # Generate report
        report_content = evaluation_results.to_report(format)
        
        if output:
            with open(output, 'w') as f:
                f.write(report_content)
            click.echo(f"Report saved to {output}")
        else:
            click.echo(report_content)
            
    except Exception as e:
        click.echo(f"Failed to generate report: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--results', '-r', multiple=True, required=True, 
              help='Results JSON files to compare (specify multiple)')
@click.option('--output', '-o', help='Output file for comparison')
def compare(results, output):
    """Compare evaluation results from multiple models"""
    try:
        if len(results) < 2:
            click.echo("At least 2 result files required for comparison", err=True)
            sys.exit(1)
        
        # Load all results
        all_results = []
        for result_file in results:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            evaluation_result = EvaluationResults.from_json(json.dumps(result_data))
            all_results.append(evaluation_result)
        
        # Generate comparison
        comparison = _generate_comparison_report(all_results)
        
        if output:
            with open(output, 'w') as f:
                f.write(comparison)
            click.echo(f"Comparison saved to {output}")
        else:
            click.echo(comparison)
            
    except Exception as e:
        click.echo(f"Failed to generate comparison: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--risk-level', type=click.Choice(['high', 'medium', 'low']), 
              default='medium', help='Risk level for configuration')
@click.option('--output', '-o', required=True, help='Output configuration file')
@click.option('--enable-safety/--disable-safety', default=True, 
              help='Enable/disable safety evaluation')
@click.option('--enable-security/--disable-security', default=True, 
              help='Enable/disable security evaluation')
@click.option('--enable-reliability/--disable-reliability', default=True, 
              help='Enable/disable reliability evaluation')
def create_config(risk_level, output, enable_safety, enable_security, enable_reliability):
    """Create a new configuration file"""
    try:
        config = EvaluationConfig(
            risk_level=RiskLevel(risk_level),
            enable_safety=enable_safety,
            enable_security=enable_security,
            enable_reliability=enable_reliability
        )
        
        config.to_yaml(output)
        click.echo(f"Configuration created: {output}")
        
    except Exception as e:
        click.echo(f"Failed to create configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_frameworks():
    """List available evaluation frameworks"""
    try:
        # Create a default config to initialize orchestrator
        config = EvaluationConfig()
        orchestrator = EvaluationOrchestrator(config)
        available_frameworks = orchestrator.get_available_frameworks()
        
        click.echo("Available evaluation frameworks:")
        click.echo("=" * 40)
        
        for category, frameworks in available_frameworks.items():
            click.echo(f"\n{category.upper()}:")
            for framework in frameworks:
                click.echo(f"  - {framework}")
        
        click.echo(f"\nTotal frameworks: {sum(len(f) for f in available_frameworks.values())}")
        
    except Exception as e:
        click.echo(f"Failed to list frameworks: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model', '-m', required=True, help='Model identifier to test')
@click.option('--groq-api-key', envvar='GROQ_API_KEY', help='Groq API key')
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', help='OpenAI API key') 
@click.option('--anthropic-api-key', envvar='ANTHROPIC_API_KEY', help='Anthropic API key')
def test_model(model, groq_api_key, openai_api_key, anthropic_api_key):
    """Test model connection and basic functionality"""
    asyncio.run(_test_model_connection(model, groq_api_key, openai_api_key, anthropic_api_key))


async def _test_model_connection(model_id, groq_api_key, openai_api_key, anthropic_api_key):
    """Test model connection."""
    try:
        # Create model instance
        model_kwargs = {}
        if groq_api_key:
            model_kwargs['groq_api_key'] = groq_api_key
        if openai_api_key:
            model_kwargs['openai_api_key'] = openai_api_key
        if anthropic_api_key:
            model_kwargs['anthropic_api_key'] = anthropic_api_key
        
        model_instance = create_model(model_id, **model_kwargs)
        
        # Test basic generation
        click.echo(f"Testing model: {model_id}")
        click.echo(f"Model info: {model_instance.get_model_info()}")
        
        test_prompt = "Hello, this is a test. Please respond with 'Test successful'."
        click.echo(f"Sending test prompt: {test_prompt}")
        
        response = await model_instance.generate(test_prompt, max_tokens=50)
        click.echo(f"Response: {response}")
        
        click.echo("✅ Model test successful!")
        
    except Exception as e:
        click.echo(f"❌ Model test failed: {e}", err=True)
        sys.exit(1)


def _print_evaluation_summary(results: EvaluationResults):
    """Print evaluation summary to console."""
    click.echo("\n" + "=" * 60)
    click.echo("EVALUATION SUMMARY")
    click.echo("=" * 60)
    click.echo(f"Model: {results.model_name}")
    click.echo(f"Timestamp: {results.timestamp}")
    click.echo(f"Risk Level: {results.config.risk_level.value.upper()}")
    click.echo(f"Overall Score: {results.overall_score:.3f}")
    click.echo(f"Risk Compliance: {'✅ PASS' if results.risk_compliance else '❌ FAIL'}")
    click.echo(f"Execution Time: {results.total_execution_time:.2f}s")
    
    # Safety metrics
    if results.safety_metrics:
        click.echo(f"\nSAFETY METRICS:")
        for metric, value in results.safety_metrics.items():
            click.echo(f"  {metric}: {value:.4f}")
    
    # Security metrics
    if results.security_metrics:
        click.echo(f"\nSECURITY METRICS:")
        for metric, value in results.security_metrics.items():
            click.echo(f"  {metric}: {value:.4f}")
    
    # Reliability metrics
    if results.reliability_metrics:
        click.echo(f"\nRELIABILITY METRICS:")
        for metric, value in results.reliability_metrics.items():
            click.echo(f"  {metric}: {value:.4f}")
    
    # Framework failures
    if results.framework_failures:
        click.echo(f"\nFRAMEWORK FAILURES:")
        for failure in results.framework_failures:
            click.echo(f"  ❌ {failure}")
    
    click.echo("=" * 60)


def _generate_comparison_report(results_list: List[EvaluationResults]) -> str:
    """Generate comparison report for multiple results."""
    report = []
    report.append("EVALUATION COMPARISON REPORT")
    report.append("=" * 50)
    
    # Model comparison table
    report.append("\nMODEL COMPARISON:")
    report.append("-" * 30)
    
    headers = ["Model", "Overall Score", "Risk Compliance", "Exec Time (s)"]
    report.append(f"{'Model':<25} {'Score':<10} {'Compliance':<12} {'Time':<10}")
    report.append("-" * 60)
    
    for result in results_list:
        compliance = "PASS" if result.risk_compliance else "FAIL"
        report.append(
            f"{result.model_name:<25} {result.overall_score:<10.3f} "
            f"{compliance:<12} {result.total_execution_time:<10.2f}"
        )
    
    # Metric comparison
    all_safety_metrics = set()
    all_security_metrics = set()
    all_reliability_metrics = set()
    
    for result in results_list:
        all_safety_metrics.update(result.safety_metrics.keys())
        all_security_metrics.update(result.security_metrics.keys())
        all_reliability_metrics.update(result.reliability_metrics.keys())
    
    # Safety comparison
    if all_safety_metrics:
        report.append("\n\nSAFETY METRICS COMPARISON:")
        report.append("-" * 30)
        for metric in sorted(all_safety_metrics):
            report.append(f"\n{metric}:")
            for result in results_list:
                value = result.safety_metrics.get(metric, "N/A")
                if isinstance(value, float):
                    report.append(f"  {result.model_name}: {value:.4f}")
                else:
                    report.append(f"  {result.model_name}: {value}")
    
    # Security comparison
    if all_security_metrics:
        report.append("\n\nSECURITY METRICS COMPARISON:")
        report.append("-" * 30)
        for metric in sorted(all_security_metrics):
            report.append(f"\n{metric}:")
            for result in results_list:
                value = result.security_metrics.get(metric, "N/A")
                if isinstance(value, float):
                    report.append(f"  {result.model_name}: {value:.4f}")
                else:
                    report.append(f"  {result.model_name}: {value}")
    
    # Reliability comparison
    if all_reliability_metrics:
        report.append("\n\nRELIABILITY METRICS COMPARISON:")
        report.append("-" * 30)
        for metric in sorted(all_reliability_metrics):
            report.append(f"\n{metric}:")
            for result in results_list:
                value = result.reliability_metrics.get(metric, "N/A")
                if isinstance(value, float):
                    report.append(f"  {result.model_name}: {value:.4f}")
                else:
                    report.append(f"  {result.model_name}: {value}")
    
    return "\n".join(report)


if __name__ == '__main__':
    cli()