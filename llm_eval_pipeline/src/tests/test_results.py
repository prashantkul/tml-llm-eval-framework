"""Tests for results system."""

import pytest
import json
import tempfile
from datetime import datetime
from pathlib import Path

from llm_eval.core.results import EvaluationResults, FrameworkResult
from llm_eval.core.config import EvaluationConfig, RiskLevel


class TestFrameworkResult:
    """Test cases for FrameworkResult."""
    
    def test_framework_result_creation(self):
        """Test framework result creation."""
        result = FrameworkResult(
            framework_name="test_framework",
            metrics={"accuracy": 0.85, "precision": 0.78},
            execution_time=12.5,
            success=True
        )
        
        assert result.framework_name == "test_framework"
        assert result.metrics["accuracy"] == 0.85
        assert result.execution_time == 12.5
        assert result.success is True
        assert result.error_message is None
    
    def test_failed_framework_result(self):
        """Test failed framework result."""
        result = FrameworkResult(
            framework_name="failed_framework",
            success=False,
            error_message="Connection timeout"
        )
        
        assert result.success is False
        assert result.error_message == "Connection timeout"
        assert len(result.metrics) == 0


class TestEvaluationResults:
    """Test cases for EvaluationResults."""
    
    def test_results_creation(self):
        """Test evaluation results creation."""
        config = EvaluationConfig(risk_level=RiskLevel.MEDIUM)
        results = EvaluationResults(
            model_name="test-model",
            timestamp=datetime.now(),
            config=config
        )
        
        assert results.model_name == "test-model"
        assert results.config.risk_level == RiskLevel.MEDIUM
        assert results.overall_score == 0.0
        assert results.risk_compliance is False
    
    def test_metrics_aggregation(self):
        """Test metrics aggregation from framework results."""
        config = EvaluationConfig()
        results = EvaluationResults(
            model_name="test-model",
            timestamp=datetime.now(),
            config=config
        )
        
        # Add safety results
        safety_result1 = FrameworkResult(
            framework_name="safety1",
            metrics={"harm_score": 0.1, "toxicity_rate": 0.05},
            success=True
        )
        safety_result2 = FrameworkResult(
            framework_name="safety2", 
            metrics={"harm_score": 0.2, "refusal_rate": 0.9},
            success=True
        )
        
        results.safety_results = [safety_result1, safety_result2]
        
        # Add security results
        security_result = FrameworkResult(
            framework_name="security1",
            metrics={"injection_success_rate": 0.03, "defense_rate": 0.97},
            success=True
        )
        
        results.security_results = [security_result]
        
        # Aggregate metrics
        results.aggregate_metrics()
        
        # Check aggregated safety metrics
        assert "harm_score" in results.safety_metrics
        assert results.safety_metrics["harm_score"] == 0.15  # Average of 0.1 and 0.2
        assert results.safety_metrics["toxicity_rate"] == 0.05
        assert results.safety_metrics["refusal_rate"] == 0.9
        
        # Check aggregated security metrics
        assert "injection_success_rate" in results.security_metrics
        assert results.security_metrics["injection_success_rate"] == 0.03
        assert results.security_metrics["defense_rate"] == 0.97
        
        # Check overall score calculation
        assert results.overall_score > 0
    
    def test_threshold_checking(self):
        """Test threshold checking for risk compliance."""
        config = EvaluationConfig(risk_level=RiskLevel.MEDIUM)
        results = EvaluationResults(
            model_name="test-model",
            timestamp=datetime.now(),
            config=config
        )
        
        # Set metrics that should pass medium risk thresholds
        results.safety_metrics = {
            "toxicity_rate": 0.001,  # Below 0.005 threshold
            "harm_score": 0.05       # Below 0.15 threshold
        }
        
        results.security_metrics = {
            "injection_success_rate": 0.02,  # Below 0.08 threshold
            "jailbreak_resistance": 0.95     # Above 0.92 threshold
        }
        
        # Check thresholds
        compliance = results.check_thresholds()
        
        assert "safety_toxicity_rate" in compliance
        assert compliance["safety_toxicity_rate"] is True
        assert "safety_harm_score" in compliance
        assert compliance["safety_harm_score"] is True
        assert "security_injection_success_rate" in compliance
        assert compliance["security_injection_success_rate"] is True
        assert "security_jailbreak_resistance" in compliance
        assert compliance["security_jailbreak_resistance"] is True
        
        # Should be compliant if all individual checks pass
        assert results.risk_compliance is True
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        config = EvaluationConfig(risk_level=RiskLevel.HIGH)
        timestamp = datetime.now()
        
        results = EvaluationResults(
            model_name="test-model",
            timestamp=timestamp,
            config=config
        )
        
        results.safety_metrics = {"harm_score": 0.1}
        results.overall_score = 0.85
        
        # Serialize to JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            results.to_json(f.name)
            
            # Load it back
            loaded_results = EvaluationResults.from_json(
                Path(f.name).read_text()
            )
            
            assert loaded_results.model_name == "test-model"
            assert loaded_results.config.risk_level == RiskLevel.HIGH
            assert loaded_results.safety_metrics["harm_score"] == 0.1
            assert loaded_results.overall_score == 0.85
            assert loaded_results.timestamp == timestamp
        
        # Clean up
        Path(f.name).unlink()
    
    def test_report_generation(self):
        """Test report generation in different formats."""
        config = EvaluationConfig()
        results = EvaluationResults(
            model_name="test-model",
            timestamp=datetime.now(),
            config=config
        )
        
        results.safety_metrics = {"harm_score": 0.1, "toxicity_rate": 0.05}
        results.security_metrics = {"injection_success_rate": 0.03}
        results.overall_score = 0.85
        results.risk_compliance = True
        
        # Test text report
        text_report = results.to_report("text")
        assert "test-model" in text_report
        assert "Overall Score: 0.850" in text_report
        assert "SAFETY METRICS" in text_report
        
        # Test markdown report
        markdown_report = results.to_report("markdown")
        assert "# LLM Evaluation Report" in markdown_report
        assert "**Model:** test-model" in markdown_report
        assert "| harm_score |" in markdown_report
        
        # Test HTML report
        html_report = results.to_report("html")
        assert "<html>" in html_report
        assert "test-model" in html_report
        assert "<table" in html_report
    
    def test_summary_stats(self):
        """Test summary statistics generation."""
        config = EvaluationConfig()
        results = EvaluationResults(
            model_name="test-model",
            timestamp=datetime.now(),
            config=config
        )
        
        # Add some framework results
        results.safety_results = [
            FrameworkResult("safety1", {"harm_score": 0.1}, success=True),
            FrameworkResult("safety2", {"toxicity_rate": 0.05}, success=False, error_message="Failed")
        ]
        results.framework_failures = ["safety2: Failed"]
        results.total_execution_time = 120.5
        results.overall_score = 0.75
        results.risk_compliance = True
        
        stats = results.get_summary_stats()
        
        assert stats["model_name"] == "test-model"
        assert stats["overall_score"] == 0.75
        assert stats["risk_compliance"] is True
        assert stats["total_frameworks"] == 2
        assert stats["failed_frameworks"] == 1
        assert stats["execution_time"] == 120.5
        assert stats["metrics_count"]["safety"] == 0  # No aggregated metrics yet