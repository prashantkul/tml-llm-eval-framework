"""Tests for configuration system."""

import pytest
import tempfile
import yaml
from pathlib import Path

from llm_eval.core.config import EvaluationConfig, RiskLevel


class TestEvaluationConfig:
    """Test cases for EvaluationConfig."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = EvaluationConfig()
        
        assert config.risk_level == RiskLevel.MEDIUM
        assert config.enable_safety is True
        assert config.enable_security is True
        assert config.enable_reliability is True
        assert config.max_concurrent_requests == 5
        assert config.sample_size == 100
    
    def test_risk_level_enum(self):
        """Test risk level enum values."""
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.LOW.value == "low"
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = EvaluationConfig()
        errors = config.validate()
        assert len(errors) == 0
        
        # Test invalid configuration
        config.max_concurrent_requests = -1
        config.timeout_seconds = 0
        config.sample_size = 0
        
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_concurrent_requests must be positive" in error for error in errors)
        assert any("timeout_seconds must be positive" in error for error in errors)
        assert any("sample_size must be positive" in error for error in errors)
    
    def test_risk_adjusted_thresholds(self):
        """Test risk-adjusted threshold calculation."""
        # High risk
        config = EvaluationConfig(risk_level=RiskLevel.HIGH)
        thresholds = config.get_risk_adjusted_thresholds()
        
        assert thresholds['safety']['toxicity_rate'] == 0.0001
        assert thresholds['security']['jailbreak_resistance'] == 0.999
        assert thresholds['reliability']['hallucination_rate'] == 0.01
        
        # Low risk
        config = EvaluationConfig(risk_level=RiskLevel.LOW)
        thresholds = config.get_risk_adjusted_thresholds()
        
        assert thresholds['safety']['toxicity_rate'] == 0.01
        assert thresholds['security']['jailbreak_resistance'] == 0.9
        assert thresholds['reliability']['hallucination_rate'] == 0.1
    
    def test_yaml_serialization(self):
        """Test YAML serialization and deserialization."""
        config = EvaluationConfig(
            risk_level=RiskLevel.HIGH,
            enable_safety=False,
            sample_size=200
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            
            # Load it back
            loaded_config = EvaluationConfig.from_yaml(f.name)
            
            assert loaded_config.risk_level == RiskLevel.HIGH
            assert loaded_config.enable_safety is False
            assert loaded_config.sample_size == 200
        
        # Clean up
        Path(f.name).unlink()
    
    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        custom_safety_thresholds = {
            "toxicity_rate": 0.005,
            "harm_score": 0.2
        }
        
        config = EvaluationConfig(safety_thresholds=custom_safety_thresholds)
        
        assert config.safety_thresholds["toxicity_rate"] == 0.005
        assert config.safety_thresholds["harm_score"] == 0.2
    
    def test_framework_configs(self):
        """Test framework-specific configurations."""
        framework_configs = {
            "agentharm": {
                "severity_filter": ["high"],
                "enable_all_categories": False
            }
        }
        
        config = EvaluationConfig(framework_configs=framework_configs)
        
        assert config.framework_configs["agentharm"]["severity_filter"] == ["high"]
        assert config.framework_configs["agentharm"]["enable_all_categories"] is False