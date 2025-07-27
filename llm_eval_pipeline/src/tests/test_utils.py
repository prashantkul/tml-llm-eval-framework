"""Tests for utility modules."""

import pytest
import tempfile
import json
import csv
from pathlib import Path

from llm_eval.utils.data_loader import DataLoader
from llm_eval.utils.metrics import MetricsCalculator


class TestDataLoader:
    """Test cases for DataLoader utility."""
    
    def test_json_loading(self):
        """Test JSON dataset loading."""
        test_data = [
            {"prompt": "Test 1", "category": "safety"},
            {"prompt": "Test 2", "category": "security"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            
            loader = DataLoader()
            loaded_data = loader.load_dataset(f.name)
            
            assert len(loaded_data) == 2
            assert loaded_data[0]["prompt"] == "Test 1"
            assert loaded_data[1]["category"] == "security"
        
        Path(f.name).unlink()
    
    def test_csv_loading(self):
        """Test CSV dataset loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["prompt", "category", "difficulty"])
            writer.writerow(["Test prompt 1", "safety", "easy"])
            writer.writerow(["Test prompt 2", "security", "hard"])
            
            f.flush()
            
            loader = DataLoader()
            loaded_data = loader.load_dataset(f.name)
            
            assert len(loaded_data) == 2
            assert loaded_data[0]["prompt"] == "Test prompt 1"
            assert loaded_data[1]["difficulty"] == "hard"
        
        Path(f.name).unlink()
    
    def test_dataset_filtering(self):
        """Test dataset filtering functionality."""
        dataset = [
            {"prompt": "Test 1", "category": "safety", "difficulty": "easy"},
            {"prompt": "Test 2", "category": "security", "difficulty": "hard"},
            {"prompt": "Test 3", "category": "safety", "difficulty": "medium"},
            {"prompt": "Test 4", "category": "reliability", "difficulty": "easy"}
        ]
        
        loader = DataLoader()
        
        # Filter by category
        safety_data = loader.filter_dataset(dataset, {"category": "safety"})
        assert len(safety_data) == 2
        assert all(item["category"] == "safety" for item in safety_data)
        
        # Filter by multiple criteria
        easy_safety = loader.filter_dataset(dataset, {
            "category": "safety",
            "difficulty": "easy"
        })
        assert len(easy_safety) == 1
        assert easy_safety[0]["prompt"] == "Test 1"
        
        # Filter by list of values
        easy_or_hard = loader.filter_dataset(dataset, {
            "difficulty": ["easy", "hard"]
        })
        assert len(easy_or_hard) == 3
    
    def test_dataset_sampling(self):
        """Test dataset sampling."""
        dataset = [{"id": i} for i in range(100)]
        
        loader = DataLoader()
        
        # Sample smaller than dataset
        sample = loader.sample_dataset(dataset, 10, random_seed=42)
        assert len(sample) == 10
        assert all(item in dataset for item in sample)
        
        # Sample larger than dataset
        large_sample = loader.sample_dataset(dataset, 150, random_seed=42)
        assert len(large_sample) == 100  # Should return full dataset
        
        # Test reproducibility
        sample1 = loader.sample_dataset(dataset, 10, random_seed=42)
        sample2 = loader.sample_dataset(dataset, 10, random_seed=42)
        assert sample1 == sample2
    
    def test_schema_validation(self):
        """Test dataset schema validation."""
        valid_dataset = [
            {"prompt": "Test 1", "category": "safety", "expected": "result"},
            {"prompt": "Test 2", "category": "security", "expected": "result"}
        ]
        
        invalid_dataset = [
            {"prompt": "Test 1", "category": "safety"},  # Missing 'expected'
            {"prompt": "Test 2", "expected": "result"}   # Missing 'category'
        ]
        
        loader = DataLoader()
        
        # Valid dataset should pass
        assert loader.validate_dataset_schema(valid_dataset, ["prompt", "category", "expected"]) is True
        
        # Invalid dataset should fail
        assert loader.validate_dataset_schema(invalid_dataset, ["prompt", "category", "expected"]) is False
        
        # Empty dataset should fail
        assert loader.validate_dataset_schema([], ["prompt"]) is False
    
    def test_dataset_stats(self):
        """Test dataset statistics generation."""
        dataset = [
            {"prompt": "Test 1", "category": "safety", "score": 0.8},
            {"prompt": "Test 2", "category": "security", "score": 0.9},
            {"prompt": "Test 3", "category": "safety"}  # Missing score
        ]
        
        loader = DataLoader()
        stats = loader.get_dataset_stats(dataset)
        
        assert stats["size"] == 3
        assert stats["field_count"] == 3
        assert "prompt" in stats["fields"]
        assert "category" in stats["fields"]
        assert "score" in stats["fields"]
        
        # Check field coverage
        assert stats["field_stats"]["prompt"]["coverage"] == 1.0  # All items have prompt
        assert stats["field_stats"]["score"]["coverage"] == 2/3   # 2 out of 3 have score


class TestMetricsCalculator:
    """Test cases for MetricsCalculator utility."""
    
    def test_basic_stats(self):
        """Test basic statistical calculations."""
        calculator = MetricsCalculator()
        
        values = [1, 2, 3, 4, 5]
        stats = calculator.calculate_basic_stats(values)
        
        assert stats["count"] == 5
        assert stats["mean"] == 3.0
        assert stats["min"] == 1
        assert stats["max"] == 5
        assert stats["median"] == 3.0
        assert stats["std"] > 0
        
        # Empty list
        empty_stats = calculator.calculate_basic_stats([])
        assert empty_stats["count"] == 0
        assert empty_stats["mean"] == 0.0
    
    def test_percentiles(self):
        """Test percentile calculations."""
        calculator = MetricsCalculator()
        
        values = list(range(1, 101))  # 1 to 100
        percentiles = calculator.calculate_percentiles(values, [25, 50, 75, 90])
        
        assert percentiles["p25"] == 25.5
        assert percentiles["p50"] == 50.5  # Median
        assert percentiles["p75"] == 75.5
        assert percentiles["p90"] == 90.5
    
    def test_safety_metrics(self):
        """Test safety-specific metrics calculation."""
        calculator = MetricsCalculator()
        
        safety_results = [
            {"harm_score": 0.1, "refusal_rate": 0.9, "jailbreak_success_rate": 0.05},
            {"harm_score": 0.2, "refusal_rate": 0.8, "jailbreak_success_rate": 0.1},
            {"harm_score": 0.15, "refusal_rate": 0.85, "jailbreak_success_rate": 0.08}
        ]
        
        metrics = calculator.calculate_safety_metrics(safety_results)
        
        assert "avg_harm_score" in metrics
        assert metrics["avg_harm_score"] == 0.15  # Average of 0.1, 0.2, 0.15
        assert "max_harm_score" in metrics
        assert metrics["max_harm_score"] == 0.2
        assert "avg_refusal_rate" in metrics
        assert metrics["avg_refusal_rate"] == 0.85  # Average of 0.9, 0.8, 0.85
    
    def test_security_metrics(self):
        """Test security-specific metrics calculation."""
        calculator = MetricsCalculator()
        
        security_results = [
            {"injection_success_rate": 0.05, "defense_rate": 0.95, "payload_effectiveness": 0.1},
            {"injection_success_rate": 0.03, "defense_rate": 0.97, "payload_effectiveness": 0.08},
        ]
        
        metrics = calculator.calculate_security_metrics(security_results)
        
        assert "avg_injection_rate" in metrics
        assert metrics["avg_injection_rate"] == 0.04  # Average of 0.05, 0.03
        assert "max_injection_rate" in metrics
        assert metrics["max_injection_rate"] == 0.05
        assert "avg_defense_rate" in metrics
        assert metrics["avg_defense_rate"] == 0.96
    
    def test_reliability_metrics(self):
        """Test reliability-specific metrics calculation."""
        calculator = MetricsCalculator()
        
        reliability_results = [
            {"overall_consistency": 0.85, "hallucination_rate": 0.05, "overall_robustness_pdr": 0.15},
            {"overall_consistency": 0.9, "hallucination_rate": 0.03, "overall_robustness_pdr": 0.12},
        ]
        
        metrics = calculator.calculate_reliability_metrics(reliability_results)
        
        assert "avg_consistency" in metrics
        assert metrics["avg_consistency"] == 0.875  # Average of 0.85, 0.9
        assert "min_consistency" in metrics
        assert metrics["min_consistency"] == 0.85
        assert "avg_hallucination_rate" in metrics
        assert metrics["avg_hallucination_rate"] == 0.04
        assert "avg_robustness" in metrics
        # Robustness is 1 - PDR, so (1-0.15 + 1-0.12)/2 = (0.85 + 0.88)/2 = 0.865
        assert abs(metrics["avg_robustness"] - 0.865) < 0.001
    
    def test_risk_adjusted_score(self):
        """Test risk-adjusted score calculation."""
        calculator = MetricsCalculator()
        
        metrics = {
            "safety_harm_score": 0.1,
            "safety_refusal_rate": 0.9,
            "security_injection_rate": 0.05,
            "security_defense_rate": 0.95,
            "reliability_consistency": 0.85,
            "reliability_hallucination_rate": 0.05
        }
        
        # Test different risk levels
        high_risk_score = calculator.calculate_risk_adjusted_score(metrics, "high")
        medium_risk_score = calculator.calculate_risk_adjusted_score(metrics, "medium")
        low_risk_score = calculator.calculate_risk_adjusted_score(metrics, "low")
        
        # All should be between 0 and 1
        assert 0 <= high_risk_score <= 1
        assert 0 <= medium_risk_score <= 1
        assert 0 <= low_risk_score <= 1
        
        # Scores should be reasonable (not testing exact values due to complexity)
        assert isinstance(high_risk_score, float)
        assert isinstance(medium_risk_score, float)
        assert isinstance(low_risk_score, float)
    
    def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        calculator = MetricsCalculator()
        
        values = [0.8, 0.85, 0.9, 0.82, 0.88, 0.87, 0.83, 0.89, 0.86, 0.84]
        
        ci_95 = calculator.calculate_confidence_intervals(values, 0.95)
        ci_90 = calculator.calculate_confidence_intervals(values, 0.90)
        
        # 95% CI should be wider than 90% CI
        ci_95_width = ci_95[1] - ci_95[0]
        ci_90_width = ci_90[1] - ci_90[0]
        assert ci_95_width > ci_90_width
        
        # Mean should be within both intervals
        mean_val = sum(values) / len(values)
        assert ci_95[0] <= mean_val <= ci_95[1]
        assert ci_90[0] <= mean_val <= ci_90[1]
    
    def test_correlation_matrix(self):
        """Test correlation matrix calculation."""
        calculator = MetricsCalculator()
        
        metrics_dict = {
            "metric_a": [1, 2, 3, 4, 5],
            "metric_b": [2, 4, 6, 8, 10],  # Perfect positive correlation with metric_a
            "metric_c": [5, 4, 3, 2, 1]    # Perfect negative correlation with metric_a
        }
        
        correlation_matrix = calculator.calculate_correlation_matrix(metrics_dict)
        
        # Self-correlation should be 1.0
        assert correlation_matrix["metric_a"]["metric_a"] == 1.0
        assert correlation_matrix["metric_b"]["metric_b"] == 1.0
        
        # Perfect positive correlation
        assert abs(correlation_matrix["metric_a"]["metric_b"] - 1.0) < 0.001
        
        # Perfect negative correlation
        assert abs(correlation_matrix["metric_a"]["metric_c"] - (-1.0)) < 0.001