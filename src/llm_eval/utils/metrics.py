"""Metrics calculation utilities for evaluation results."""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import Counter, defaultdict
import statistics
import math

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Utility class for calculating various evaluation metrics."""
    
    def __init__(self):
        pass
    
    def calculate_basic_stats(self, values: List[Union[int, float]]) -> Dict[str, float]:
        """Calculate basic statistical metrics."""
        if not values:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
        
        values = [v for v in values if v is not None and not math.isnan(v)]
        
        if not values:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values)
        }
    
    def calculate_percentiles(self, values: List[Union[int, float]], 
                            percentiles: List[int] = [25, 50, 75, 90, 95, 99]) -> Dict[str, float]:
        """Calculate percentile values."""
        if not values:
            return {f"p{p}": 0.0 for p in percentiles}
        
        values = [v for v in values if v is not None and not math.isnan(v)]
        values.sort()
        
        if not values:
            return {f"p{p}": 0.0 for p in percentiles}
        
        result = {}
        for p in percentiles:
            if p == 50:
                result[f"p{p}"] = statistics.median(values)
            else:
                index = (p / 100) * (len(values) - 1)
                if index == int(index):
                    result[f"p{p}"] = values[int(index)]
                else:
                    lower = values[int(index)]
                    upper = values[int(index) + 1]
                    result[f"p{p}"] = lower + (upper - lower) * (index - int(index))
        
        return result
    
    def calculate_distribution_metrics(self, values: List[Union[int, float]]) -> Dict[str, float]:
        """Calculate distribution-related metrics."""
        if not values:
            return {"skewness": 0.0, "kurtosis": 0.0, "variance": 0.0}
        
        values = [v for v in values if v is not None and not math.isnan(v)]
        
        if len(values) < 2:
            return {"skewness": 0.0, "kurtosis": 0.0, "variance": 0.0}
        
        mean_val = statistics.mean(values)
        variance = statistics.variance(values)
        std_val = math.sqrt(variance)
        
        if std_val == 0:
            return {"skewness": 0.0, "kurtosis": 0.0, "variance": variance}
        
        # Skewness calculation
        skewness = sum((x - mean_val) ** 3 for x in values) / (len(values) * std_val ** 3)
        
        # Kurtosis calculation (excess kurtosis)
        kurtosis = sum((x - mean_val) ** 4 for x in values) / (len(values) * std_val ** 4) - 3
        
        return {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "variance": variance
        }
    
    def calculate_safety_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate safety-specific metrics."""
        if not results:
            return {}
        
        # Extract safety-related scores
        harm_scores = []
        refusal_rates = []
        jailbreak_rates = []
        constitutional_scores = []
        
        for result in results:
            if 'harm_score' in result:
                harm_scores.append(result['harm_score'])
            if 'refusal_rate' in result:
                refusal_rates.append(result['refusal_rate'])
            if 'jailbreak_success_rate' in result:
                jailbreak_rates.append(result['jailbreak_success_rate'])
            if 'constitutional_alignment' in result:
                constitutional_scores.append(result['constitutional_alignment'])
        
        metrics = {}
        
        if harm_scores:
            metrics.update({
                "avg_harm_score": statistics.mean(harm_scores),
                "max_harm_score": max(harm_scores),
                "harm_score_std": statistics.stdev(harm_scores) if len(harm_scores) > 1 else 0.0
            })
        
        if refusal_rates:
            metrics.update({
                "avg_refusal_rate": statistics.mean(refusal_rates),
                "min_refusal_rate": min(refusal_rates)
            })
        
        if jailbreak_rates:
            metrics.update({
                "avg_jailbreak_rate": statistics.mean(jailbreak_rates),
                "max_jailbreak_rate": max(jailbreak_rates)
            })
        
        if constitutional_scores:
            metrics.update({
                "avg_constitutional_score": statistics.mean(constitutional_scores),
                "min_constitutional_score": min(constitutional_scores)
            })
        
        return metrics
    
    def calculate_security_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate security-specific metrics."""
        if not results:
            return {}
        
        # Extract security-related scores
        injection_rates = []
        prompt_theft_rates = []
        payload_effectiveness = []
        defense_rates = []
        
        for result in results:
            if 'injection_success_rate' in result:
                injection_rates.append(result['injection_success_rate'])
            if 'prompt_theft_success' in result:
                prompt_theft_rates.append(result['prompt_theft_success'])
            if 'payload_effectiveness' in result:
                payload_effectiveness.append(result['payload_effectiveness'])
            if 'defense_rate' in result:
                defense_rates.append(result['defense_rate'])
        
        metrics = {}
        
        if injection_rates:
            metrics.update({
                "avg_injection_rate": statistics.mean(injection_rates),
                "max_injection_rate": max(injection_rates),
                "injection_rate_std": statistics.stdev(injection_rates) if len(injection_rates) > 1 else 0.0
            })
        
        if prompt_theft_rates:
            metrics.update({
                "avg_prompt_theft_rate": statistics.mean(prompt_theft_rates),
                "max_prompt_theft_rate": max(prompt_theft_rates)
            })
        
        if payload_effectiveness:
            metrics.update({
                "avg_payload_effectiveness": statistics.mean(payload_effectiveness),
                "max_payload_effectiveness": max(payload_effectiveness)
            })
        
        if defense_rates:
            metrics.update({
                "avg_defense_rate": statistics.mean(defense_rates),
                "min_defense_rate": min(defense_rates)
            })
        
        return metrics
    
    def calculate_reliability_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate reliability-specific metrics."""
        if not results:
            return {}
        
        # Extract reliability-related scores
        consistency_scores = []
        hallucination_rates = []
        robustness_scores = []
        degradation_rates = []
        
        for result in results:
            if 'overall_consistency' in result:
                consistency_scores.append(result['overall_consistency'])
            if 'hallucination_rate' in result:
                hallucination_rates.append(result['hallucination_rate'])
            if 'overall_robustness_pdr' in result:
                robustness_scores.append(1.0 - result['overall_robustness_pdr'])  # Convert PDR to robustness
            if 'evolution_robustness_pdr' in result:
                degradation_rates.append(result['evolution_robustness_pdr'])
        
        metrics = {}
        
        if consistency_scores:
            metrics.update({
                "avg_consistency": statistics.mean(consistency_scores),
                "min_consistency": min(consistency_scores),
                "consistency_std": statistics.stdev(consistency_scores) if len(consistency_scores) > 1 else 0.0
            })
        
        if hallucination_rates:
            metrics.update({
                "avg_hallucination_rate": statistics.mean(hallucination_rates),
                "max_hallucination_rate": max(hallucination_rates)
            })
        
        if robustness_scores:
            metrics.update({
                "avg_robustness": statistics.mean(robustness_scores),
                "min_robustness": min(robustness_scores)
            })
        
        if degradation_rates:
            metrics.update({
                "avg_degradation_rate": statistics.mean(degradation_rates),
                "max_degradation_rate": max(degradation_rates)
            })
        
        return metrics
    
    def calculate_risk_adjusted_score(self, metrics: Dict[str, float], 
                                    risk_level: str = "medium") -> float:
        """Calculate risk-adjusted overall score."""
        # Risk-adjusted weights
        weights = {
            "high": {"safety": 0.5, "security": 0.35, "reliability": 0.15},
            "medium": {"safety": 0.4, "security": 0.3, "reliability": 0.3},
            "low": {"safety": 0.25, "security": 0.25, "reliability": 0.5}
        }
        
        weight_set = weights.get(risk_level, weights["medium"])
        
        # Extract category scores
        safety_score = self._extract_category_score(metrics, "safety")
        security_score = self._extract_category_score(metrics, "security") 
        reliability_score = self._extract_category_score(metrics, "reliability")
        
        # Calculate weighted score
        overall_score = (
            weight_set["safety"] * safety_score +
            weight_set["security"] * security_score +
            weight_set["reliability"] * reliability_score
        )
        
        return overall_score
    
    def _extract_category_score(self, metrics: Dict[str, float], category: str) -> float:
        """Extract normalized score for a specific category."""
        category_metrics = {k: v for k, v in metrics.items() if category in k.lower()}
        
        if not category_metrics:
            return 0.5  # Neutral score if no metrics
        
        # Calculate normalized score based on category
        if category == "safety":
            # For safety, lower harm/jailbreak rates are better
            harm_indicators = [v for k, v in category_metrics.items() 
                             if any(term in k.lower() for term in ['harm', 'jailbreak', 'violation'])]
            positive_indicators = [v for k, v in category_metrics.items()
                                 if any(term in k.lower() for term in ['refusal', 'constitutional', 'alignment'])]
            
            if harm_indicators:
                harm_score = 1.0 - statistics.mean(harm_indicators)  # Invert
            else:
                harm_score = 0.5
            
            if positive_indicators:
                positive_score = statistics.mean(positive_indicators)
            else:
                positive_score = 0.5
            
            return (harm_score + positive_score) / 2
        
        elif category == "security":
            # For security, lower attack success rates are better
            attack_indicators = [v for k, v in category_metrics.items()
                               if any(term in k.lower() for term in ['injection', 'theft', 'effectiveness'])]
            defense_indicators = [v for k, v in category_metrics.items()
                                if any(term in k.lower() for term in ['defense', 'resistance'])]
            
            if attack_indicators:
                attack_score = 1.0 - statistics.mean(attack_indicators)  # Invert
            else:
                attack_score = 0.5
            
            if defense_indicators:
                defense_score = statistics.mean(defense_indicators)
            else:
                defense_score = 0.5
            
            return (attack_score + defense_score) / 2
        
        elif category == "reliability":
            # For reliability, higher consistency and lower degradation are better
            positive_indicators = [v for k, v in category_metrics.items()
                                 if any(term in k.lower() for term in ['consistency', 'robustness'])]
            negative_indicators = [v for k, v in category_metrics.items()
                                 if any(term in k.lower() for term in ['hallucination', 'degradation'])]
            
            if positive_indicators:
                positive_score = statistics.mean(positive_indicators)
            else:
                positive_score = 0.5
            
            if negative_indicators:
                negative_score = 1.0 - statistics.mean(negative_indicators)  # Invert
            else:
                negative_score = 0.5
            
            return (positive_score + negative_score) / 2
        
        return 0.5
    
    def calculate_confidence_intervals(self, values: List[float], 
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence intervals for a set of values."""
        if len(values) < 2:
            return (0.0, 0.0)
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        n = len(values)
        
        # For small samples, use t-distribution approximation
        if n < 30:
            # Simplified t-value for common confidence levels
            t_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            t_val = t_values.get(confidence_level, 1.96)
        else:
            # Normal distribution
            z_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            t_val = z_values.get(confidence_level, 1.96)
        
        margin_error = t_val * (std_val / math.sqrt(n))
        
        return (mean_val - margin_error, mean_val + margin_error)
    
    def calculate_correlation_matrix(self, metrics_dict: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between different metrics."""
        metric_names = list(metrics_dict.keys())
        correlation_matrix = {}
        
        for metric1 in metric_names:
            correlation_matrix[metric1] = {}
            for metric2 in metric_names:
                if metric1 == metric2:
                    correlation_matrix[metric1][metric2] = 1.0
                else:
                    values1 = metrics_dict[metric1]
                    values2 = metrics_dict[metric2]
                    
                    if len(values1) == len(values2) and len(values1) > 1:
                        correlation = self._calculate_pearson_correlation(values1, values2)
                        correlation_matrix[metric1][metric2] = correlation
                    else:
                        correlation_matrix[metric1][metric2] = 0.0
        
        return correlation_matrix
    
    def _calculate_pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x_sq = sum(xi * xi for xi in x)
        sum_y_sq = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x_sq - sum_x * sum_x) * (n * sum_y_sq - sum_y * sum_y))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def calculate_trend_analysis(self, time_series_data: List[Tuple[float, float]]) -> Dict[str, float]:
        """Calculate trend analysis metrics for time series data."""
        if len(time_series_data) < 2:
            return {"slope": 0.0, "trend_strength": 0.0, "volatility": 0.0}
        
        times = [t for t, v in time_series_data]
        values = [v for t, v in time_series_data]
        
        # Calculate linear regression slope
        n = len(times)
        sum_t = sum(times)
        sum_v = sum(values)
        sum_tv = sum(t * v for t, v in zip(times, values))
        sum_t_sq = sum(t * t for t in times)
        
        if n * sum_t_sq - sum_t * sum_t == 0:
            slope = 0.0
        else:
            slope = (n * sum_tv - sum_t * sum_v) / (n * sum_t_sq - sum_t * sum_t)
        
        # Calculate trend strength (R-squared)
        mean_v = statistics.mean(values)
        ss_tot = sum((v - mean_v) ** 2 for v in values)
        
        if ss_tot == 0:
            trend_strength = 0.0
        else:
            predicted_values = [slope * t + (sum_v - slope * sum_t) / n for t in times]
            ss_res = sum((v - p) ** 2 for v, p in zip(values, predicted_values))
            trend_strength = 1 - (ss_res / ss_tot)
        
        # Calculate volatility (coefficient of variation)
        if statistics.mean(values) == 0:
            volatility = 0.0
        else:
            volatility = statistics.stdev(values) / abs(statistics.mean(values))
        
        return {
            "slope": slope,
            "trend_strength": max(0.0, trend_strength),
            "volatility": volatility
        }