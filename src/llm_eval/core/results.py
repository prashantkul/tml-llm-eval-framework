"""Results aggregation and reporting system."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging
from pathlib import Path

from .config import EvaluationConfig, RiskLevel

logger = logging.getLogger(__name__)


@dataclass
class FrameworkResult:
    """Results from a single evaluation framework."""
    framework_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class EvaluationResults:
    """Comprehensive evaluation results."""
    model_name: str
    timestamp: datetime
    config: EvaluationConfig
    
    # Framework results
    safety_results: List[FrameworkResult] = field(default_factory=list)
    security_results: List[FrameworkResult] = field(default_factory=list) 
    reliability_results: List[FrameworkResult] = field(default_factory=list)
    
    # Aggregated metrics
    safety_metrics: Dict[str, float] = field(default_factory=dict)
    security_metrics: Dict[str, float] = field(default_factory=dict)
    reliability_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Overall assessment
    overall_score: float = 0.0
    risk_compliance: bool = False
    compliance_details: Dict[str, bool] = field(default_factory=dict)
    
    # Execution metadata
    total_execution_time: float = 0.0
    framework_failures: List[str] = field(default_factory=list)
    
    def aggregate_metrics(self) -> None:
        """Aggregate metrics from individual framework results."""
        # Aggregate safety metrics
        self.safety_metrics = self._aggregate_framework_metrics(self.safety_results)
        
        # Aggregate security metrics
        self.security_metrics = self._aggregate_framework_metrics(self.security_results)
        
        # Aggregate reliability metrics  
        self.reliability_metrics = self._aggregate_framework_metrics(self.reliability_results)
        
        # Calculate overall score
        self._calculate_overall_score()
        
        # Check compliance
        self.check_thresholds()
    
    def _aggregate_framework_metrics(self, results: List[FrameworkResult]) -> Dict[str, float]:
        """Aggregate metrics from multiple framework results."""
        aggregated = {}
        metric_counts = {}
        
        for result in results:
            if not result.success:
                continue
                
            for metric, value in result.metrics.items():
                # Only aggregate numeric values
                if isinstance(value, (int, float)):
                    if metric not in aggregated:
                        aggregated[metric] = 0.0
                        metric_counts[metric] = 0
                    
                    aggregated[metric] += value
                    metric_counts[metric] += 1
        
        # Average the metrics
        for metric in aggregated:
            if metric_counts[metric] > 0:
                aggregated[metric] /= metric_counts[metric]
        
        return aggregated
    
    def _calculate_overall_score(self) -> None:
        """Calculate overall risk-adjusted score."""
        weights = self._get_risk_weights()
        
        # Calculate category scores (inverted for risk metrics where lower is better)
        safety_score = self._calculate_category_score(self.safety_metrics, invert_risk_metrics=True)
        security_score = self._calculate_category_score(self.security_metrics, invert_risk_metrics=True)
        reliability_score = self._calculate_category_score(self.reliability_metrics, invert_risk_metrics=False)
        
        # Weighted average
        self.overall_score = (
            weights['safety'] * safety_score +
            weights['security'] * security_score + 
            weights['reliability'] * reliability_score
        )
    
    def _calculate_category_score(self, metrics: Dict[str, float], invert_risk_metrics: bool = True) -> float:
        """Calculate normalized score for a category of metrics."""
        if not metrics:
            return 0.0
        
        scores = []
        for metric, value in metrics.items():
            if invert_risk_metrics and any(risk_term in metric.lower() for risk_term in ['rate', 'success', 'leakage']):
                # For risk metrics, lower values are better
                score = 1.0 - min(value, 1.0)
            else:
                # For quality metrics, higher values are better
                score = min(value, 1.0)
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _get_risk_weights(self) -> Dict[str, float]:
        """Get risk-adjusted weights for different evaluation categories."""
        if self.config.risk_level == RiskLevel.HIGH:
            return {'safety': 0.5, 'security': 0.35, 'reliability': 0.15}
        elif self.config.risk_level == RiskLevel.MEDIUM:
            return {'safety': 0.4, 'security': 0.3, 'reliability': 0.3}
        else:  # LOW
            return {'safety': 0.25, 'security': 0.25, 'reliability': 0.5}
    
    def check_thresholds(self) -> Dict[str, bool]:
        """Check if metrics meet risk-based thresholds."""
        thresholds = self.config.get_risk_adjusted_thresholds()
        compliance = {}
        
        # Check safety thresholds
        for metric, threshold in thresholds['safety'].items():
            if metric in self.safety_metrics:
                # For risk metrics, value should be below threshold
                if any(risk_term in metric.lower() for risk_term in ['rate', 'success', 'leakage', 'score']):
                    compliance[f"safety_{metric}"] = self.safety_metrics[metric] <= threshold
                else:
                    # For quality metrics, value should be above threshold
                    compliance[f"safety_{metric}"] = self.safety_metrics[metric] >= threshold
        
        # Check security thresholds
        for metric, threshold in thresholds['security'].items():
            if metric in self.security_metrics:
                if any(risk_term in metric.lower() for risk_term in ['rate', 'success', 'leakage', 'score']):
                    compliance[f"security_{metric}"] = self.security_metrics[metric] <= threshold
                else:
                    compliance[f"security_{metric}"] = self.security_metrics[metric] >= threshold
        
        # Check reliability thresholds
        for metric, threshold in thresholds['reliability'].items():
            if metric in self.reliability_metrics:
                if any(risk_term in metric.lower() for risk_term in ['rate', 'degradation', 'pdr']):
                    compliance[f"reliability_{metric}"] = self.reliability_metrics[metric] <= threshold
                else:
                    compliance[f"reliability_{metric}"] = self.reliability_metrics[metric] >= threshold
        
        self.compliance_details = compliance
        self.risk_compliance = all(compliance.values()) if compliance else False
        
        return compliance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        result_dict = asdict(self)
        result_dict['timestamp'] = self.timestamp.isoformat()
        result_dict['config'] = asdict(self.config)
        result_dict['config']['risk_level'] = self.config.risk_level.value
        return result_dict
    
    def to_json(self, file_path: Optional[str] = None) -> str:
        """Export results to JSON."""
        result_dict = self.to_dict()
        json_str = json.dumps(result_dict, indent=2, default=str)
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(json_str)
            logger.info(f"Results saved to {file_path}")
        
        return json_str
    
    @classmethod
    def from_json(cls, json_str: str) -> "EvaluationResults":
        """Load results from JSON string."""
        data = json.loads(json_str)
        
        # Convert timestamp back
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Reconstruct config
        config_data = data['config']
        config_data['risk_level'] = RiskLevel(config_data['risk_level'])
        data['config'] = EvaluationConfig(**config_data)
        
        # Reconstruct framework results
        for category in ['safety_results', 'security_results', 'reliability_results']:
            if category in data:
                data[category] = [FrameworkResult(**result) for result in data[category]]
        
        return cls(**data)
    
    def to_report(self, format: str = "text") -> str:
        """Generate human-readable report."""
        if format == "text":
            return self._generate_text_report()
        elif format == "html":
            return self._generate_html_report()
        elif format == "markdown":
            return self._generate_markdown_report()
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _generate_text_report(self) -> str:
        """Generate plain text report."""
        report = []
        report.append("=" * 60)
        report.append("LLM EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Model: {self.model_name}")
        report.append(f"Timestamp: {self.timestamp}")
        report.append(f"Risk Level: {self.config.risk_level.value.upper()}")
        report.append(f"Overall Score: {self.overall_score:.3f}")
        report.append(f"Risk Compliance: {'✓ PASS' if self.risk_compliance else '✗ FAIL'}")
        report.append(f"Execution Time: {self.total_execution_time:.2f}s")
        report.append("")
        
        # Safety metrics
        if self.safety_metrics:
            report.append("SAFETY METRICS")
            report.append("-" * 20)
            for metric, value in self.safety_metrics.items():
                status = self._get_metric_status('safety', metric, value)
                report.append(f"  {metric}: {value:.4f} {status}")
            report.append("")
        
        # Security metrics
        if self.security_metrics:
            report.append("SECURITY METRICS")
            report.append("-" * 20)
            for metric, value in self.security_metrics.items():
                status = self._get_metric_status('security', metric, value)
                report.append(f"  {metric}: {value:.4f} {status}")
            report.append("")
        
        # Reliability metrics
        if self.reliability_metrics:
            report.append("RELIABILITY METRICS")
            report.append("-" * 20)
            for metric, value in self.reliability_metrics.items():
                status = self._get_metric_status('reliability', metric, value)
                report.append(f"  {metric}: {value:.4f} {status}")
            report.append("")
        
        # Framework failures
        if self.framework_failures:
            report.append("FRAMEWORK FAILURES")
            report.append("-" * 20)
            for failure in self.framework_failures:
                report.append(f"  ✗ {failure}")
            report.append("")
        
        return "\n".join(report)
    
    def _generate_markdown_report(self) -> str:
        """Generate Markdown report."""
        report = []
        report.append("# LLM Evaluation Report")
        report.append("")
        report.append(f"**Model:** {self.model_name}")
        report.append(f"**Timestamp:** {self.timestamp}")
        report.append(f"**Risk Level:** {self.config.risk_level.value.upper()}")
        report.append(f"**Overall Score:** {self.overall_score:.3f}")
        report.append(f"**Risk Compliance:** {'✅ PASS' if self.risk_compliance else '❌ FAIL'}")
        report.append(f"**Execution Time:** {self.total_execution_time:.2f}s")
        report.append("")
        
        # Metrics tables
        for category, metrics in [("Safety", self.safety_metrics), 
                                 ("Security", self.security_metrics),
                                 ("Reliability", self.reliability_metrics)]:
            if metrics:
                report.append(f"## {category} Metrics")
                report.append("")
                report.append("| Metric | Value | Status |")
                report.append("|--------|-------|--------|")
                for metric, value in metrics.items():
                    status = self._get_metric_status(category.lower(), metric, value)
                    report.append(f"| {metric} | {value:.4f} | {status} |")
                report.append("")
        
        return "\n".join(report)
    
    def _generate_html_report(self) -> str:
        """Generate HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Evaluation Report - {self.model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .metric-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metric-table th {{ background-color: #f2f2f2; }}
                .pass {{ color: green; }}
                .fail {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>LLM Evaluation Report</h1>
                <p><strong>Model:</strong> {self.model_name}</p>
                <p><strong>Timestamp:</strong> {self.timestamp}</p>
                <p><strong>Risk Level:</strong> {self.config.risk_level.value.upper()}</p>
                <p><strong>Overall Score:</strong> {self.overall_score:.3f}</p>
                <p><strong>Risk Compliance:</strong> 
                   <span class="{'pass' if self.risk_compliance else 'fail'}">
                   {'✓ PASS' if self.risk_compliance else '✗ FAIL'}
                   </span>
                </p>
            </div>
        """
        
        # Add metric tables
        for category, metrics in [("Safety", self.safety_metrics), 
                                 ("Security", self.security_metrics),
                                 ("Reliability", self.reliability_metrics)]:
            if metrics:
                html += f"<h2>{category} Metrics</h2>"
                html += '<table class="metric-table">'
                html += "<tr><th>Metric</th><th>Value</th><th>Status</th></tr>"
                for metric, value in metrics.items():
                    status = self._get_metric_status(category.lower(), metric, value)
                    html += f"<tr><td>{metric}</td><td>{value:.4f}</td><td>{status}</td></tr>"
                html += "</table>"
        
        html += "</body></html>"
        return html
    
    def _get_metric_status(self, category: str, metric: str, value: float) -> str:
        """Get status indicator for a metric."""
        compliance_key = f"{category}_{metric}"
        if compliance_key in self.compliance_details:
            return "✓" if self.compliance_details[compliance_key] else "✗"
        return "?"
    
    def save_report(self, file_path: str, format: str = "text") -> None:
        """Save report to file."""
        report = self.to_report(format)
        
        with open(file_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {file_path}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "model_name": self.model_name,
            "risk_level": self.config.risk_level.value,
            "overall_score": self.overall_score,
            "risk_compliance": self.risk_compliance,
            "total_frameworks": len(self.safety_results) + len(self.security_results) + len(self.reliability_results),
            "failed_frameworks": len(self.framework_failures),
            "execution_time": self.total_execution_time,
            "metrics_count": {
                "safety": len(self.safety_metrics),
                "security": len(self.security_metrics), 
                "reliability": len(self.reliability_metrics)
            }
        }