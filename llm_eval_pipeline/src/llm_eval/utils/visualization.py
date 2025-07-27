"""Visualization utilities for evaluation results."""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """Utility class for creating visualizations of evaluation results."""
    
    def __init__(self):
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Install with: pip install plotly")
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas not available. Install with: pip install pandas")
    
    def create_overview_dashboard(self, results: Dict[str, Any], 
                                output_path: Optional[str] = None) -> str:
        """Create comprehensive overview dashboard."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for visualizations")
        
        # Create subplot layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Overall Scores", "Risk Compliance",
                "Safety Metrics", "Security Metrics", 
                "Reliability Metrics", "Framework Status"
            ],
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "radar"}, {"type": "radar"}],
                [{"type": "radar"}, {"type": "bar"}]
            ]
        )
        
        # Overall scores bar chart
        scores = {
            "Overall Score": results.get('overall_score', 0),
            "Safety Score": self._calculate_category_score(results.get('safety_metrics', {})),
            "Security Score": self._calculate_category_score(results.get('security_metrics', {})),
            "Reliability Score": self._calculate_category_score(results.get('reliability_metrics', {}))
        }
        
        fig.add_trace(
            go.Bar(x=list(scores.keys()), y=list(scores.values()), 
                  name="Scores", marker_color='lightblue'),
            row=1, col=1
        )
        
        # Risk compliance pie chart
        compliance_details = results.get('compliance_details', {})
        passed = sum(1 for v in compliance_details.values() if v)
        failed = len(compliance_details) - passed
        
        fig.add_trace(
            go.Pie(labels=['Passed', 'Failed'], values=[passed, failed],
                  marker_colors=['green', 'red']),
            row=1, col=2
        )
        
        # Safety metrics radar
        safety_metrics = results.get('safety_metrics', {})
        if safety_metrics:
            self._add_radar_chart(fig, safety_metrics, "Safety", row=2, col=1)
        
        # Security metrics radar  
        security_metrics = results.get('security_metrics', {})
        if security_metrics:
            self._add_radar_chart(fig, security_metrics, "Security", row=2, col=2)
        
        # Reliability metrics radar
        reliability_metrics = results.get('reliability_metrics', {})
        if reliability_metrics:
            self._add_radar_chart(fig, reliability_metrics, "Reliability", row=3, col=1)
        
        # Framework status bar chart
        framework_failures = results.get('framework_failures', [])
        safety_results = results.get('safety_results', [])
        security_results = results.get('security_results', [])
        reliability_results = results.get('reliability_results', [])
        
        total_frameworks = len(safety_results) + len(security_results) + len(reliability_results)
        successful_frameworks = total_frameworks - len(framework_failures)
        
        fig.add_trace(
            go.Bar(x=['Successful', 'Failed'], 
                  y=[successful_frameworks, len(framework_failures)],
                  marker_color=['green', 'red']),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Evaluation Dashboard - {results.get('model_name', 'Unknown Model')}",
            showlegend=False,
            height=1000
        )
        
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Dashboard saved to {output_path}")
        
        return fig.to_html()
    
    def create_safety_analysis(self, safety_results: List[Dict[str, Any]], 
                             output_path: Optional[str] = None) -> str:
        """Create detailed safety analysis visualization."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for visualizations")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Harm Scores Distribution", "Category Breakdown",
                "Jailbreak Success Rates", "Safety Timeline"
            ]
        )
        
        # Extract harm scores
        harm_scores = []
        categories = {}
        jailbreak_rates = []
        
        for result in safety_results:
            if 'harm_score' in result.get('metrics', {}):
                harm_scores.append(result['metrics']['harm_score'])
            
            if 'category_breakdown' in result.get('metadata', {}):
                for cat, score in result['metadata']['category_breakdown'].items():
                    if cat not in categories:
                        categories[cat] = []
                    categories[cat].append(score)
            
            if 'jailbreak_success_rate' in result.get('metrics', {}):
                jailbreak_rates.append(result['metrics']['jailbreak_success_rate'])
        
        # Harm scores histogram
        if harm_scores:
            fig.add_trace(
                go.Histogram(x=harm_scores, name="Harm Scores", nbinsx=20),
                row=1, col=1
            )
        
        # Category breakdown
        if categories:
            category_means = {cat: sum(scores)/len(scores) for cat, scores in categories.items()}
            fig.add_trace(
                go.Bar(x=list(category_means.keys()), y=list(category_means.values()),
                      name="Category Scores"),
                row=1, col=2
            )
        
        # Jailbreak rates
        if jailbreak_rates:
            fig.add_trace(
                go.Scatter(y=jailbreak_rates, mode='lines+markers', 
                          name="Jailbreak Rates"),
                row=2, col=1
            )
        
        # Timeline (execution times)
        execution_times = [result.get('execution_time', 0) for result in safety_results]
        if execution_times:
            fig.add_trace(
                go.Scatter(y=execution_times, mode='lines+markers',
                          name="Execution Time"),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Safety Analysis Dashboard",
            height=800
        )
        
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Safety analysis saved to {output_path}")
        
        return fig.to_html()
    
    def create_security_analysis(self, security_results: List[Dict[str, Any]], 
                               output_path: Optional[str] = None) -> str:
        """Create detailed security analysis visualization."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for visualizations")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Attack Success Rates", "Attack Types Effectiveness",
                "Defense Performance", "Security Trends"
            ]
        )
        
        # Extract security metrics
        injection_rates = []
        attack_types = {}
        defense_rates = []
        
        for result in security_results:
            metrics = result.get('metrics', {})
            
            if 'injection_success_rate' in metrics:
                injection_rates.append(metrics['injection_success_rate'])
            
            if 'attack_types' in metrics:
                for attack_type, success_rate in metrics['attack_types'].items():
                    if attack_type not in attack_types:
                        attack_types[attack_type] = []
                    attack_types[attack_type].append(success_rate)
            
            if 'defense_rate' in metrics:
                defense_rates.append(metrics['defense_rate'])
        
        # Attack success rates histogram
        if injection_rates:
            fig.add_trace(
                go.Histogram(x=injection_rates, name="Injection Rates", nbinsx=15),
                row=1, col=1
            )
        
        # Attack types effectiveness
        if attack_types:
            attack_means = {attack: sum(rates)/len(rates) for attack, rates in attack_types.items()}
            fig.add_trace(
                go.Bar(x=list(attack_means.keys()), y=list(attack_means.values()),
                      name="Attack Effectiveness"),
                row=1, col=2
            )
        
        # Defense performance
        if defense_rates:
            fig.add_trace(
                go.Scatter(y=defense_rates, mode='lines+markers',
                          name="Defense Rates"),
                row=2, col=1
            )
        
        # Security trends (combined metrics)
        if injection_rates and defense_rates:
            combined_security = [1 - inj for inj in injection_rates]  # Invert injection rate
            fig.add_trace(
                go.Scatter(y=combined_security, mode='lines+markers',
                          name="Security Score"),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Security Analysis Dashboard", 
            height=800
        )
        
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Security analysis saved to {output_path}")
        
        return fig.to_html()
    
    def create_reliability_analysis(self, reliability_results: List[Dict[str, Any]], 
                                  output_path: Optional[str] = None) -> str:
        """Create detailed reliability analysis visualization."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for visualizations")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Consistency Scores", "Robustness Metrics",
                "Degradation Analysis", "Reliability Trends"
            ]
        )
        
        # Extract reliability metrics
        consistency_scores = []
        robustness_scores = []
        degradation_rates = []
        
        for result in reliability_results:
            metrics = result.get('metrics', {})
            
            if 'overall_consistency' in metrics:
                consistency_scores.append(metrics['overall_consistency'])
            
            if 'overall_robustness_pdr' in metrics:
                robustness_scores.append(1.0 - metrics['overall_robustness_pdr'])
            
            if 'evolution_robustness_pdr' in metrics:
                degradation_rates.append(metrics['evolution_robustness_pdr'])
        
        # Consistency scores box plot
        if consistency_scores:
            fig.add_trace(
                go.Box(y=consistency_scores, name="Consistency"),
                row=1, col=1
            )
        
        # Robustness metrics
        if robustness_scores:
            fig.add_trace(
                go.Histogram(x=robustness_scores, name="Robustness", nbinsx=15),
                row=1, col=2
            )
        
        # Degradation analysis
        if degradation_rates:
            fig.add_trace(
                go.Scatter(y=degradation_rates, mode='lines+markers',
                          name="Degradation Rate"),
                row=2, col=1
            )
        
        # Reliability trends
        if consistency_scores and robustness_scores:
            min_len = min(len(consistency_scores), len(robustness_scores))
            combined_reliability = [
                (c + r) / 2 for c, r in zip(consistency_scores[:min_len], robustness_scores[:min_len])
            ]
            fig.add_trace(
                go.Scatter(y=combined_reliability, mode='lines+markers',
                          name="Combined Reliability"),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Reliability Analysis Dashboard",
            height=800
        )
        
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Reliability analysis saved to {output_path}")
        
        return fig.to_html()
    
    def create_comparison_chart(self, results_list: List[Dict[str, Any]], 
                              output_path: Optional[str] = None) -> str:
        """Create comparison chart for multiple model results."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for visualizations")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Overall Scores Comparison", "Safety Comparison",
                "Security Comparison", "Reliability Comparison"
            ]
        )
        
        model_names = [result.get('model_name', f'Model_{i}') for i, result in enumerate(results_list)]
        
        # Overall scores comparison
        overall_scores = [result.get('overall_score', 0) for result in results_list]
        fig.add_trace(
            go.Bar(x=model_names, y=overall_scores, name="Overall Score"),
            row=1, col=1
        )
        
        # Safety scores comparison
        safety_scores = [
            self._calculate_category_score(result.get('safety_metrics', {}))
            for result in results_list
        ]
        fig.add_trace(
            go.Bar(x=model_names, y=safety_scores, name="Safety Score", marker_color='green'),
            row=1, col=2
        )
        
        # Security scores comparison
        security_scores = [
            self._calculate_category_score(result.get('security_metrics', {}))
            for result in results_list
        ]
        fig.add_trace(
            go.Bar(x=model_names, y=security_scores, name="Security Score", marker_color='red'),
            row=2, col=1
        )
        
        # Reliability scores comparison
        reliability_scores = [
            self._calculate_category_score(result.get('reliability_metrics', {}))
            for result in results_list
        ]
        fig.add_trace(
            go.Bar(x=model_names, y=reliability_scores, name="Reliability Score", marker_color='blue'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Model Comparison Dashboard",
            height=800,
            showlegend=False
        )
        
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Comparison chart saved to {output_path}")
        
        return fig.to_html()
    
    def _add_radar_chart(self, fig, metrics: Dict[str, float], category: str, row: int, col: int):
        """Add radar chart to subplot."""
        # Normalize metrics to 0-1 range for radar chart
        normalized_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # Assume most metrics are already 0-1, but handle edge cases
                normalized_value = max(0, min(1, value))
                normalized_metrics[key] = normalized_value
        
        if normalized_metrics:
            categories = list(normalized_metrics.keys())
            values = list(normalized_metrics.values())
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=category
                ),
                row=row, col=col
            )
    
    def _calculate_category_score(self, metrics: Dict[str, float]) -> float:
        """Calculate normalized category score."""
        if not metrics:
            return 0.0
        
        # Simple average of metrics (assuming they're already normalized)
        valid_metrics = [v for v in metrics.values() if isinstance(v, (int, float))]
        if not valid_metrics:
            return 0.0
        
        return sum(valid_metrics) / len(valid_metrics)
    
    def export_metrics_table(self, results: Dict[str, Any], 
                           output_path: str, format: str = "html"):
        """Export metrics as formatted table."""
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas not available for table export")
            return
        
        # Collect all metrics
        all_metrics = {}
        all_metrics.update(results.get('safety_metrics', {}))
        all_metrics.update(results.get('security_metrics', {}))
        all_metrics.update(results.get('reliability_metrics', {}))
        
        # Create DataFrame
        df = pd.DataFrame([
            {"Metric": key, "Value": value, "Category": self._categorize_metric(key)}
            for key, value in all_metrics.items()
        ])
        
        if format.lower() == "html":
            df.to_html(output_path, index=False)
        elif format.lower() == "csv":
            df.to_csv(output_path, index=False)
        elif format.lower() == "excel":
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Metrics table exported to {output_path}")
    
    def _categorize_metric(self, metric_name: str) -> str:
        """Categorize metric based on name."""
        metric_lower = metric_name.lower()
        
        if any(term in metric_lower for term in ['harm', 'safety', 'jailbreak', 'constitutional']):
            return "Safety"
        elif any(term in metric_lower for term in ['injection', 'attack', 'security', 'defense']):
            return "Security"
        elif any(term in metric_lower for term in ['consistency', 'robustness', 'reliability', 'degradation']):
            return "Reliability"
        else:
            return "Other"