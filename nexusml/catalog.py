"""Vocabulary that keeps the BUSINESS question separate from the ANALYTICAL problem.

analysis_type = the business problem   (sales_performance, churn, profitability, ...)
problem_type  = the analytical method  (classification, regression, time_series, clustering, descriptive)
Example: analysis_type="sales_performance", problem_type="time_series".
"""
from typing import Dict, List, Optional

ANALYSIS_TYPES: Dict[str, str] = {
    "sales_performance": "Revenue / units sold, growth, seasonality, drivers of sales",
    "profitability": "Margins, cost-to-serve, what makes some segments more profitable",
    "customer_analysis": "Customer behaviour, value, segments, retention levers",
    "churn": "Who leaves, why, and who is at risk",
    "marketing_performance": "Campaign / channel effectiveness, conversion, ROI",
    "inventory": "Stock levels, turnover, stock-outs, demand vs supply",
    "operational_efficiency": "Throughput, cycle time, defects, utilisation, delays",
    "financial_variance": "Actual vs plan/budget/prior period and what drives the gap",
    "forecasting": "Projecting a business measure forward",
    "anomaly_investigation": "Explaining unusual spikes, drops or outliers",
    "general_exploration": "No specific business problem stated",
}
PROBLEM_TYPES: List[str] = ["classification", "regression", "time_series", "clustering", "descriptive"]
SUPERVISED = ("classification", "regression", "time_series")
ALIASES = {"forecasting": "time_series", "forecast": "time_series", "timeseries": "time_series",
           "segmentation": "clustering", "cluster": "clustering", "eda": "descriptive"}

def norm_problem_type(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    p = str(p).strip().lower().replace(" ", "_").replace("-", "_")
    p = ALIASES.get(p, p)
    return p if p in PROBLEM_TYPES else None


def norm_analysis_type(a: Optional[str]) -> Optional[str]:
    if not a:
        return None
    a = str(a).strip().lower().replace(" ", "_").replace("-", "_")
    return a if a in ANALYSIS_TYPES else None
