from typing import TypedDict, List, Dict, Any

class RetentionAgentState(TypedDict):
    """
    Represents the state of the customer retention agent.
    """
    customer_data: Dict[str, Any]
    churn_probability: float
    risk_level: str
    primary_churn_driver: str
    retrieved_strategies: List[str]
    final_retention_plan: str
