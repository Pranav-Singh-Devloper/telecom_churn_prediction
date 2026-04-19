from langgraph.graph import StateGraph, END

# --- CHANGED TO ABSOLUTE IMPORTS ---
from src.agent.state import RetentionAgentState
from src.agent.nodes import analyze_risk, retrieve_strategy, draft_intervention
# -----------------------------------

def route_risk(state: RetentionAgentState):
    """Conditional routing logic: Only retrieve and draft if the risk is High/Critical."""
    print(f"🛤️ [Route] Risk Level: {state['risk_level']} -> {'Proceeding to Strategy' if state['risk_level'] in ['High', 'Critical'] else 'Ending Workflow'}")
    if state["risk_level"] in ["High", "Critical"]:
        return "retrieve_strategy"
    return END

# 1. Initialize the Graph with our Memory State
workflow = StateGraph(RetentionAgentState)

# 2. Add the Nodes (The "Doers")
workflow.add_node("analyze_risk", analyze_risk)
workflow.add_node("retrieve_strategy", retrieve_strategy)
workflow.add_node("draft_intervention", draft_intervention)

# 3. Define the Flow (The "Edges")
workflow.set_entry_point("analyze_risk")

# Add the conditional branching
workflow.add_conditional_edges(
    "analyze_risk",
    route_risk,
    {
        "retrieve_strategy": "retrieve_strategy",
        END: END
    }
)

# Connect the rest of the flow
workflow.add_edge("retrieve_strategy", "draft_intervention")
workflow.add_edge("draft_intervention", END)

# 4. Compile the fully operational Agent
retention_agent = workflow.compile()

# --- For Testing Locally ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Simulate data coming from your ML Pipeline
    test_state = {
        "customer_data": {"Account_length": 120, "Total_day_minutes": 350},
        "churn_probability": 0.85,
        "risk_level": "",
        "primary_churn_driver": "Customer_service_calls",
        "retrieved_strategies": [],
        "final_retention_plan": ""
    }
    
    print("\n🚀 Starting Agentic Workflow...\n")
    final_state = retention_agent.invoke(test_state)
    
    print("\n✅ Final Agent Output:\n")
    print(final_state.get("final_retention_plan", "No plan generated (Customer is low risk)."))