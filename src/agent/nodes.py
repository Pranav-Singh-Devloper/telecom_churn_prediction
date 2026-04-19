import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# --- CHANGED TO ABSOLUTE IMPORTS ---
from src.agent.state import RetentionAgentState
from src.agent.retriever import RetentionKnowledgeBase
# -----------------------------------

# Initialize the RAG Database
knowledge_base = RetentionKnowledgeBase()

# Configure the LLM to use Groq's Free API (Llama 3)
llm = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), 
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.1-8b-instant", 
    temperature=0.2 # Low temperature to reduce hallucinations
)

def analyze_risk(state: RetentionAgentState) -> RetentionAgentState:
    """Node 1: Evaluates if the churn risk warrants intervention."""
    print(f"🤖 [Node: analyze_risk] Evaluating ML prediction: {state['churn_probability']:.2f}")
    
    prob = state["churn_probability"]
    
    if prob >= 0.70:
        state["risk_level"] = "Critical"
    elif prob >= 0.30: # Our custom threshold
        state["risk_level"] = "High"
    else:
        state["risk_level"] = "Low"
        
    return state

def retrieve_strategy(state: RetentionAgentState) -> RetentionAgentState:
    """Node 2: Fetches internal SOPs from the Vector DB based on the primary driver."""
    print("🤖 [Node: retrieve_strategy] Querying Vector DB for SOPs...")
    
    driver = state["primary_churn_driver"]
    query = f"Customer is churning due to: {driver}"
    
    # Retrieve top strategy from ChromaDB
    strategy = knowledge_base.retrieve_strategy(query)
    state["retrieved_strategies"] = [strategy]
    
    return state

def draft_intervention(state: RetentionAgentState) -> RetentionAgentState:
    """Node 3: Prompts the LLM to write the final structured report."""
    print("🤖 [Node: draft_intervention] Generating structured intervention plan...")
    
    # Strictly matching the rubric's "Structured Output" requirements
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Telecom Retention Strategist. 
        Your job is to generate a structured intervention plan for a customer at risk of churning.
        You MUST adhere strictly to the provided internal SOPs and not hallucinate discounts.
        
        You MUST output your response as a valid JSON object with EXACTLY these four keys:
        - "risk_summary": A brief string summarizing the risk.
        - "recommendations": A list of specific actionable recommendation strings.
        - "sources": A string citing the applied SOP.
        - "disclaimer": A standard disclaimer string.
        
        Do not output any markdown code blocks like ```json, just the raw JSON object.
        """),
        ("user", """
        Customer Data: {customer_data}
        ML Risk Score: {risk_level} ({churn_probability})
        Primary Driver of Churn: {primary_churn_driver}
        
        Internal SOP to apply: {retrieved_strategies}
        """)
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "customer_data": state["customer_data"],
        "risk_level": state["risk_level"],
        "churn_probability": round(state["churn_probability"], 2),
        "primary_churn_driver": state["primary_churn_driver"],
        "retrieved_strategies": state["retrieved_strategies"][0]
    })
    
    state["final_retention_plan"] = response.content
    return state