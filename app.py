import sys
import os

# Tell Python to look inside the 'src' folder for modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import gradio as gr
import pandas as pd
import joblib
import plotly.graph_objects as go
from preprocessing import TelecomDataCleaner 

# --- Import the Phase 2 LangGraph Agent ---
from src.agent.graph import retention_agent

# Load the trained model pipeline
model = joblib.load('models/churn_model.pkl')

def create_gauge_chart(probability):
    """Creates a sleek Plotly gauge chart for churn risk."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Risk Probability", 'font': {'size': 24}},
        number = {'suffix': "%", 'font': {'size': 40}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "rgba(46, 204, 113, 0.8)"},
                {'range': [30, 70], 'color': "rgba(241, 196, 15, 0.8)"},
                {'range': [70, 100], 'color': "rgba(231, 76, 60, 0.8)"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100}
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def get_xai_plot():
    """Extracts coefficients from the trained model and returns a DARK-THEME optimized Plotly XAI chart."""
    
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    features = preprocessor.get_feature_names_out()
    coefs = classifier.coef_[0]
    
    clean_features = [f.replace('num__', '').replace('cat__', '').replace('_', ' ') for f in features]
    
    importance_df = pd.DataFrame({'Feature': clean_features, 'Weight': coefs})
    importance_df['Abs_Weight'] = importance_df['Weight'].abs()
    importance_df = importance_df.sort_values(by='Abs_Weight', ascending=False).head(12)
    importance_df = importance_df.sort_values(by='Weight', ascending=True)
    
    colors = [
        'rgba(231, 76, 60, 0.9)' if w > 0 
        else 'rgba(46, 204, 113, 0.9)' 
        for w in importance_df['Weight']
    ]
    
    fig = go.Figure(go.Bar(
        x=importance_df['Weight'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color=colors,
        text=importance_df['Weight'].round(3),
        textposition='outside',
        textfont=dict(color='white', size=13),
        hovertemplate="<b>%{y}</b><br>Impact: %{x}<extra></extra>"
    ))
    
    fig.update_layout(
        title={
            'text': 'Global Feature Importance Engine (XAI)',
            'font': {'size': 22, 'color': '#e5e7eb'},
            'x': 0.5
        },
        
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),

        # 🔥 DARK MODE FIXES
        plot_bgcolor='rgba(0,0,0,0)',   # transparent
        paper_bgcolor='rgba(0,0,0,0)',

        font=dict(color='#e5e7eb'),

        xaxis=dict(
            title='Coefficient Weight (Impact on Churn)',
            title_font=dict(color='#9ca3af'),
            tickfont=dict(color='#d1d5db'),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.08)',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.25)',
            zerolinewidth=1
        ),

        yaxis=dict(
            title='',
            tickfont=dict(color='#d1d5db')
        )
    )
    
    return fig

def prepare_dataframe(account_length, intl_plan, vmail_plan, day_mins, customer_service_calls):
    """Helper function to format Gradio inputs for the ML Pipeline."""
    return pd.DataFrame({
        'Account length': [account_length],
        'International plan': [intl_plan],
        'Voice mail plan': [vmail_plan],
        'Total day minutes': [day_mins],
        'Customer service calls': [customer_service_calls],
        'State': ['KS'], 'Area code': [415], 'Number vmail messages': [0],
        'Total day calls': [100], 'Total day charge': [30.0],
        'Total eve minutes': [200.0], 'Total eve calls': [100], 'Total eve charge': [17.0],
        'Total night minutes': [200.0], 'Total night calls': [100], 'Total night charge': [9.0],
        'Total intl minutes': [10.0], 'Total intl calls': [5], 'Total intl charge': [2.7]
    })

def predict_churn(account_length, intl_plan, vmail_plan, day_mins, customer_service_calls):
    """Phase 1: Generates the ML Prediction and UI Gauge."""
    df = prepare_dataframe(account_length, intl_plan, vmail_plan, day_mins, customer_service_calls)
    probability = model.predict_proba(df)[0][1] 
    
    if probability >= 0.7:
        status = "🚨 CRITICAL RISK: Immediate Intervention Required"
    elif probability >= 0.3:
        status = "⚠️ HIGH RISK: Agentic Strategy Recommended"
    else:
        status = "✅ LOW RISK: Customer is Stable"
        
    gauge_plot = create_gauge_chart(probability)
    return status, gauge_plot, True

# def generate_agentic_strategy(account_length, intl_plan, vmail_plan, day_mins, customer_service_calls, prediction_run):
    """Phase 2: Bridges the Gradio UI with the LangGraph AI Agent."""
    
    if not prediction_run:
        return """
# For the light cards, force dark text
<div style="background-color: #ffffff; border: 1px solid #e1e4e8; padding: 24px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); color: #111827;">
    <h3 style="color: #111827; margin-top: 0; margin-bottom: 16px; font-size: 1.3em; border-bottom: 2px solid #f1f3f5; padding-bottom: 10px;">
        💡 Actionable Recommendations
    </h3>
    <ul style="color: #111827; line-height: 1.6; font-size: 1.05em; padding-left: 20px; margin-bottom: 0;">
        {recs_html}
    </ul>
</div>
"""

    # 1. Get real prediction from ML model
    df = prepare_dataframe(account_length, intl_plan, vmail_plan, day_mins, customer_service_calls)
    probability = model.predict_proba(df)[0][1]
    
    # 2. Dynamically determine the XAI driver based on inputs
    if customer_service_calls >= 4:
        driver = "Customer_service_calls"
    elif intl_plan == "No" and day_mins > 200:
        driver = "International_plan_No"
    elif day_mins < 100:
        driver = "Low_Usage"
    else:
        driver = "Day_Cost_Per_Min"

    # 3. Setup the LangGraph Memory State
    state = {
        "customer_data": {
            "Account_length": account_length,
            "Total_day_minutes": day_mins,
            "Customer_service_calls": customer_service_calls,
            "International_plan": intl_plan
        },
        "churn_probability": float(probability),
        "risk_level": "", # Agent Node 1 will evaluate this
        "primary_churn_driver": driver,
        "retrieved_strategies": [],
        "final_retention_plan": ""
    }
    
    # 4. Invoke the LangGraph Workflow
    final_state = retention_agent.invoke(state)
    
    # 5. Extract the LLM's structured report
    raw_report = final_state.get("final_retention_plan", "").strip()
    
    if not raw_report:
        return """
<div style="background-color: #d4edda; border-left: 6px solid #28a745; padding: 18px 20px; border-radius: 8px; color: #155724; font-family: sans-serif;">
    <h3 style="margin-top: 0; margin-bottom: 12px; font-size: 1.4em;">✅ Customer Status: Stable (Low Risk)</h3>
    <p style="margin-bottom: 10px; font-size: 1.05em;">The Machine Learning model indicates a <b>Low Probability of Churn</b>.</p>
    <p style="margin-bottom: 10px; font-size: 1.05em;">Our automated retention agent has analyzed the profile and concluded that <b>no active intervention is required</b> at this time.</p>
    <p style="margin: 0; font-style: italic; color: #2e7d32;">Continue monitoring through standard lifecycle engagement.</p>
</div>
"""

    risk_level = final_state.get("risk_level", "High")
    risk_color = "#e74c3c" if risk_level == "Critical" else "#e67e22"
    risk_icon = "🚨" if risk_level == "Critical" else "⚠️"
    
    import json
    
    # Try parsing JSON
    try:
        # Clean up the output in case the LLM ignored instructions and used markdown
        if raw_report.startswith("```json"):
            raw_report = raw_report.replace("```json", "", 1).strip()
        if raw_report.endswith("```"):
            raw_report = raw_report[:-3].strip()

        data = json.loads(raw_report)
        summary = data.get("risk_summary", "")
        recs = data.get("recommendations", [])
        if isinstance(recs, str):
            recs_html = f"<li>{recs}</li>"
        else:
            recs_html = "".join([f"<li style='margin-bottom: 8px;'>{r}</li>" for r in recs])
        sources = data.get("sources", "")
        disclaimer = data.get("disclaimer", "")

        return f"""
<div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px;">
    
    <!-- Risk Banner -->
    <div style="background-color:{risk_color};border-left:6px solid {risk_color};padding:18px 20px;border-radius:10px;margin-bottom:24px;color:white;"><h3 style="color:white;margin:0 0 12px 0;font-size:1.4em;">{risk_icon} {risk_level.upper()} RISK PROFILE DETECTED</h3></div>
    <!-- Recommendations Card -->
    <div style="background-color: #ffffff; border: 1px solid #e1e4e8; padding: 24px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
        <h3 style="color: #2c3e50; margin-top: 0; margin-bottom: 16px; font-size: 1.3em; border-bottom: 2px solid #f1f3f5; padding-bottom: 10px;">💡 Actionable Recommendations</h3>
        <ul style="color: #34495e; line-height: 1.6; font-size: 1.05em; padding-left: 20px; margin-bottom: 0;">
            {recs_html}
        </ul>
    </div>
    
    <!-- Footer Cards -->
    <div style="display: flex; gap: 20px;">
        <div style="flex: 1; background-color: #f1f8ff; border: 1px solid #c8e1ff; padding: 16px; border-radius: 10px; color: #111827;">
            <b style="font-size: 1.1em; display: block; margin-bottom: 8px;">📚 Applied SOPs</b>
            <span style="font-size: 0.95em; line-height: 1.5; color: #111827;">{sources}</span>
        </div>
        <div style="flex: 1; background-color: #fff8f8; border: 1px solid #ffdce0; padding: 16px; border-radius: 10px; color: #111827;">
            <b style="font-size: 1.1em; display: block; margin-bottom: 8px;">⚠️ Disclaimer</b>
            <span style="font-size: 0.9em; line-height: 1.4; color: #111827;">{disclaimer}</span>
        </div>
    </div>
    
</div>
"""
    except json.JSONDecodeError:
        # Fallback if the LLM didn't return valid JSON
        return f"""
<div style="background-color: {risk_color}11; border-left: 6px solid {risk_color}; padding: 18px 20px; border-radius: 8px; margin-bottom: 24px;">
    <h3 style="color: {risk_color}; margin-top: 0; margin-bottom: 12px; font-size: 1.4em;">{risk_icon} {risk_level.upper()} RISK PROFILE DETECTED</h3>
</div>

{raw_report}
"""

def generate_agentic_strategy(account_length, intl_plan, vmail_plan, day_mins, customer_service_calls, prediction_run):
    """Phase 2: Bridges the Gradio UI with the LangGraph AI Agent."""

    if not prediction_run:
        return """
<div style="background-color: #fff3cd; border-left: 6px solid #ffc107; padding: 15px; border-radius: 8px; color: #111827; font-family: sans-serif;">
    <h3 style="margin-top: 0; margin-bottom: 10px; color: #111827;">⚠️ Action Required: Run Prediction First</h3>
    <p style="margin: 0; color: #111827;">
        Please navigate back to the <b>Live Prediction Engine</b> tab and click <b>"Analyze Churn Risk"</b> to generate the initial ML prediction before requesting an Agentic Strategy.
        The AI Agent requires this baseline risk assessment to formulate a targeted intervention plan.
    </p>
</div>
"""

    # 1. Get real prediction from ML model
    df = prepare_dataframe(account_length, intl_plan, vmail_plan, day_mins, customer_service_calls)
    probability = model.predict_proba(df)[0][1]

    # 2. Dynamically determine the XAI driver based on inputs
    if customer_service_calls >= 4:
        driver = "Customer_service_calls"
    elif intl_plan == "No" and day_mins > 200:
        driver = "International_plan_No"
    elif day_mins < 100:
        driver = "Low_Usage"
    else:
        driver = "Day_Cost_Per_Min"

    # 3. Setup the LangGraph Memory State
    state = {
        "customer_data": {
            "Account_length": account_length,
            "Total_day_minutes": day_mins,
            "Customer_service_calls": customer_service_calls,
            "International_plan": intl_plan
        },
        "churn_probability": float(probability),
        "risk_level": "",  # Agent Node 1 will evaluate this
        "primary_churn_driver": driver,
        "retrieved_strategies": [],
        "final_retention_plan": ""
    }

    # 4. Invoke the LangGraph Workflow
    final_state = retention_agent.invoke(state)

    # 5. Extract the LLM's structured report
    raw_report = final_state.get("final_retention_plan", "").strip()

    if not raw_report:
        return """
<div style="background-color: #d4edda; border-left: 6px solid #28a745; padding: 18px 20px; border-radius: 8px; color: #111827; font-family: sans-serif;">
    <h3 style="margin-top: 0; margin-bottom: 12px; font-size: 1.4em; color: #111827;">✅ Customer Status: Stable (Low Risk)</h3>
    <p style="margin-bottom: 10px; font-size: 1.05em; color: #111827;">The Machine Learning model indicates a <b>Low Probability of Churn</b>.</p>
    <p style="margin-bottom: 10px; font-size: 1.05em; color: #111827;">Our automated retention agent has analyzed the profile and concluded that <b>no active intervention is required</b> at this time.</p>
    <p style="margin: 0; font-style: italic; color: #1b5e20;">Continue monitoring through standard lifecycle engagement.</p>
</div>
"""

    risk_level = final_state.get("risk_level", "High")
    risk_color = "#e74c3c" if risk_level == "Critical" else "#e67e22"
    risk_icon = "🚨" if risk_level == "Critical" else "⚠️"

    import json

    try:
        if raw_report.startswith("```json"):
            raw_report = raw_report.replace("```json", "", 1).strip()
        if raw_report.endswith("```"):
            raw_report = raw_report[:-3].strip()

        data = json.loads(raw_report)
        summary = data.get("risk_summary", "")
        recs = data.get("recommendations", [])
        if isinstance(recs, str):
            recs_html = f"<li style='margin-bottom: 8px; color: #111827;'>{recs}</li>"
        else:
            recs_html = "".join([f"<li style='margin-bottom: 8px; color: #111827;'>{r}</li>" for r in recs])

        sources = data.get("sources", "")
        disclaimer = data.get("disclaimer", "")

        return f"""
<div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 900px; color: #111827;">

    <!-- Risk Banner -->
    <div style="background-color: {risk_color}11; border-left: 6px solid {risk_color}; padding: 18px 20px; border-radius: 10px; margin-bottom: 24px;">
        <h3 style="color: {risk_color}; margin-top: 0; margin-bottom: 12px; font-size: 1.4em;">{risk_icon} {risk_level.upper()} RISK PROFILE DETECTED</h3>
        <p style="margin: 0; color: #111827; font-size: 1.05em; line-height: 1.5;">
            <b>Summary:</b> {summary}
        </p>
    </div>

    <!-- Recommendations Card -->
    <div style="background-color: #ffffff; border: 1px solid #e5e7eb; padding: 24px; border-radius: 14px; margin-bottom: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); color: #111827;">
        <h3 style="color: #111827; margin-top: 0; margin-bottom: 16px; font-size: 1.3em; border-bottom: 2px solid #f1f3f5; padding-bottom: 10px;">
            💡 Actionable Recommendations
        </h3>
        <ul style="color: #111827; line-height: 1.6; font-size: 1.05em; padding-left: 20px; margin-bottom: 0;">
            {recs_html}
        </ul>
    </div>

    <!-- Footer Cards -->
    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 280px; background-color: #f1f8ff; border: 1px solid #c8e1ff; padding: 16px; border-radius: 12px; color: #111827;">
            <b style="font-size: 1.1em; display: block; margin-bottom: 8px; color: #111827;">📚 Applied SOPs</b>
            <span style="font-size: 0.95em; line-height: 1.5; color: #111827;">{sources}</span>
        </div>
        <div style="flex: 1; min-width: 280px; background-color: #fff8f8; border: 1px solid #ffdce0; padding: 16px; border-radius: 12px; color: #111827;">
            <b style="font-size: 1.1em; display: block; margin-bottom: 8px; color: #111827;">⚠️ Disclaimer</b>
            <span style="font-size: 0.9em; line-height: 1.4; color: #111827;">{disclaimer}</span>
        </div>
    </div>

</div>
"""
    except json.JSONDecodeError:
        return f"""
<div style="background-color: {risk_color}11; border-left: 6px solid {risk_color}; padding: 18px 20px; border-radius: 10px; margin-bottom: 24px; color: #111827;">
    <h3 style="color: {risk_color}; margin-top: 0; margin-bottom: 12px; font-size: 1.4em;">
        {risk_icon} {risk_level.upper()} RISK PROFILE DETECTED
    </h3>
</div>

<div style="color: #111827;">
    {raw_report}
</div>
"""

# --- Build the Upgraded Gradio UI ---
with gr.Blocks(theme=gr.themes.Base()) as demo:
    prediction_run_state = gr.State(False)
    gr.Markdown("<h1 style='text-align: center;'>📉 AI Customer Churn Intelligence</h1>")
    gr.Markdown("<p style='text-align: center;'>From Predictive Analytics to Agentic Intervention.</p>")
    
    with gr.Tabs():
        # TAB 1: The Predictor
        with gr.TabItem("Live Prediction Engine"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Customer Profile")
                    account_length = gr.Number(label="Account Length (Months)", value=12)
                    day_mins = gr.Number(label="Total Day Minutes (per month)", value=150.0)
                    customer_service_calls = gr.Slider(minimum=0, maximum=10, step=1, label="Customer Service Calls", value=1)
                    intl_plan = gr.Radio(choices=["Yes", "No"], label="International Plan", value="No")
                    vmail_plan = gr.Radio(choices=["Yes", "No"], label="Voice Mail Plan", value="No")
                    predict_btn = gr.Button("Analyze Churn Risk", variant="primary")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Risk Assessment")
                    status_output = gr.Textbox(label="System Recommendation", show_label=False)
                    plot_output = gr.Plot(label="Probability Analysis")
        
        # TAB 2: Phase 2 AI Agent (NEW)
        with gr.TabItem("Agentic Strategist (Milestone 2)"):
            gr.Markdown("### 🤖 Autonomous Retention Agent (LangGraph + RAG)")
            gr.Markdown("This system analyzes the ML risk score, queries the ChromaDB vector store for SOPs, and uses Groq (Llama-3.1) to generate a structured retention plan.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Generate Intervention")
                    gr.Markdown("*(Uses the inputs from the Customer Profile on the Live Prediction tab).*")
                    generate_btn = gr.Button("🧠 Generate Agentic Strategy", variant="primary")
                    
                with gr.Column(scale=2):
                    gr.Markdown("#### AI-Generated Retention Plan")
                    agent_output = gr.HTML(value="<div style='color: gray; font-style: italic;'>Waiting for execution...</div>", label="Agent Output")
                    
        # TAB 3: System Analytics (Interactive XAI)
        with gr.TabItem("System Analytics (XAI)"):
            gr.Markdown("### 🧠 Transparent AI: How the Engine Thinks")
            gr.Markdown("""
            Welcome to the **Explainable AI (XAI)** dashboard. Instead of a black box, you can interact with the actual neural weights (coefficients) of the deployed Logistic Regression model. 
            * **Red Bars** push the customer *towards* churn (High Risk).
            * **Green Bars** anchor the customer *away* from churn (Low Risk).
            Hover over the bars to see exact values.
            """)
            
            with gr.Row():
                xai_plot = gr.Plot(value=get_xai_plot())
                
            with gr.Accordion("🛠️ Advanced Pipeline Architecture", open=False):
                gr.HTML("""
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        text-align: center;
                        font-family: 'Courier New', Courier, monospace;
                        padding: 20px;
                        background: #ffffff;
                        border-radius: 10px;
                        border: 1px solid #e5e7eb;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                        color: #111827;
                    ">

                        <div style="flex: 1; padding: 10px; border-right: 2px dashed #d1d5db;">
                            <span style="font-size: 2em;">📥</span><br>
                            <b style="color:#111827;">1. Data Ingestion</b><br>
                            <span style="font-size: 0.85em; color: #374151;">Raw Telecom Data</span>
                        </div>

                        <div style="flex: 1; padding: 10px; border-right: 2px dashed #d1d5db;">
                            <span style="font-size: 2em;">⚙️</span><br>
                            <b style="color:#111827;">2. Preprocessing</b><br>
                            <span style="font-size: 0.85em; color: #374151;">StandardScaler & OneHot</span>
                        </div>

                        <div style="flex: 1; padding: 10px; border-right: 2px dashed #d1d5db;">
                            <span style="font-size: 2em;">⚖️</span><br>
                            <b style="color:#111827;">3. Balancing</b><br>
                            <span style="font-size: 0.85em; color: #374151;">SMOTE Synthetic Data</span>
                        </div>

                        <div style="flex: 1; padding: 10px;">
                            <span style="font-size: 2em;">🎯</span><br>
                            <b style="color:#111827;">4. Optimization</b><br>
                            <span style="font-size: 0.85em; color: #374151;">Cost-Sensitive Threshold (0.3)</span>
                        </div>

                    </div>
                    """)
        # TAB 4: EDA & Evaluation
        with gr.TabItem("EDA & Evaluation Analytics"):
            gr.Markdown("### 📊 Exploratory Data Analysis & System Evaluation")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Exploratory Data Analysis")
                    gr.Image("visualizations/4_Correlation_Heatmap.png", show_label=False)
                    gr.Image("visualizations/5_Service_Calls_EDA.png", show_label=False)
                with gr.Column():
                    gr.Markdown("#### Model Evaluation (Cost-Sensitive)")
                    gr.Image("visualizations/3_Confusion_Matrix.png", show_label=False)
                    gr.Image("visualizations/2_Risk_Distribution.png", show_label=False)
            
    # --- Connect UI Logic ---
    
    # Trigger Phase 1: ML Predictor
    predict_btn.click(
        fn=predict_churn,
        inputs=[account_length, intl_plan, vmail_plan, day_mins, customer_service_calls],
        outputs=[status_output, plot_output, prediction_run_state]
    )
    
    # Trigger Phase 2: AI Agent
    generate_btn.click(
        fn=generate_agentic_strategy,
        inputs=[account_length, intl_plan, vmail_plan, day_mins, customer_service_calls, prediction_run_state],
        outputs=[agent_output]
    )

if __name__ == "__main__":
    demo.launch()