import sys
import os

# Tell Python to look inside the 'src' folder for modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import gradio as gr
import pandas as pd
import joblib
import plotly.graph_objects as go
from preprocessing import TelecomDataCleaner 

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
            'bar': {'color': "rgba(0,0,0,0)"}, # Hide default bar
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "rgba(46, 204, 113, 0.8)"},   # Green
                {'range': [30, 70], 'color': "rgba(241, 196, 15, 0.8)"},  # Yellow
                {'range': [70, 100], 'color': "rgba(231, 76, 60, 0.8)"}], # Red
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100}
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def predict_churn(account_length, intl_plan, vmail_plan, day_mins, customer_service_calls):
    # Create input dictionary
    input_data = {
        'Account length': [account_length],
        'International plan': [intl_plan],
        'Voice mail plan': [vmail_plan],
        'Total day minutes': [day_mins],
        'Customer service calls': [customer_service_calls],
        # Baseline fillers
        'State': ['KS'], 'Area code': [415], 'Number vmail messages': [0],
        'Total day calls': [100], 'Total day charge': [30.0],
        'Total eve minutes': [200.0], 'Total eve calls': [100], 'Total eve charge': [17.0],
        'Total night minutes': [200.0], 'Total night calls': [100], 'Total night charge': [9.0],
        'Total intl minutes': [10.0], 'Total intl calls': [5], 'Total intl charge': [2.7]
    }
    
    df = pd.DataFrame(input_data)
    
    # Predict probability
    probability = model.predict_proba(df)[0][1] 
    
    # Generate formatting
    if probability > 0.5:
        status = "🚨 HIGH RISK: Immediate Intervention Required"
    elif probability > 0.3:
        status = "⚠️ MODERATE RISK: Monitor Customer"
    else:
        status = "✅ LOW RISK: Customer is Stable"
        
    gauge_plot = create_gauge_chart(probability)
    return status, gauge_plot

# --- Build the Upgraded Gradio UI ---
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>📉 AI Customer Churn Intelligence</h1>")
    gr.Markdown("<p style='text-align: center;'>Predictive analytics dashboard powered by Logistic Regression.</p>")
    
    with gr.Tabs():
        # TAB 1: The Predictor
        with gr.TabItem("Live Prediction Engine"):
            with gr.Row():
                # Input Column
                with gr.Column(scale=1):
                    gr.Markdown("### Customer Profile")
                    account_length = gr.Number(label="Account Length (Months)", value=12)
                    day_mins = gr.Number(label="Total Day Minutes (per month)", value=150.0)
                    customer_service_calls = gr.Slider(minimum=0, maximum=10, step=1, label="Customer Service Calls", value=1)
                    intl_plan = gr.Radio(choices=["Yes", "No"], label="International Plan", value="No")
                    vmail_plan = gr.Radio(choices=["Yes", "No"], label="Voice Mail Plan", value="No")
                    predict_btn = gr.Button("Analyze Churn Risk", variant="primary")
                    
                # Output Column
                with gr.Column(scale=1):
                    gr.Markdown("### Risk Assessment")
                    status_output = gr.Textbox(label="System Recommendation", show_label=False)
                    plot_output = gr.Plot(label="Probability Analysis")
        
        # TAB 2: Model Architecture
        with gr.TabItem("System Analytics"):
            gr.Markdown("### 🧠 Under the Hood: The ML Pipeline")
            gr.Markdown("""
            This predictive engine was built strictly using Traditional Machine Learning to satisfy Milestone 1 requirements.
            
            * **Algorithm:** Logistic Regression with Scikit-Learn.
            * **Preprocessing:** Custom Pipeline integrating `StandardScaler` and `OneHotEncoder`.
            * **Key Churn Drivers Identified:** 1. Customer Service Calls (High correlation with churn)
                2. Total Day Minutes (High usage without proper plans leads to churn)
                3. International Plan (Lack of cost-effective global plans)
            """)
        
        # TAB 3: EDA & Evaluation Metrics
        with gr.TabItem("EDA & Evaluation Analytics"):
            gr.Markdown("### 📊 Exploratory Data Analysis & System Evaluation")
            gr.Markdown("Visual insights and evaluation metrics driving the predictive engine.")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Exploratory Data Analysis")
                    gr.Image("visualizations/4_Correlation_Heatmap.png", show_label=False)
                    gr.Image("visualizations/5_Service_Calls_EDA.png", show_label=False)
                
                with gr.Column():
                    gr.Markdown("#### Model Evaluation (Cost-Sensitive)")
                    gr.Image("visualizations/3_Confusion_Matrix.png", show_label=False)
                    gr.Image("visualizations/2_Risk_Distribution.png", show_label=False)
            
    # Connect logic
    predict_btn.click(
        fn=predict_churn,
        inputs=[account_length, intl_plan, vmail_plan, day_mins, customer_service_calls],
        outputs=[status_output, plot_output]
    )

if __name__ == "__main__":
    demo.launch()