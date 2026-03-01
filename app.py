import sys
import os

#Tell Python to look inside the 'src' folder for modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import gradio as gr
import pandas as pd
import joblib

# Load the trained model pipeline
model = joblib.load('models/churn_model.pkl')

def predict_churn(account_length, intl_plan, vmail_plan, day_mins, customer_service_calls):
    """
    Takes user input from Gradio, creates a DataFrame, and predicts churn.
    """
    # 1. Create a dictionary with the user inputs
    input_data = {
        'Account length': [account_length],
        'International plan': [intl_plan],
        'Voice mail plan': [vmail_plan],
        'Total day minutes': [day_mins],
        'Customer service calls': [customer_service_calls],
        
        # Fill the rest of the required columns with baseline/average values
        # (Notice we use the original names with spaces, our TelecomDataCleaner will fix them!)
        'State': ['KS'], 'Area code': [415], 'Number vmail messages': [0],
        'Total day calls': [100], 'Total day charge': [30.0],
        'Total eve minutes': [200.0], 'Total eve calls': [100], 'Total eve charge': [17.0],
        'Total night minutes': [200.0], 'Total night calls': [100], 'Total night charge': [9.0],
        'Total intl minutes': [10.0], 'Total intl calls': [5], 'Total intl charge': [2.7]
    }
    
    # 2. Convert to DataFrame
    df = pd.DataFrame(input_data)
    
    # 3. Predict (The Pipeline automatically cleans, scales, and predicts)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1] # Get probability of class 1 (Churn)
    
    # 4. Format Output
    if prediction == 1:
        return f"🚨 HIGH RISK of Churn! (Probability: {probability:.1%})"
    else:
        return f"✅ Low Risk (Safe). (Probability: {probability:.1%})"

# --- Build the Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📉 Telecom Customer Churn Predictor")
    gr.Markdown("Enter customer details below to predict if they are likely to cancel their service.")
    
    with gr.Row():
        with gr.Column():
            account_length = gr.Number(label="Account Length (Months)", value=12)
            day_mins = gr.Number(label="Total Day Minutes (per month)", value=150.0)
            customer_service_calls = gr.Slider(minimum=0, maximum=10, step=1, label="Customer Service Calls", value=1)
            
        with gr.Column():
            intl_plan = gr.Radio(choices=["Yes", "No"], label="International Plan", value="No")
            vmail_plan = gr.Radio(choices=["Yes", "No"], label="Voice Mail Plan", value="No")
            
    predict_btn = gr.Button("Predict Churn Risk", variant="primary")
    output = gr.Textbox(label="Prediction Result", text_align="center")
    
    predict_btn.click(
        fn=predict_churn,
        inputs=[account_length, intl_plan, vmail_plan, day_mins, customer_service_calls],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()