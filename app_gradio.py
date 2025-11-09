"""
Simple Gradio GUI for E-commerce Customer Spending Prediction
Run with: python app_gradio.py
"""

import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Load data and train model
def load_data_and_train():
    """Load data and train the model"""
    base_dir = Path(__file__).resolve().parent
    for _p in [base_dir / 'Ecommerce Customers', base_dir / 'Ecommerce Customers.csv']:
        if _p.exists():
            customers = pd.read_csv(_p)
            break
    else:
        raise FileNotFoundError('Dataset not found next to script')
    
    X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
    y = customers['Yearly Amount Spent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    
    return lm, customers

# Load model
lm, customers = load_data_and_train()

# Prediction function
def predict_spending(avg_session, time_app, time_website, membership):
    """Predict yearly spending based on customer features"""
    input_data = np.array([[avg_session, time_app, time_website, membership]])
    prediction = lm.predict(input_data)[0]
    
    # Get feature importance
    features = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
    coefficients = lm.coef_
    
    impact_text = "\n".join([
        f"• {feat}: ${coef:.2f} per unit"
        for feat, coef in zip(features, coefficients)
    ])
    
    return f"${prediction:,.2f}", impact_text

# Create Gradio interface
with gr.Blocks(title="Customer Spending Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💰 Customer Spending Predictor")
    gr.Markdown("Predict yearly spending based on customer behavior and characteristics.")
    
    with gr.Row():
        with gr.Column():
            avg_session = gr.Slider(
                label="Average Session Length (minutes)",
                minimum=float(customers['Avg. Session Length'].min()),
                maximum=float(customers['Avg. Session Length'].max()),
                value=float(customers['Avg. Session Length'].mean()),
                step=0.1
            )
            time_app = gr.Slider(
                label="Time on App (minutes)",
                minimum=float(customers['Time on App'].min()),
                maximum=float(customers['Time on App'].max()),
                value=float(customers['Time on App'].mean()),
                step=0.1
            )
            time_website = gr.Slider(
                label="Time on Website (minutes)",
                minimum=float(customers['Time on Website'].min()),
                maximum=float(customers['Time on Website'].max()),
                value=float(customers['Time on Website'].mean()),
                step=0.1
            )
            membership = gr.Slider(
                label="Length of Membership (years)",
                minimum=float(customers['Length of Membership'].min()),
                maximum=float(customers['Length of Membership'].max()),
                value=float(customers['Length of Membership'].mean()),
                step=0.1
            )
            predict_btn = gr.Button("🔮 Predict Spending", variant="primary")
        
        with gr.Column():
            prediction_output = gr.Textbox(
                label="Predicted Yearly Spending",
                value="$0.00",
                interactive=False
            )
            feature_impact = gr.Textbox(
                label="Feature Impact (per unit change)",
                lines=5,
                interactive=False
            )
    
    # Model info section
    with gr.Accordion("📊 Model Information", open=False):
        X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
        y = customers['Yearly Amount Spent']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        predictions = lm.predict(X_test)
        
        r2_train = lm.score(X_train, y_train)
        r2_test = lm.score(X_test, y_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = math.sqrt(mean_squared_error(y_test, predictions))
        
        gr.Markdown(f"""
        **Model Performance Metrics:**
        - R² Score (Train): {r2_train:.4f}
        - R² Score (Test): {r2_test:.4f}
        - Mean Absolute Error: ${mae:.2f}
        - Root Mean Squared Error: ${rmse:.2f}
        
        **Dataset Info:**
        - Total Customers: {len(customers)}
        - Average Spending: ${customers['Yearly Amount Spent'].mean():.2f}
        """)
    
    # Connect inputs to prediction function
    predict_btn.click(
        fn=predict_spending,
        inputs=[avg_session, time_app, time_website, membership],
        outputs=[prediction_output, feature_impact]
    )

if __name__ == "__main__":
    demo.launch(share=False)  # Set share=True to get a public link


