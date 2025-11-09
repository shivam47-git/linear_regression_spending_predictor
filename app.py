"""
Streamlit GUI for E-commerce Customer Spending Prediction
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import statsmodels.api as sm
import scipy.stats as stats

# Page configuration
st.set_page_config(
    page_title="Customer Spending Predictor",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .graph-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">💰 Customer Spending Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

# File uploader in sidebar
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV or Excel file",
    type=['csv', 'xlsx', 'xls'],
    help="Upload your dataset to perform linear regression analysis"
)

# Load data and train model functions
def load_data_from_file(uploaded_file):
    """Load data from uploaded file"""
    if uploaded_file is not None:
        try:
            # Check file extension
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', 'xls')):
                return pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    return None

def detect_columns(customers):
    """Automatically detect feature columns and target column"""
    numeric_cols = customers.select_dtypes(include=[np.number]).columns.tolist()
    
    # Common target column names
    target_keywords = ['spent', 'amount', 'price', 'cost', 'value', 'target', 'y', 'output']
    target_col = None
    
    for col in numeric_cols:
        if any(keyword in col.lower() for keyword in target_keywords):
            target_col = col
            break
    
    # If no target found, use last numeric column
    if target_col is None and len(numeric_cols) > 0:
        target_col = numeric_cols[-1]
    
    # Feature columns are all numeric except target
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    # If we have the default dataset, use known columns
    if 'Yearly Amount Spent' in customers.columns:
        target_col = 'Yearly Amount Spent'
        feature_cols = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
    
    return feature_cols, target_col

def train_model(customers, feature_cols, target_col):
    """Train the linear regression model"""
    if not feature_cols or target_col is None:
        return None, None, None, None, None, None, None, None
    
    X = customers[feature_cols]
    y = customers[target_col]
    
    # Check for missing values
    if X.isnull().any().any() or y.isnull().any():
        st.warning("⚠️ Dataset contains missing values. They will be handled automatically.")
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    
    predictions = lm.predict(X_test)
    
    metrics = {
        'r2_train': lm.score(X_train, y_train),
        'r2_test': lm.score(X_test, y_test),
        'mae': mean_absolute_error(y_test, predictions),
        'mse': mean_squared_error(y_test, predictions),
        'rmse': math.sqrt(mean_squared_error(y_test, predictions))
    }
    
    return lm, X_train, X_test, y_train, y_test, predictions, metrics

# Check if file is uploaded
if uploaded_file is None:
    # Show welcome screen when no file is uploaded
    st.info("📁 **Please upload a CSV or Excel file to begin the analysis.**")
    st.markdown("""
    ### How to use:
    1. Use the file uploader in the sidebar (left side)
    2. Select a CSV or Excel file (.csv, .xlsx, .xls)
    3. The app will automatically:
       - Detect feature and target columns
       - Train a linear regression model
       - Generate all analysis graphs
       - Display results in the navigation pages
    
    ### Requirements:
    - Your file should contain numeric columns
    - At least one column should be suitable as a target variable
    - The app will automatically detect columns with keywords like: 'spent', 'amount', 'price', 'cost', 'value', 'target', 'y', 'output'
    """)
    st.stop()

# Load data only if file is uploaded
customers = load_data_from_file(uploaded_file)

if customers is None:
    st.error("❌ Error loading file. Please check your file format and try again.")
    st.stop()

# Show success message
st.success(f"✅ File loaded successfully! Dataset has {len(customers)} rows and {len(customers.columns)} columns.")

# Detect columns
feature_cols, target_col = detect_columns(customers)

if not feature_cols or target_col is None:
    st.error("❌ Could not detect feature and target columns. Please ensure your dataset has numeric columns.")
    st.stop()

# Train model
lm, X_train, X_test, y_train, y_test, predictions, metrics = train_model(customers, feature_cols, target_col)

if lm is None:
    st.error("❌ Could not train model. Please check your dataset.")
    st.stop()

# Sidebar navigation
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Choose a page:",
    ["🏠 Home", "📊 All Graphs", "🔮 Predict Spending", "📈 Model Performance", "📊 Visualizations", "🔍 Data Explorer"]
)

# Show dataset info in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Dataset Info")
st.sidebar.text(f"Rows: {len(customers)}")
st.sidebar.text(f"Columns: {len(customers.columns)}")
st.sidebar.text(f"Features: {len(feature_cols)}")
st.sidebar.text(f"Target: {target_col}")

# Home Page
if page == "🏠 Home":
    st.header("Welcome to the Customer Spending Predictor!")
    st.markdown("""
    This application uses **Linear Regression** to predict yearly spending of e-commerce customers 
    based on their behavior and characteristics.
    
    ### Features:
    - 🔮 **Predict Spending**: Input customer data and get spending predictions
    - 📈 **Model Performance**: View model metrics and evaluation
    - 📊 **Visualizations**: Explore relationships between features
    - 🔍 **Data Explorer**: Browse and analyze the dataset
    
    ### Dataset Overview:
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", len(customers))
    col2.metric("Features", len(feature_cols))
    avg_target = customers[target_col].mean()
    col3.metric(f"Avg. {target_col}", f"${avg_target:.2f}" if isinstance(avg_target, (int, float)) else f"{avg_target:.2f}")
    col4.metric("Model R² Score", f"{metrics['r2_test']:.3f}")
    
    st.markdown("### Dataset Preview:")
    st.dataframe(customers.head(10), use_container_width=True)
    
    st.markdown("### Feature Columns:")
    st.write(", ".join(feature_cols))
    st.markdown(f"### Target Column:")
    st.write(target_col)

# Prediction Page
elif page == "🔮 Predict Spending":
    st.header("🔮 Predict Customer Spending")
    st.markdown("Enter customer information to predict their yearly spending:")
    
    col1, col2 = st.columns(2)
    
    # Dynamic sliders based on feature columns
    input_values = {}
    cols = st.columns(2)
    
    for idx, feature in enumerate(feature_cols):
        col = cols[idx % 2]
        with col:
            min_val = float(customers[feature].min())
            max_val = float(customers[feature].max())
            mean_val = float(customers[feature].mean())
            input_values[feature] = st.slider(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100 if max_val != min_val else 0.1
            )
    
    # Make prediction
    input_data = np.array([[input_values[feat] for feat in feature_cols]])
    prediction = lm.predict(input_data)[0]
    
    st.markdown("---")
    st.markdown("### 📊 Prediction Result:")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background-color: #e8f4f8; border-radius: 10px;">
            <h2 style="color: #1f77b4; margin-bottom: 1rem;">Predicted Yearly Spending</h2>
            <h1 style="color: #2ca02c; font-size: 3rem;">${prediction:,.2f}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Show feature importance
    st.markdown("### 📋 Feature Impact:")
    coefficients = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': lm.coef_,
        'Impact': ['Positive' if c > 0 else 'Negative' for c in lm.coef_]
    })
    st.dataframe(coefficients, use_container_width=True)

# Model Performance Page
elif page == "📈 Model Performance":
    st.header("📈 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score (Train)", f"{metrics['r2_train']:.4f}")
    col2.metric("R² Score (Test)", f"{metrics['r2_test']:.4f}")
    col3.metric("MAE", f"${metrics['mae']:.2f}")
    col4.metric("RMSE", f"${metrics['rmse']:.2f}")
    
    st.markdown("---")
    
    # Model coefficients
    st.subheader("Model Coefficients")
    cdf = pd.DataFrame(lm.coef_, X_train.columns, columns=['Coefficient'])
    cdf['Absolute Impact'] = abs(cdf['Coefficient'])
    cdf = cdf.sort_values('Absolute Impact', ascending=False)
    st.dataframe(cdf, use_container_width=True)
    
    # Actual vs Predicted plot
    st.subheader("Actual vs Predicted Values")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.6, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Yearly Amount Spent')
    ax.set_ylabel('Predicted Yearly Amount Spent')
    ax.set_title('Model Predictions vs Actual Values')
    st.pyplot(fig)
    
    # Residuals analysis
    st.subheader("Residuals Analysis")
    residuals = y_test - predictions
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(residuals, bins=30, kde=True, ax=ax)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Residuals')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=residuals, alpha=0.6, ax=ax)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Actual Values')
        st.pyplot(fig)

# All Graphs Page - Shows all plots from linear_reg.py
elif page == "📊 All Graphs":
    st.header("📊 All Analysis Graphs")
    st.markdown("This page displays all graphs from the linear regression analysis. **Scroll down to view all visualizations.**")
    st.markdown("---")
    
    # Create a container for all graphs (Streamlit pages naturally scroll)
    with st.container():
        # Graph 1: Time on Website vs Target
        if 'Time on Website' in customers.columns:
            st.subheader("1. Time on Website vs " + target_col)
            fig = sns.jointplot(x='Time on Website', y=target_col, data=customers, alpha=0.5)
            st.pyplot(fig.fig)
            plt.close()
        
        # Graph 2: Time on App vs Target
        if 'Time on App' in customers.columns:
            st.subheader("2. Time on App vs " + target_col)
            fig = sns.jointplot(x='Time on App', y=target_col, data=customers, alpha=0.5)
            st.pyplot(fig.fig)
            plt.close()
        
        # Graph 3: Pair Plot
        st.subheader("3. Pair Plot (All Features)")
        st.info("⚠️ This may take a moment to load...")
        plot_cols = feature_cols + [target_col]
        fig = sns.pairplot(customers[plot_cols], kind='scatter', plot_kws={'alpha':0.4}, diag_kws={'alpha':0.55, 'bins':40})
        st.pyplot(fig.fig)
        plt.close()
        
        # Graph 4: Length of Membership vs Target (or first feature vs target)
        if 'Length of Membership' in customers.columns:
            st.subheader("4. Length of Membership vs " + target_col)
            fig = sns.lmplot(x='Length of Membership', y=target_col, data=customers, scatter_kws={'alpha':0.3})
            st.pyplot(fig.fig)
            plt.close()
        elif len(feature_cols) > 0:
            st.subheader(f"4. {feature_cols[0]} vs " + target_col)
            fig = sns.lmplot(x=feature_cols[0], y=target_col, data=customers, scatter_kws={'alpha':0.3})
            st.pyplot(fig.fig)
            plt.close()
        
        # Graph 5: Actual vs Predicted
        st.subheader("5. Actual vs Predicted Values")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=predictions, alpha=0.6, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel(f'Actual {target_col}')
        ax.set_ylabel(f'Predicted {target_col}')
        ax.set_title('Model Predictions vs Actual Values')
        st.pyplot(fig)
        plt.close()
        
        # Graph 6: Residuals Histogram
        st.subheader("6. Distribution of Residuals")
        residuals = y_test - predictions
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(residuals, bins=30, kde=True, ax=ax)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Residuals')
        st.pyplot(fig)
        plt.close()
        
        # Graph 7: Q-Q Plot
        st.subheader("7. Q-Q Plot (Normality Check)")
        fig, ax = plt.subplots(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot of Residuals')
        st.pyplot(fig)
        plt.close()
        
        # Graph 8: Correlation Heatmap
        st.subheader("8. Correlation Heatmap")
        plot_cols = feature_cols + [target_col]
        numeric_cols = customers[plot_cols]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_cols.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Heatmap')
        st.pyplot(fig)
        plt.close()
    
    # Summary statistics
    st.markdown("---")
    st.subheader("📊 Summary Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Model Metrics:**")
        st.write(f"- R² Score (Train): {metrics['r2_train']:.4f}")
        st.write(f"- R² Score (Test): {metrics['r2_test']:.4f}")
        st.write(f"- Mean Absolute Error: {metrics['mae']:.2f}")
        st.write(f"- Root Mean Squared Error: {metrics['rmse']:.2f}")
    with col2:
        st.write("**Dataset Info:**")
        st.write(f"- Total Rows: {len(customers)}")
        st.write(f"- Features: {len(feature_cols)}")
        st.write(f"- Target Column: {target_col}")

# Visualizations Page
elif page == "📊 Visualizations":
    st.header("📊 Data Visualizations")
    
    viz_option = st.selectbox(
        "Choose a visualization:",
        [
            "Time on Website vs Spending",
            "Time on App vs Spending",
            "Length of Membership vs Spending",
            "Pair Plot (All Features)",
            "Correlation Heatmap"
        ]
    )
    
    if viz_option == "Time on Website vs Spending" and 'Time on Website' in customers.columns:
        fig = sns.jointplot(x='Time on Website', y=target_col, data=customers, alpha=0.5)
        st.pyplot(fig.fig)
        plt.close()
    
    elif viz_option == "Time on App vs Spending" and 'Time on App' in customers.columns:
        fig = sns.jointplot(x='Time on App', y=target_col, data=customers, alpha=0.5)
        st.pyplot(fig.fig)
        plt.close()
    
    elif viz_option == "Length of Membership vs Spending" and 'Length of Membership' in customers.columns:
        fig = sns.lmplot(x='Length of Membership', y=target_col, data=customers, scatter_kws={'alpha':0.3})
        st.pyplot(fig.fig)
        plt.close()
    
    elif viz_option == "Pair Plot (All Features)":
        st.info("⚠️ This may take a moment to load...")
        plot_cols = feature_cols + [target_col]
        fig = sns.pairplot(customers[plot_cols], 
                          kind='scatter', plot_kws={'alpha':0.4}, diag_kws={'alpha':0.55, 'bins':40})
        st.pyplot(fig.fig)
        plt.close()
    
    elif viz_option == "Correlation Heatmap":
        plot_cols = feature_cols + [target_col]
        numeric_cols = customers[plot_cols]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_cols.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Heatmap')
        st.pyplot(fig)
        plt.close()

# Data Explorer Page
elif page == "🔍 Data Explorer":
    st.header("🔍 Data Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Statistics")
        st.dataframe(customers.describe(), use_container_width=True)
    
    with col2:
        st.subheader("Dataset Info")
        st.text(f"Shape: {customers.shape}")
        st.text(f"Columns: {', '.join(customers.columns)}")
        st.text(f"Missing Values: {customers.isnull().sum().sum()}")
        st.text(f"Feature Columns: {', '.join(feature_cols)}")
        st.text(f"Target Column: {target_col}")
    
    st.markdown("---")
    st.subheader("Browse Dataset")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        min_target = st.slider(f"Min {target_col}", 
                                float(customers[target_col].min()),
                                float(customers[target_col].max()),
                                float(customers[target_col].min()))
    with col2:
        max_target = st.slider(f"Max {target_col}",
                                float(customers[target_col].min()),
                                float(customers[target_col].max()),
                                float(customers[target_col].max()))
    
    filtered_data = customers[
        (customers[target_col] >= min_target) &
        (customers[target_col] <= max_target)
    ]
    
    st.dataframe(filtered_data, use_container_width=True)
    st.caption(f"Showing {len(filtered_data)} of {len(customers)} rows")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with ❤️ using Streamlit | Linear Regression Model</p>
</div>
""", unsafe_allow_html=True)

