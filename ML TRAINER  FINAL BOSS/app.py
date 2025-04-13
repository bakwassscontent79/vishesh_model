import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import joblib
import io

# Import custom modules
from data_preprocessing import load_and_preprocess_data, display_data_statistics
from model_training import (
    train_models, 
    hyperparameter_tuning, 
    get_default_hyperparameters,
    get_hyperparameter_options
)
from model_evaluation import (
    evaluate_models, 
    calculate_metrics, 
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance
)
from visualization import (
    plot_model_comparison, 
    plot_metrics_comparison, 
    plot_tuning_results
)
from utils import download_results

# Page configuration
st.set_page_config(
    page_title="ML Model Comparison Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("📊 Classification Model Comparison Dashboard")
st.markdown("""
This dashboard compares multiple machine learning models for classification tasks. 
You can use the built-in heart disease dataset, or upload your own CSV file.
""")

# Sidebar
st.sidebar.title("Model Settings")

# Dataset selection
st.sidebar.header("Dataset Selection")
dataset_option = st.sidebar.radio(
    "Select Dataset Source",
    ["Built-in Heart Disease Dataset", "Upload Your Own Dataset"]
)

# Data loading and preprocessing
@st.cache_data
def get_data():
    return load_and_preprocess_data()

@st.cache_data
def process_uploaded_data(df, target_column):
    """
    Process an uploaded dataset
    
    Args:
        df (pd.DataFrame): The uploaded dataframe
        target_column (str): The name of the target column
        
    Returns:
        tuple: (data, X, y, feature_names, target_name)
    """
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Basic preprocessing
    # Fill missing values with median for numerical columns
    for col in data.columns:
        if data[col].dtype != 'object' and data[col].isnull().any():
            data[col] = data[col].fillna(data[col].median())
    
    # Drop any remaining rows with missing values
    data = data.dropna()
    
    # Extract features and target
    if target_column in data.columns:
        X = data.drop(target_column, axis=1).values
        y = data[target_column].values
        feature_names = list(data.drop(target_column, axis=1).columns)
        
        # Check if target is binary (0/1)
        unique_values = np.unique(y)
        if len(unique_values) == 2:
            # Ensure target is 0/1
            if not set(unique_values) == {0, 1}:
                # Map the lowest value to 0 and highest to 1
                y = np.where(y == min(unique_values), 0, 1)
        else:
            st.warning("This dashboard is designed for binary classification. Your target variable has more than 2 classes.")
        
        # Scale the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return data, X, y, feature_names, target_column
    else:
        st.error(f"Target column '{target_column}' not found in the dataset.")
        st.stop()

# Data loading based on selection
if dataset_option == "Built-in Heart Disease Dataset":
    with st.spinner('Loading and preprocessing heart disease data...'):
        data, X, y, feature_names, target_name = get_data()
else:
    # File uploader
    st.sidebar.subheader("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the CSV file
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("File uploaded successfully!")
            
            # Display dataset preview in sidebar
            with st.sidebar.expander("Preview Dataset"):
                st.dataframe(df.head())
            
            # Select target column
            target_column = st.sidebar.selectbox("Select Target Column", df.columns)
            
            # Process the uploaded data
            with st.spinner('Processing uploaded data...'):
                data, X, y, feature_names, target_name = process_uploaded_data(df, target_column)
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            st.stop()
    else:
        # If no file is uploaded, use the heart disease dataset
        st.sidebar.info("No file uploaded. Using the heart disease dataset.")
        with st.spinner('Loading and preprocessing heart disease data...'):
            data, X, y, feature_names, target_name = get_data()

# Train-test split
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random Seed", 0, 1000, 42, 1)

# Model selection
st.sidebar.header("Select Models")
models_to_train = {
    "Logistic Regression": st.sidebar.checkbox("Logistic Regression", True),
    "Decision Tree": st.sidebar.checkbox("Decision Tree", True),
    "Random Forest": st.sidebar.checkbox("Random Forest", True),
    "SVM": st.sidebar.checkbox("Support Vector Machine", True),
    "Gradient Boosting": st.sidebar.checkbox("Gradient Boosting", True),
    "KNN": st.sidebar.checkbox("K-Nearest Neighbors", True)
}

selected_models = [model for model, selected in models_to_train.items() if selected]

if not selected_models:
    st.warning("Please select at least one model to train.")
    st.stop()

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview", 
    "🧮 Model Training", 
    "📈 Model Evaluation", 
    "🔧 Hyperparameter Tuning",
    "📋 Detailed Results"
])

# Tab 1: Data Overview
with tab1:
    st.header("Dataset Overview")
    
    # Show basic info about the dataset
    if dataset_option == "Built-in Heart Disease Dataset":
        st.subheader("Heart Disease Dataset")
    else:
        st.subheader("Uploaded Dataset")
    st.write(f"Number of samples: {data.shape[0]}")
    st.write(f"Number of features: {len(feature_names)}")
    
    # Display the first few rows of the dataset
    st.subheader("Sample Data")
    st.dataframe(data.head())
    
    # Display descriptive statistics
    with st.expander("View Descriptive Statistics"):
        display_data_statistics(data)
    
    # Target Distribution
    st.subheader("Target Distribution")
    fig = px.pie(
        names=['No Heart Disease', 'Heart Disease'],
        values=[(y == 0).sum(), (y == 1).sum()],
        title="Target Class Distribution",
        color_discrete_sequence=["#3498db", "#e74c3c"]
    )
    st.plotly_chart(fig)
    
    # Feature visualizations
    st.subheader("Feature Visualizations")
    
    viz_type = st.selectbox(
        "Select Visualization Type", 
        ["Histograms", "Box Plots", "Correlation Matrix"]
    )
    
    if viz_type == "Histograms":
        feature_to_plot = st.selectbox("Select Feature for Histogram", feature_names)
        fig = px.histogram(
            data, 
            x=feature_to_plot, 
            color=target_name,
            marginal="rug",
            title=f"Distribution of {feature_to_plot} by Target Class",
            color_discrete_sequence=["#3498db", "#e74c3c"]
        )
        st.plotly_chart(fig)
        
    elif viz_type == "Box Plots":
        feature_to_plot = st.selectbox("Select Feature for Box Plot", feature_names)
        fig = px.box(
            data, 
            x=target_name, 
            y=feature_to_plot,
            title=f"Box Plot of {feature_to_plot} by Target Class",
            color=target_name,
            color_discrete_sequence=["#3498db", "#e74c3c"]
        )
        st.plotly_chart(fig)
        
    elif viz_type == "Correlation Matrix":
        corr = data.corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig)

# Tab 2: Model Training
with tab2:
    st.header("Model Training")
    
    # Train-test split information
    st.write(f"Training on {int((1-test_size) * len(X))} samples, testing on {int(test_size * len(X))} samples")
    
    # Training button
    if st.button("Train Models"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train models
        with st.spinner('Training models... This may take a minute...'):
            models, train_times = train_models(X_train, y_train, selected_models)
            
            # Evaluate models
            results = evaluate_models(models, X_train, X_test, y_train, y_test)
            
            # Save results to session state
            st.session_state['models'] = models
            st.session_state['results'] = results
            st.session_state['train_times'] = train_times
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['feature_names'] = feature_names
            
        st.success("Models trained successfully!")
    
    # Display training results if available
    if 'results' in st.session_state:
        st.subheader("Training Results")
        
        # Model comparison
        st.write("### Model Performance Comparison")
        fig = plot_model_comparison(st.session_state['results'], 'test')
        st.plotly_chart(fig)
        
        # Training time comparison
        st.write("### Training Time Comparison")
        fig = px.bar(
            x=list(st.session_state['train_times'].keys()),
            y=list(st.session_state['train_times'].values()),
            labels={"x": "Model", "y": "Training Time (seconds)"},
            title="Model Training Time Comparison"
        )
        st.plotly_chart(fig)

# Tab 3: Model Evaluation
with tab3:
    st.header("Model Evaluation")
    
    if 'results' not in st.session_state:
        st.info("Please train the models first in the 'Model Training' tab.")
    else:
        results = st.session_state['results']
        models = st.session_state['models']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        # Select model for detailed evaluation
        selected_model = st.selectbox(
            "Select a model for detailed evaluation",
            list(models.keys())
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display metrics
            st.subheader("Performance Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
                'Training': [
                    results[selected_model]['train_accuracy'],
                    results[selected_model]['train_precision'],
                    results[selected_model]['train_recall'],
                    results[selected_model]['train_f1'],
                    results[selected_model]['train_roc_auc']
                ],
                'Testing': [
                    results[selected_model]['test_accuracy'],
                    results[selected_model]['test_precision'],
                    results[selected_model]['test_recall'],
                    results[selected_model]['test_f1'],
                    results[selected_model]['test_roc_auc']
                ]
            })
            st.dataframe(metrics_df, use_container_width=True)
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm_fig = plot_confusion_matrix(
                results[selected_model]['test_confusion_matrix'],
                title=f"{selected_model} Confusion Matrix"
            )
            st.pyplot(cm_fig)
        
        # Feature importance
        st.subheader("Feature Importance")
        try:
            fi_fig = plot_feature_importance(
                models[selected_model], 
                st.session_state['feature_names'],
                model_name=selected_model
            )
            st.pyplot(fi_fig)
        except:
            st.write("Feature importance not available for this model type.")

# Tab 4: Hyperparameter Tuning
with tab4:
    st.header("Hyperparameter Tuning")
    
    if 'models' not in st.session_state:
        st.info("Please train the models first in the 'Model Training' tab.")
    else:
        # Select model for tuning
        model_for_tuning = st.selectbox(
            "Select a model for hyperparameter tuning",
            list(st.session_state['models'].keys())
        )
        
        # Get hyperparameter options for the selected model
        param_options = get_hyperparameter_options(model_for_tuning)
        default_params = get_default_hyperparameters(model_for_tuning)
        
        st.subheader("Hyperparameter Settings")
        
        # Allow user to set hyperparameters
        tuning_params = {}
        col1, col2 = st.columns(2)
        
        with col1:
            for i, (param, values) in enumerate(list(param_options.items())[:len(param_options)//2 + len(param_options)%2]):
                if isinstance(values, list):
                    tuning_params[param] = st.selectbox(
                        f"{param}", values, index=values.index(default_params[param]) if default_params[param] in values else 0
                    )
                elif isinstance(values, tuple) and len(values) == 3:
                    min_val, max_val, step = values
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        tuning_params[param] = st.slider(
                            f"{param}", min_val, max_val, default_params[param], step
                        )
                    else:
                        tuning_params[param] = st.slider(
                            f"{param}", float(min_val), float(max_val), float(default_params[param]), float(step)
                        )
        
        with col2:
            for i, (param, values) in enumerate(list(param_options.items())[len(param_options)//2 + len(param_options)%2:]):
                if isinstance(values, list):
                    tuning_params[param] = st.selectbox(
                        f"{param} ", values, index=values.index(default_params[param]) if default_params[param] in values else 0
                    )
                elif isinstance(values, tuple) and len(values) == 3:
                    min_val, max_val, step = values
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        tuning_params[param] = st.slider(
                            f"{param} ", min_val, max_val, default_params[param], step
                        )
                    else:
                        tuning_params[param] = st.slider(
                            f"{param} ", float(min_val), float(max_val), float(default_params[param]), float(step)
                        )
        
        # Cross-validation settings
        st.subheader("Cross-Validation Settings")
        cv_folds = st.slider("Number of CV Folds", 2, 10, 5, 1)
        
        # Tuning button
        if st.button("Tune Model"):
            with st.spinner(f'Tuning {model_for_tuning}... This may take a while...'):
                # Get data from session state
                X_train = st.session_state['X_train']
                y_train = st.session_state['y_train']
                X_test = st.session_state['X_test']
                y_test = st.session_state['y_test']
                
                # Perform hyperparameter tuning
                tuned_model, cv_results = hyperparameter_tuning(
                    model_for_tuning, tuning_params, X_train, y_train, cv_folds
                )
                
                # Evaluate tuned model
                tuned_results = {}
                st.session_state['models'][f"{model_for_tuning} (Tuned)"] = tuned_model
                
                # Calculate metrics for tuned model
                tuned_results = calculate_metrics(
                    tuned_model, X_train, X_test, y_train, y_test
                )
                
                # Update results in session state
                st.session_state['results'][f"{model_for_tuning} (Tuned)"] = tuned_results
                st.session_state['cv_results'] = cv_results
                
            st.success("Model tuned successfully!")
            
        # Show tuning results if available
        if 'cv_results' in st.session_state:
            st.subheader("Tuning Results")
            
            # Show cross-validation results
            st.write("### Cross-Validation Scores")
            cv_results = st.session_state['cv_results']
            
            # Plot mean test scores
            cv_df = pd.DataFrame(cv_results)
            st.dataframe(cv_df[['mean_test_score', 'std_test_score', 'rank_test_score']])
            
            # Compare original and tuned model
            if f"{model_for_tuning} (Tuned)" in st.session_state['results']:
                st.write("### Original vs. Tuned Model Comparison")
                
                # Prepare data for comparison
                comparison_data = {
                    'Model': [model_for_tuning, f"{model_for_tuning} (Tuned)"],
                    'Accuracy': [
                        st.session_state['results'][model_for_tuning]['test_accuracy'],
                        st.session_state['results'][f"{model_for_tuning} (Tuned)"]['test_accuracy']
                    ],
                    'Precision': [
                        st.session_state['results'][model_for_tuning]['test_precision'],
                        st.session_state['results'][f"{model_for_tuning} (Tuned)"]['test_precision']
                    ],
                    'Recall': [
                        st.session_state['results'][model_for_tuning]['test_recall'],
                        st.session_state['results'][f"{model_for_tuning} (Tuned)"]['test_recall']
                    ],
                    'F1 Score': [
                        st.session_state['results'][model_for_tuning]['test_f1'],
                        st.session_state['results'][f"{model_for_tuning} (Tuned)"]['test_f1']
                    ],
                    'ROC AUC': [
                        st.session_state['results'][model_for_tuning]['test_roc_auc'],
                        st.session_state['results'][f"{model_for_tuning} (Tuned)"]['test_roc_auc']
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Display comparison
                st.dataframe(comparison_df, use_container_width=True)
                
                # Plot comparison
                fig = plot_tuning_results(comparison_data)
                st.plotly_chart(fig)

# Tab 5: Detailed Results
with tab5:
    st.header("Detailed Results")
    
    if 'results' not in st.session_state:
        st.info("Please train the models first in the 'Model Training' tab.")
    else:
        # All models metrics comparison
        st.subheader("All Models Metrics Comparison")
        
        # Select metrics to display
        metrics = st.multiselect(
            "Select metrics to compare", 
            ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
            default=["Accuracy", "F1 Score", "ROC AUC"]
        )
        
        if metrics:
            # Display metrics comparison
            fig = plot_metrics_comparison(st.session_state['results'], metrics)
            st.plotly_chart(fig)
        
        # Download results
        st.subheader("Download Results")
        
        download_format = st.selectbox(
            "Select download format",
            ["CSV", "JSON", "Excel"]
        )
        
        if st.button("Generate Download Link"):
            # Prepare results for download
            download_data = download_results(
                st.session_state['results'], 
                st.session_state['train_times'],
                format=download_format.lower()
            )
            
            # Create download link
            st.download_button(
                label=f"Download Results as {download_format}",
                data=download_data,
                file_name=f"heart_disease_model_comparison.{download_format.lower().replace('excel', 'xlsx')}",
                mime={
                    "csv": "text/csv",
                    "json": "application/json",
                    "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                }[download_format.lower()]
            )
        
        # Save models
        st.subheader("Save Best Model")
        
        # Select model to save
        model_to_save = st.selectbox(
            "Select a model to save",
            list(st.session_state['models'].keys()),
            key="model_to_save"
        )
        
        if st.button("Save Model"):
            with st.spinner("Saving model..."):
                # Save model using joblib
                model_data = joblib.dumps(st.session_state['models'][model_to_save])
                
                # Create download link
                st.download_button(
                    label=f"Download {model_to_save} Model",
                    data=model_data,
                    file_name=f"heart_disease_{model_to_save.lower().replace(' ', '_')}.joblib",
                    mime="application/octet-stream"
                )

# Add footer
st.markdown("""---""")
st.markdown("""
<div style="text-align: center">
    <p>Heart Disease ML Model Comparison Dashboard | Created with Streamlit</p>
</div>
""", unsafe_allow_html=True)
