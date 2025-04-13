import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_model_comparison(results, dataset_type='test'):
    """
    Create a bar chart to compare model performance.
    
    Args:
        results (dict): Dictionary of performance metrics for each model.
        dataset_type (str, optional): 'train' or 'test'. Defaults to 'test'.
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with the model comparison chart.
    """
    # Prepare data for the plot
    models = list(results.keys())
    accuracy = [results[model][f'{dataset_type}_accuracy'] for model in models]
    precision = [results[model][f'{dataset_type}_precision'] for model in models]
    recall = [results[model][f'{dataset_type}_recall'] for model in models]
    f1 = [results[model][f'{dataset_type}_f1'] for model in models]
    roc_auc = [results[model][f'{dataset_type}_roc_auc'] for model in models]
    
    # Create a dataframe for the plot
    data = {
        'Model': models,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }
    
    df = pd.DataFrame(data)
    
    # Melt the dataframe for plotting
    df_melted = df.melt(
        id_vars='Model', 
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        var_name='Metric', 
        value_name='Value'
    )
    
    # Create the bar chart
    fig = px.bar(
        df_melted, 
        x='Model', 
        y='Value', 
        color='Metric', 
        barmode='group',
        title=f'Model Performance Comparison ({dataset_type.capitalize()} Set)',
        labels={'Value': 'Score', 'Model': 'Model'},
        height=500
    )
    
    fig.update_layout(
        xaxis_title='Model',
        yaxis_title='Score',
        legend_title='Metric',
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig

def plot_metrics_comparison(results, metrics=None, dataset_type='test'):
    """
    Create a radar chart to compare multiple metrics across models.
    
    Args:
        results (dict): Dictionary of performance metrics for each model.
        metrics (list, optional): List of metrics to compare. Defaults to None.
        dataset_type (str, optional): 'train' or 'test'. Defaults to 'test'.
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with the metrics comparison chart.
    """
    if metrics is None:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    
    # Prepare data for the plot
    models = list(results.keys())
    
    # Create figure
    fig = go.Figure()
    
    # Metric mapping from display name to result key
    metric_mapping = {
        'Accuracy': f'{dataset_type}_accuracy',
        'Precision': f'{dataset_type}_precision',
        'Recall': f'{dataset_type}_recall',
        'F1 Score': f'{dataset_type}_f1',
        'ROC AUC': f'{dataset_type}_roc_auc'
    }
    
    # Add traces for each model
    for model in models:
        values = [results[model][metric_mapping[metric]] for metric in metrics]
        # Add the first value again to close the radar chart
        values.append(values[0])
        metrics_plot = metrics + [metrics[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_plot,
            fill='toself',
            name=model
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=f'Model Metrics Comparison ({dataset_type.capitalize()} Set)',
        showlegend=True,
        height=600
    )
    
    return fig

def plot_tuning_results(comparison_data):
    """
    Create a bar chart to compare original and tuned model performance.
    
    Args:
        comparison_data (dict): Dictionary with model comparison data.
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with the comparison chart.
    """
    # Prepare data for the plot
    df = pd.DataFrame(comparison_data)
    
    # Melt the dataframe for plotting
    df_melted = df.melt(
        id_vars='Model', 
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        var_name='Metric', 
        value_name='Value'
    )
    
    # Create the bar chart
    fig = px.bar(
        df_melted, 
        x='Metric', 
        y='Value', 
        color='Model', 
        barmode='group',
        title='Original vs. Tuned Model Performance',
        labels={'Value': 'Score', 'Metric': 'Metric'},
        height=500
    )
    
    fig.update_layout(
        xaxis_title='Metric',
        yaxis_title='Score',
        legend_title='Model',
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig
