import pandas as pd
import json
import io

def download_results(results, train_times, format='csv'):
    """
    Prepare model comparison results for download.
    
    Args:
        results (dict): Dictionary of performance metrics for each model.
        train_times (dict): Dictionary of training times for each model.
        format (str, optional): Download format ('csv', 'json', or 'excel'). Defaults to 'csv'.
        
    Returns:
        bytes or str: Data in the specified format.
    """
    # Prepare metrics for each model
    model_results = []
    
    for model_name, metrics in results.items():
        model_result = {
            'Model': model_name,
            'Training Time (s)': train_times.get(model_name, 0),
            'Train Accuracy': metrics['train_accuracy'],
            'Test Accuracy': metrics['test_accuracy'],
            'Train Precision': metrics['train_precision'],
            'Test Precision': metrics['test_precision'],
            'Train Recall': metrics['train_recall'],
            'Test Recall': metrics['test_recall'],
            'Train F1': metrics['train_f1'],
            'Test F1': metrics['test_f1'],
            'Train ROC AUC': metrics['train_roc_auc'],
            'Test ROC AUC': metrics['test_roc_auc']
        }
        
        model_results.append(model_result)
    
    # Create a dataframe
    df = pd.DataFrame(model_results)
    
    # Return data in the specified format
    if format == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    elif format == 'json':
        return df.to_json(orient='records', indent=4).encode('utf-8')
    elif format == 'excel':
        output = io.BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
        return output.getvalue()
    else:
        raise ValueError(f"Format {format} not supported. Use 'csv', 'json', or 'excel'.")
