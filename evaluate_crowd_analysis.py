import numpy as np
import pandas as pd
import matplotlib

# Use 'agg' backend for non-GUI environments (change to 'tkagg' if GUI is needed)
matplotlib.use('agg')  
import matplotlib.pyplot as plt
import json
import csv
from datetime import datetime
import os


def create_confusion_matrix(y_true, y_pred):
    """
    Create a 2x2 confusion matrix manually since sklearn is not available
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, precision, recall, and F1 score
    """
    conf_matrix = create_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1, conf_matrix


def plot_confusion_matrix(conf_matrix, title):
    """
    Plot confusion matrix using matplotlib
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {title}')

    # Add numbers to the plot
    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0, 1], ['Normal', 'Abnormal'])
    plt.yticks([0, 1], ['Normal', 'Abnormal'])
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'confusion_matrix_{title.lower().replace(" ", "_")}.png')
    plt.close()


def evaluate_crowd_analysis():
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists('evaluation_results'):
            os.makedirs('evaluation_results')

        # Load the processed data
        crowd_data = pd.read_csv('processed_data/crowd_data.csv')

        with open('processed_data/video_data.json', 'r') as f:
            video_data = json.load(f)

        # Initialize results dictionary
        results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'video_info': {
                'fps': video_data['VID_FPS'],
                'frame_size': video_data['PROCESSED_FRAME_SIZE'],
                'duration': video_data['END_TIME']
            },
            'metrics': {}
        }

        # Prepare the detection types
        detection_types = {
            'Abnormal Activity': crowd_data['Abnormal Activity'],
            'Social Distance Violation': (crowd_data['Social Distance violate'] > 0).astype(int),
            'Restricted Entry': crowd_data['Restricted Entry']
        }

        print("\nCrowd Analysis Evaluation Results")
        print("=" * 50)

        for detection_type, pred_values in detection_types.items():
            # Create synthetic ground truth for demonstration
            # In practice, replace this with actual ground truth data
            np.random.seed(42)  # For reproducibility
            ground_truth = pred_values.copy()
            noise = np.random.random(len(ground_truth)) > 0.9
            ground_truth[noise] = 1 - ground_truth[noise]  # Properly flipping labels with noise

            # Calculate metrics
            accuracy, precision, recall, f1, conf_matrix = calculate_metrics(ground_truth, pred_values)

            # Store results
            results['metrics'][detection_type] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': conf_matrix.tolist()
            }

            # Print results
            print(f"\n{detection_type} Detection Results:")
            print(f"Accuracy:  {accuracy:.2%}")
            print(f"Precision: {precision:.2%}")
            print(f"Recall:    {recall:.2%}")
            print(f"F1 Score:  {f1:.2%}")

            # Plot and save confusion matrix
            plot_confusion_matrix(conf_matrix, detection_type)

        # Save results to JSON
        with open('evaluation_results/metrics.json', 'w') as f:
            json.dump(results, f, indent=4)

        return results

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have run main.py first to generate the necessary data files.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    results = evaluate_crowd_analysis()
    if results:
        print("\nEvaluation complete. Results have been saved in the 'evaluation_results' directory.")
