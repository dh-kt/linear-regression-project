#visualization_file
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_results(y_test, y_pred):
    # check y_test and y_pred have the same shape
    if y_test.shape != y_pred.shape:
        raise ValueError("y_test and y_pred must have the same shape.")

    """Visualize actual vs predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label='Predictions')
    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs. Predicted Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()