#main file that connect all modules from data_loading import load_data
from data_preprocessing import preprocess_data
from model_training import train_and_evaluate_model
from visualization import visualize_results
from data_loading import load_data

# Main script
if __name__ == "__main__":
    # Load data
    filepath = "D:\\hirko\\AAA\\2025\\shoes_fact.csv"
    df = load_data(filepath)

    # Preprocess data
    target_column = 'price'
    df = preprocess_data(df, target_column)

    # Split data into features (X) and target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column + '_transformed']

       # Train and evaluate the model
    task = 'regression'  # Change to 'classification' if your task is classification
    results = train_and_evaluate_model(X, y, task=task)

    # Train and evaluate the model
    y_test, y_pred = train_and_evaluate_model(X, y)

    # Visualize results
    visualize_results(y_test, y_pred)