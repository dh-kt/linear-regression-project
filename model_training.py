#model_training_file
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report

def train_and_evaluate_model(X, y, task = 'regression'):
    """Train a Linear Regression model and evaluate its performance."""
    """Train and evaluate a model based on the task (regression or classification)."""

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure X contains only numeric data
    X_train = X_train.select_dtypes(include=['number'])
    X_test = X_test.select_dtypes(include=['number'])

    # Ensure y is a 1D array
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    """Train a Linear Regression model and evaluate its performance."""

#added for numeric X and y 1D array
    X = X.select_dtypes(include=['number'])
    y = y.values.ravel()

    #Initialize the model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\nModel Evaluation:")
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared (R²):", r2)

#Check for Overfitting
    y_train_pred = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    print("\nCheck for Overfitting")
    print("Training MAE:", mae_train)
    print("Training R²:", r2_train)

#Check the Scale of the Target Variable
    average_price = y_test.mean()
    print(' \nThe Scale of the Target Variable')
    print("Average price in test set:", average_price)
    print("MAE as a percentage of average price:", (mae / average_price) * 100)

    return y_test, y_pred
