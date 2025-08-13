# Authors: Clement Gomiero, Christian Wiemer
# Some code was adapted from https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_1_BookContent/BookContent.html

# Project Purpose:
# This project aims to develop, train, and evaluate various machine learning models for credit card fraud detection using simulated transactional data. 
# The methodology includes data generation, fraud simulation, and model performance assessment across multiple metrics like accuracy, precision, recall, F1-score, and ROC AUC. 
# Key objectives are to analyze the trade-offs between computational resources and model effectiveness and to identify the most efficient algorithm for large-scale fraud detection.
# The models used are: 
#   - Random Forest
#   - Logistic regression
#   - Gaussian Naive Bayes
#   - Gradient Boosting Classifier
#   - K-Neighbors Classifier
#   - Support Vector Classifier (SVC)
#   - Artificial Neural Network


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import time
import psutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from sklearn.linear_model import SGDClassifier


# Returns the current CPU usage as a percentage and memory usage as a percentage.
def get_cpu_usage():
    cpu_usage = psutil.cpu_percent(interval=1)  # CPU usage percentage
    memory_info = psutil.virtual_memory()       # Memory usage
    return cpu_usage, memory_info.percent

# Returns GPU utilization and memory usage
def get_gpu_usage():
    if torch.cuda.is_available():
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            gpu_utilization, gpu_memory_usage = map(float, result.stdout.strip().split(','))
            return gpu_memory_usage, gpu_utilization
    return None, None

# Function to load and preprocess the data
# The Function loads two files but with a few changes, it is possible to test and train on the same file
def load_and_preprocess_data():
    # Load the datasets to test on
    transactions_df = pd.read_csv('1000C_data/transactions.csv')
    customer_profiles_df = pd.read_csv('1000C_data/customer_profiles.csv')
    terminal_profiles_df = pd.read_csv('1000C_data/terminal_profiles.csv')

    # Load the datasets to train the model on
    training_transactions_df = pd.read_csv('100C_data/transactions.csv')
    training_customer_profiles_df = pd.read_csv('100C_data/customer_profiles.csv')
    training_terminal_profiles_df = pd.read_csv('100C_data/terminal_profiles.csv')
    
    # Preprocess the training data
    training_transactions_df = training_transactions_df.merge(
        training_customer_profiles_df[['CUSTOMER_ID', 'x_customer_id', 'y_customer_id', 'mean_amount', 'std_amount', 'mean_nb_tx_per_day']],
        on='CUSTOMER_ID', 
        how='left'
    ).merge(
        training_terminal_profiles_df[['TERMINAL_ID', 'x_terminal_id', 'y_terminal_id']],
        on='TERMINAL_ID',
        how='left'
    )
    training_transactions_df['FRAUD_LABEL'] = training_transactions_df['TX_FRAUD']

    # Preprocess the testing data
    transactions_df = transactions_df.merge(
        customer_profiles_df[['CUSTOMER_ID', 'x_customer_id', 'y_customer_id', 'mean_amount', 'std_amount', 'mean_nb_tx_per_day']],
        on='CUSTOMER_ID', 
        how='left'
    ).merge(
        terminal_profiles_df[['TERMINAL_ID', 'x_terminal_id', 'y_terminal_id']],
        on='TERMINAL_ID',
        how='left'
    )
    transactions_df['FRAUD_LABEL'] = transactions_df['TX_FRAUD']

    # Select features for the model
    features = [
        'TX_AMOUNT',                 # Transaction amount
        'x_customer_id', 'y_customer_id',  # Customer location
        'x_terminal_id', 'y_terminal_id',  # Terminal location
        'mean_amount', 'std_amount',       # Customer behavioral metrics
        'mean_nb_tx_per_day',              # Average daily transactions
        'TX_TIME_SECONDS', 'TX_TIME_DAYS' # Time-related features
    ]
    
    # Create the training and testing datasets
    X_train = training_transactions_df[features]
    y_train = training_transactions_df['FRAUD_LABEL']
    X_test = transactions_df[features]
    y_test = transactions_df['FRAUD_LABEL']

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Function to print model metrics
def print_model_metrics(model, X_test, y_test, elapsed_time):
    print('Testing Model: ', model)
    
    start_time = time.time()
    if(model == "MLPClassifier"):
        # Evaluate the model
        pred = (model.predict(X_test) > 0.5).astype(int)  # Convert probabilities to binary predictions
    else:
        # Make predictions
        pred = model.predict(X_test)
        
    cpu_usage_after, memory_usage_after = get_cpu_usage() 
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    # For probabilistic predictions (required for ROC AUC, Log Loss)
    if hasattr(model, "predict_proba"):
        pred_proba = model.predict_proba(X_test)[:, 1]  # For binary classification
    else:
        pred_proba = None
    
    # Get various metrics
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='binary')  # Modify for multi-class
    recall = recall_score(y_test, pred, average='binary')  # Modify for multi-class
    f1 = f1_score(y_test, pred, average='binary')  # Modify for multi-class
    roc_auc = roc_auc_score(y_test, pred_proba) if pred_proba is not None else None
    
    # Store the results in a dictionary
    results = {
        'Model': type(model).__name__,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'Time (s)': elapsed_time,
        'CPU Usage': cpu_usage_after,
        'Memory': memory_usage_after
    }
    
    return results

def plot_roc_curves(models, x_test_total, y_test_total):
    # Initialize the plot
    plt.figure(figsize=(12, 8))
    
    # Loop through each model and its corresponding test data
    for model, X_test, y_test in zip(models, x_test_total, y_test_total):
        try:
            # Check if the model supports predict_proba
            if hasattr(model, 'predict_proba'):
                # Predict probabilities for the positive class
                pred_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                # Use decision_function for models like SVM if available
                pred_proba = model.decision_function(X_test)
            else:
                print(f"Model {type(model).__name__} does not support probability estimation. Skipping.")
                continue
            
            # Compute ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test, pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot the ROC curve
            plt.plot(fpr, tpr, label=f'{type(model).__name__} (AUC = {roc_auc:.3f})')
        
        except Exception as e:
            # Catch any errors for individual models and continue
            print(f"Error processing {type(model).__name__}: {e}")
            continue
    
    # Add a diagonal line for random guessing
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
    
    # Customize the plot
    plt.title('ROC Curves for Models', fontsize=16)
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    
    # Display the plot
    plt.show()
    
# Function to train a Random Forest model
def train_random_forest(X_train, y_train, X_test, y_test):
    print("Training Random Forest Model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
        
    results = print_model_metrics(rf_model, X_test, y_test, elapsed_time = 0)
    return X_test, y_test, rf_model, results
    #plot_roc_curves(X_test, y_test, rf_model)

# Function to train a Logistic Regression model
def train_logistic_regression(X_train, y_train, X_test, y_test):
    print("Training Logistic Regression Model...")
    logreg_model = LogisticRegression(max_iter=1000, random_state=42)
    logreg_model.fit(X_train, y_train)
    
    model_stats = print_model_metrics(logreg_model, X_test, y_test, elapsed_time = 0)
    return X_test, y_test, logreg_model, model_stats
    #plot_roc_curves(X_test, y_test, logreg_model)

# Function to train a Naive Bayes model
def train_naive_bayes(X_train, y_train, X_test, y_test):
    print("Training Naive Bayes Model...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
 
    model_stats = print_model_metrics(nb_model, X_test, y_test, elapsed_time = 0)
    return X_test, y_test, nb_model, model_stats
    #plot_roc_curves(X_test, y_test, nb_model)
    
# Function to train a Support Vector Classifier (SVC) model
def train_svc(X_train, y_train, X_test, y_test):
    print("Training Support Vector Classifier Model...")
    svc_model = SGDClassifier(loss="hinge", random_state=42)
    
    # Train the model in batches of 1000
    batch_size = 1000

    for start in range(0, len(X_train), batch_size):
        end = start + batch_size
        X_batch = X_train[start:end]
        y_batch = y_train[start:end]
    
        # Fit the model incrementally
        svc_model.partial_fit(X_batch, y_batch, classes=np.unique(y_train))
    
    model_stats = print_model_metrics(svc_model, X_test, y_test, elapsed_time = 0)
    return X_test, y_test, svc_model, model_stats
    #plot_roc_curves(X_test, y_test, svc_model)

# Function to train and evaluate Gradient Boosting Classifier
def train_gradient_boosting(X_train, y_train, X_test, y_test):

    # Train the Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)
 
    model_stats = print_model_metrics(gb_model, X_test, y_test, elapsed_time = 0)
    return X_test, y_test, gb_model, model_stats
    #plot_roc_curves(X_test, y_test, gb_model)

def train_ann(X_train, y_train, X_test, y_test, input_dim=None, epochs=10, batch_size=32):
    # Build the ANN model
    ann_model = Sequential([
        Dense(128, input_dim=input_dim or X_train.shape[1], activation='relu'),  # Input layer + first hidden layer
        Dropout(0.3),  # Dropout to prevent overfitting
        Dense(64, activation='relu'),  # Second hidden layer
        Dropout(0.3),  # Dropout to prevent overfitting
        Dense(1, activation='sigmoid')  # Output layer with sigmoid for binary classification
    ])
    
    # Compile the model
    ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = ann_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    # Calculate model metrics
    model_stats = print_model_metrics(ann_model, X_test, y_test, elapsed_time=0)
    return X_test, y_test, ann_model, model_stats
    
def train_knn(X_train, y_train, X_test, y_test, n_neighbors=5):
    # Initialize and train the kNN model
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    
    model_stats = print_model_metrics(knn_model, X_test, y_test, elapsed_time = 0)
    return X_test, y_test, knn_model, model_stats
    #plot_roc_curves(X_test, y_test, knn_model)

# Main function to run all models
def run_all_models():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Train and evaluate each model
    x_test_rf, y_test_rf, random_forest, rf_result = train_random_forest(X_train, y_train, X_test, y_test)
    x_test_log, y_test_log, logisitic_reg, log_result = train_logistic_regression(X_train, y_train, X_test, y_test)
    x_test_nb, y_test_nb,naive_bayes, nb_result = train_naive_bayes(X_train, y_train, X_test, y_test)
    x_test_svc, y_test_svc,svc, svc_result = train_svc(X_train, y_train, X_test, y_test)
    x_test_gb, y_test_gb, gb, gb_result = train_gradient_boosting(X_train, y_train, X_test, y_test)
    x_test_ann, y_test_ann, ann, ann_result = train_ann(X_train, y_train, X_test, y_test)
    x_test_knn, y_test_knn, knn, knn_result = train_knn(X_train, y_train, X_test, y_test)
    
    # Set all the results together to print the data inside the ROC AUC Graph
    models = [random_forest, logisitic_reg, naive_bayes, svc, gb, knn, ann]
    x_test_total = [x_test_rf, x_test_log, x_test_nb, x_test_svc, x_test_gb, x_test_knn, x_test_ann]
    y_test_total = [y_test_rf, y_test_log, y_test_nb, y_test_svc, y_test_gb, y_test_knn, y_test_ann]
    all_results = [rf_result, log_result, nb_result, svc_result, gb_result, knn_result, ann_result]
    
    results_df = pd.DataFrame(all_results)

    # Format the 'Accuracy' and 'Time (s)' columns to show 3 decimal places
    results_df = results_df.round(3)
    
    # Print the results
    print("\nTESTING RESULTS")
    print(results_df)
    
    plot_roc_curves(models, x_test_total, y_test_total)
    

# Run all models
run_all_models()
