# Authors: Clement Gomiero, Christian Wiemer
# Some code was adapted from https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_1_BookContent/BookContent.html

# Program Summary:
# This program simulates credit card transactions for fraud detection.
# It generates synthetic data for customers, terminals, and transactions, 
# including legitimate and fraudulent cases based on predefined scenarios. 
# The output includes datasets for customer profiles, terminal profiles, 
# and transactions, along with fraud statistics.
#
# Key Features:
# - Creates synthetic customer and terminal profiles.
# - Simulates transactions with timestamps, amounts, and fraud labels.
# - Introduces fraud scenarios to mimic real-world patterns.
# - Saves generated data and statistics for further analysis.


#--------------- Transaction Feature Columns -----------------------
# TRANSACTION_ID
# TX_DATETIME
# CUSTOMER_ID
# TERMINAL_ID
# TX_AMOUNT
# TX_FRAUD

#---------------- Customer Data Columns ---------------------------
# CUSTOMER_ID
# x_customer_id, y_customer_id = [100][100] (coordinates)
# mean_amount, std_amount (mean and standard deviation of transaction amounts)
# mean_nb_tx_per_day (average daily transactions)
# available_terminals (list of accessible terminals for the customer)


import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
import random

import matplotlib.pyplot as plt
import seaborn as sns


# Class to represent a transaction with details like ID, datetime, customer, terminal, amount, and fraud status
class Transaction:
    def __init__(self, transaction_id, tx_datetime, customer_id, terminal_id, tx_amount, tx_fraud):
        # Initialize transaction attributes
        self.transaction_id = transaction_id  # Unique identifier for the transaction
        self.tx_datetime = tx_datetime        # Date and time of the transaction
        self.customer_id = customer_id        # Customer's unique identifier
        self.terminal_id = terminal_id        # Terminal's unique identifier
        self.tx_amount = tx_amount            # Amount of the transaction
        self.tx_fraud = tx_fraud              # Fraud label (1: Fraudulent, 0: Legitimate)

    def __str__(self):
        # Generate a user-friendly string representation of the transaction
        fraud_status = "Fraudulent" if self.tx_fraud == 1 else "Legitimate"
        return (f"Transaction ID: {self.transaction_id}\n"
                f"Date & Time: {self.tx_datetime}\n"
                f"Customer ID: {self.customer_id}\n"
                f"Terminal ID: {self.terminal_id}\n"
                f"Transaction Amount: {self.tx_amount}\n"
                f"Fraud Label: {fraud_status}\n")


# Class to represent a customer profile with attributes such as location, transaction statistics, etc.
class Customer:
    def __init__(self, customer_id, x_location, y_location, mean_amount, std_amount, mean_nb_tx_per_day):
        # Initialize customer attributes
        self.customer_id = customer_id        # Unique identifier for the customer
        self.x_location = x_location          # x-coordinate of the customer's location
        self.y_location = y_location          # y-coordinate of the customer's location
        self.mean_amount = mean_amount        # Average transaction amount
        self.std_amount = std_amount          # Standard deviation of transaction amounts
        self.mean_nb_tx_per_day = mean_nb_tx_per_day  # Average number of transactions per day

    def __str__(self):
        # Generate a user-friendly string representation of the customer
        return (f"Customer ID: {self.customer_id}\n"
                f"Location: ({self.x_location}, {self.y_location})\n"
                f"Mean Transaction Amount: {self.mean_amount}\n"
                f"Standard Deviation of Amount: {self.std_amount}\n"
                f"Mean Transactions Per Day: {self.mean_nb_tx_per_day}\n")


# Function to generate a table of customer profiles with random data
def generate_customer_profiles_table(n_customers, random_state=0):
    np.random.seed(random_state)  # Set random seed for reproducibility
    customer_id_properties = []

    # Generate random properties for each customer
    for customer_id in range(n_customers):
        x_customer_id = np.random.uniform(0, 100)  # Random x-coordinate
        y_customer_id = np.random.uniform(0, 100)  # Random y-coordinate
        mean_amount = np.random.uniform(5, 100)    # Random mean transaction amount
        std_amount = mean_amount / 2              # Standard deviation as half of the mean
        mean_nb_tx_per_day = np.random.uniform(0, 4)  # Random mean daily transactions

        customer_id_properties.append([customer_id, x_customer_id, y_customer_id, mean_amount, std_amount, mean_nb_tx_per_day])

    # Create a DataFrame with the generated data
    customer_profiles_table = pd.DataFrame(customer_id_properties, columns=['CUSTOMER_ID', 'x_customer_id', 'y_customer_id', 'mean_amount', 'std_amount', 'mean_nb_tx_per_day'])
    return customer_profiles_table


# Function to generate a table of terminal profiles with random data
def generate_terminal_profiles_table(n_terminals, random_state=0):
    np.random.seed(random_state)  # Set random seed for reproducibility
    terminal_id_properties = []

    # Generate random properties for each terminal
    for terminal_id in range(n_terminals):
        x_terminal_id = np.random.uniform(0, 100)  # Random x-coordinate
        y_terminal_id = np.random.uniform(0, 100)  # Random y-coordinate

        terminal_id_properties.append([terminal_id, x_terminal_id, y_terminal_id])

    # Create a DataFrame with the generated data
    terminal_profiles_table = pd.DataFrame(terminal_id_properties, columns=['TERMINAL_ID', 'x_terminal_id', 'y_terminal_id'])
    return terminal_profiles_table


# Function to find terminals within a specified radius of a customer
def get_list_terminals_within_radius(customer_profile, x_y_terminals, r):
    # Customer location as a numpy array
    x_y_customer = customer_profile[['x_customer_id', 'y_customer_id']].values.astype(float)

    # Compute squared differences in coordinates
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)

    # Compute Euclidean distances
    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))

    # Identify terminals within the radius
    available_terminals = list(np.where(dist_x_y < r)[0])

    # Return the list of terminal IDs
    return available_terminals


# Function to generate a table of transactions for a given customer over a specified period
def generate_transactions_table(customer_profile, start_date="2018-04-01", nb_days=10):
    customer_transactions = []
    random.seed(int(customer_profile.CUSTOMER_ID))  # Seed based on customer ID for reproducibility
    np.random.seed(int(customer_profile.CUSTOMER_ID))

    # Loop through each day in the specified period
    for day in range(nb_days):
        # Randomly determine the number of transactions for the day
        nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)

        if nb_tx > 0:  # Generate transactions if count is positive
            for tx in range(nb_tx):
                # Generate a random transaction time (close to noon)
                time_tx = int(np.random.normal(86400 / 2, 20000))

                if 0 < time_tx < 86400:  # Validate the transaction time
                    # Generate a random transaction amount
                    amount = np.random.normal(customer_profile.mean_amount, customer_profile.std_amount)
                    if amount < 0:  # Ensure amount is non-negative
                        amount = np.random.uniform(0, customer_profile.mean_amount * 2)
                    amount = np.round(amount, decimals=2)

                    # Randomly assign a terminal ID from available terminals
                    if len(customer_profile.available_terminals) > 0:
                        terminal_id = random.choice(customer_profile.available_terminals)
                        customer_transactions.append([time_tx + day * 86400, day, customer_profile.CUSTOMER_ID, terminal_id, amount])

    # Create a DataFrame for the transactions
    customer_transactions = pd.DataFrame(customer_transactions, columns=['TX_TIME_SECONDS', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT'])
    if len(customer_transactions) > 0:
        # Add datetime column to the DataFrame
        customer_transactions['TX_DATETIME'] = pd.to_datetime(customer_transactions["TX_TIME_SECONDS"], unit='s', origin=start_date)
        customer_transactions = customer_transactions[['TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']]
    return customer_transactions

# Function that generate a set of synthesized data based on the number of terminals, customers and days
def generate_dataset(n_customers=10000, n_terminals=1000000, nb_days=90, start_date="2018-04-01", r=5):
    # Generate a dataset of transactions, including customer and terminal profiles, and link them based on proximity.

    # Input:
    #     n_customers (int): Number of customers to generate.
    #     n_terminals (int): Number of terminals to generate.
    #     nb_days (int): Number of days over which transactions occur.
    #     start_date (str): Start date of the transaction period.
    #     r (float): Radius within which a terminal is available for a customer.

    # Output:
    #     tuple: DataFrames for customer profiles, terminal profiles, and transactions.
        
    start_time = time.time()
    # Generate customer profiles with random attributes
    customer_profiles_table = generate_customer_profiles_table(n_customers, random_state=0)
    print("Time to generate customer profiles table: {0:.2f}s".format(time.time() - start_time))

    start_time = time.time()
    # Generate terminal profiles with random coordinates
    terminal_profiles_table = generate_terminal_profiles_table(n_terminals, random_state=1)
    print("Time to generate terminal profiles table: {0:.2f}s".format(time.time() - start_time))

    start_time = time.time()
    # Calculate available terminals for each customer based on proximity
    x_y_terminals = terminal_profiles_table[['x_terminal_id', 'y_terminal_id']].values.astype(float)
    customer_profiles_table['available_terminals'] = customer_profiles_table.apply(
        lambda x: get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r), axis=1)
    customer_profiles_table['nb_terminals'] = customer_profiles_table.available_terminals.apply(len)
    print("Time to associate terminals to customers: {0:.2f}s".format(time.time() - start_time))

    start_time = time.time()
    # Generate transactions for each customer over the specified number of days
    transactions_df = customer_profiles_table.groupby('CUSTOMER_ID').apply(
        lambda x: generate_transactions_table(x.iloc[0], nb_days=nb_days)).reset_index(drop=True)
    print("Time to generate transactions: {0:.2f}s".format(time.time() - start_time))

    # Sort transactions chronologically and reset indices
    transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)
    transactions_df.reset_index(inplace=True)
    transactions_df.rename(columns={'index': 'TRANSACTION_ID'}, inplace=True)

    return customer_profiles_table, terminal_profiles_table, transactions_df

# After the dataset is generated, this function will add a series of fraud transactions on given 3 scenarios
def add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df):
    # Input:
    #     customer_profiles_table (DataFrame): Customer profile data.
    #     terminal_profiles_table (DataFrame): Terminal profile data.
    #     transactions_df (DataFrame): Transactions data.

    # Output:
    #     DataFrame: Updated transactions DataFrame with fraud labels and scenarios.
    
    # Initialize fraud labels and scenarios
    transactions_df['TX_FRAUD'] = 0
    transactions_df['TX_FRAUD_SCENARIO'] = 0

    # Scenario 1: Fraudulent transactions based on high transaction amounts
    transactions_df.loc[transactions_df.TX_AMOUNT > 220, 'TX_FRAUD'] = 1
    transactions_df.loc[transactions_df.TX_AMOUNT > 220, 'TX_FRAUD_SCENARIO'] = 1
    nb_frauds_scenario_1 = transactions_df.TX_FRAUD.sum()
    print("Number of frauds from scenario 1: ", nb_frauds_scenario_1)

    # Scenario 2: Compromised terminals cause fraud over a 28-day period
    for day in range(transactions_df.TX_TIME_DAYS.max()):
        compromised_terminals = terminal_profiles_table.TERMINAL_ID.sample(n=2, random_state=day)
        compromised_transactions = transactions_df[
            (transactions_df.TX_TIME_DAYS >= day) &
            (transactions_df.TX_TIME_DAYS < day + 28) &
            (transactions_df.TERMINAL_ID.isin(compromised_terminals))
        ]
        transactions_df.loc[compromised_transactions.index, 'TX_FRAUD'] = 1
        transactions_df.loc[compromised_transactions.index, 'TX_FRAUD_SCENARIO'] = 2

    nb_frauds_scenario_2 = transactions_df.TX_FRAUD.sum() - nb_frauds_scenario_1
    print("Number of frauds from scenario 2: ", nb_frauds_scenario_2)

    # Scenario 3: Compromised customers perform fraud, inflating transaction amounts
    for day in range(transactions_df.TX_TIME_DAYS.max()):
        compromised_customers = customer_profiles_table.CUSTOMER_ID.sample(n=3, random_state=day).values
        compromised_transactions = transactions_df[
            (transactions_df.TX_TIME_DAYS >= day) &
            (transactions_df.TX_TIME_DAYS < day + 14) &
            (transactions_df.CUSTOMER_ID.isin(compromised_customers))
        ]
        nb_compromised_transactions = len(compromised_transactions)
        random.seed(day)
        index_frauds = random.sample(list(compromised_transactions.index.values), k=int(nb_compromised_transactions / 3))
        transactions_df.loc[index_frauds, 'TX_AMOUNT'] *= 5
        transactions_df.loc[index_frauds, 'TX_FRAUD'] = 1
        transactions_df.loc[index_frauds, 'TX_FRAUD_SCENARIO'] = 3

    nb_frauds_scenario_3 = transactions_df.TX_FRAUD.sum() - nb_frauds_scenario_2 - nb_frauds_scenario_1
    print("Number of frauds from scenario 3: ", nb_frauds_scenario_3)

    return transactions_df

# Compute statistics on transactions and fraud occurrences.
def get_stats(transactions_df):
    nb_tx_per_day = transactions_df.groupby(['TX_TIME_DAYS'])['CUSTOMER_ID'].count()
    nb_fraud_per_day = transactions_df.groupby(['TX_TIME_DAYS'])['TX_FRAUD'].sum()
    nb_fraudcard_per_day = transactions_df[transactions_df['TX_FRAUD'] > 0].groupby(['TX_TIME_DAYS']).CUSTOMER_ID.nunique()
    return nb_tx_per_day, nb_fraud_per_day, nb_fraudcard_per_day


if __name__ == "__main__":
    # Generate dataset
    (customer_profiles_table, terminal_profiles_table, transactions_df) = generate_dataset(
        n_customers=2000, n_terminals=1000, nb_days=180, start_date="2018-04-01", r=3)

    # Add frauds to the dataset
    start_time = time.time()
    transactions_df = add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df)
    print(f"Time taken to add frauds: {time.time() - start_time:.2f} seconds")

    # Print dataset summary
    print("Percentage of fraud: ", transactions_df.TX_FRAUD.mean())
    print("Number of fraud: ", transactions_df.TX_FRAUD.sum())
    print("Top transactions: \n", transactions_df.head())
    print("Scenario 1: ", transactions_df[transactions_df.TX_FRAUD_SCENARIO == 1].shape)
    print("Scenario 2: ", transactions_df[transactions_df.TX_FRAUD_SCENARIO == 2].shape)
    print("Scenario 3: ", transactions_df[transactions_df.TX_FRAUD_SCENARIO == 3].shape)

    # Compute statistics and save to CSV
    (nb_tx_per_day, nb_fraud_per_day, nb_fraudcard_per_day) = get_stats(transactions_df)
    n_days = len(nb_tx_per_day)
    tx_stats = pd.DataFrame({"value": pd.concat([nb_tx_per_day / 50, nb_fraud_per_day, nb_fraudcard_per_day])})
    tx_stats['stat_type'] = ["nb_tx_per_day"] * n_days + ["nb_fraud_per_day"] * n_days + ["nb_fraudcard_per_day"] * n_days
    tx_stats = tx_stats.reset_index()

    # Save profiles and transactions to CSV files
    customer_profiles_table.to_csv('2000C_data/customer_profiles.csv', index=False)
    terminal_profiles_table.to_csv('2000C_data/terminal_profiles.csv', index=False)
    transactions_df.to_csv('2000C_data/transactions.csv', index=False)
    tx_stats.to_csv('transaction_statistics.csv', index=False)

    print("Data saved to CSV files.")
