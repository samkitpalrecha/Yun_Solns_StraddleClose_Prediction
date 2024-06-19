import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans
import joblib
import pickle


df_dict = {}
indices = ['NIFTY', 'FINNIFTY', 'SENSEX', 'BANKNIFTY', 'MIDCPNIFTY', 'BANKEX']

for index in indices:
    df = pd.read_csv(f'Data/{index}.csv')
    df_dict[index] = df


"""# Function for Creating Train and Test DataFrames out of the Df"""


def create_train_test_data(df, dte_start, index):
    # feature df
    df_feature = df.drop(columns=[f'close_dte{dte_start}'])

    # Extract features as NumPy array
    X = df_feature.to_numpy()

    # Standardize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    df_features_scaled = pd.DataFrame(X, columns=df_feature.columns)

    # target df
    df_target = df[f'close_dte{dte_start}']

    df = pd.concat([df_features_scaled, df_target], axis=1)
    X = df.drop(f'close_dte{dte_start}', axis=1)  # All columns except target
    y = df[f'close_dte{dte_start}']

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with open(f'Scalers/scaler{index}_dte{dte_start}.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return X_train, X_test, y_train, y_test


"""# Function for Training and Predicting with K-Means Algorithm"""


def train_k_means(X_train, X_test, num_clusters, index, dte):

    kmeans = KMeans(n_clusters=num_clusters)

    kmeans.fit(X_train)

    cluster_labels = kmeans.labels_

    # Predict the cluster label for the new vector
    predicted_cluster = kmeans.predict(X_test)

    X_train['cluster'] = cluster_labels
    X_test['cluster'] = predicted_cluster

    joblib.dump(kmeans, f'K_Means/kmeans_{index}_dte{dte}.pkl')

    return X_train, X_test


"""# Function for Restructoring the Data into Right Form"""


def structure_data(X_train, X_test, y_train, y_test, dte_start):
    # Get a list of unique cluster labels
    unique_clusters_train = X_train['cluster'].unique()

    # Create an empty dictionary to store DataFrames for each cluster
    X_train_clusters = {}

    # Loop through each unique cluster label
    for cluster in unique_clusters_train:
      # Filter the DataFrame to rows belonging to the current cluster
      X_train_clusters[cluster] = X_train[X_train['cluster'] == cluster]
      X_train_clusters[cluster] = X_train_clusters[cluster][f'open_dte{dte_start}']

    # Get a list of unique cluster labels
    unique_clusters_test = X_test['cluster'].unique()

    # Create an empty dictionary to store DataFrames for each cluster
    X_test_clusters = {}

    # Loop through each unique cluster label
    for cluster in unique_clusters_test:
      # Filter the DataFrame to rows belonging to the current cluster
      X_test_clusters[cluster] = X_test[X_test['cluster'] == cluster]
      X_test_clusters[cluster] = X_test_clusters[cluster][f'open_dte{dte_start}']

    y_train_clusters = {}

    for cluster in unique_clusters_train:
       index = []
       for index_x in X_train_clusters[cluster].index:
           for index_y in y_train.index:
              if index_x == index_y:
                 index.append(index_x)
       y_train_clusters[cluster] = y_train[index]
       y_train_clusters[cluster] = y_train_clusters[cluster].astype("float32")

    y_test_clusters = {}

    for cluster in unique_clusters_test:
       index = []
       for index_x in X_test_clusters[cluster].index:
           for index_y in y_test.index:
            if index_x == index_y:
                index.append(index_x)
       y_test_clusters[cluster] = y_test[index]
       y_test_clusters[cluster] = y_test_clusters[cluster].astype("float32")

    return X_train_clusters, X_test_clusters, y_train_clusters, y_test_clusters


"""# Function for Training and Testing our Model on Each Cluster of Data"""


def model(X_train,  y_train, lr, epochs, l2r, index, dte, cluster):

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    X_train = X_train.reshape(-1, 1)

    # Define the model
    model_nn = keras.Sequential([
        keras.layers.Dense(units=1024, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(units=512, activation='relu', kernel_regularizer=l2(l2r)),
        keras.layers.Dense(units=256, activation='relu', kernel_regularizer=l2(l2r)),
        keras.layers.Dense(units=128, activation='relu', kernel_regularizer=l2(l2r)),
        keras.layers.Dense(units=64, activation='relu', kernel_regularizer=l2(l2r)),
        keras.layers.Dense(units=32, activation='relu', kernel_regularizer=l2(l2r)),
        keras.layers.Dense(units=8, activation='relu', kernel_regularizer=l2(l2r)),
        keras.layers.Dense(units=1)
    ])

    # Define optimizer with learning rate scheduler
    optimizer = Adam(learning_rate=lr)

    # Compile the model
    model_nn.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    # Train the model
    model_nn.fit(X_train, y_train, epochs=epochs, batch_size=1)
    model_nn.save(f'Neural_Nets/Neural_Net_{index}_dte{dte}_{cluster}.keras')

    return model_nn

def predict(model_nn, X_test, y_test):
    
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    X_test = X_test.reshape(-1, 1)

    # Make predictions on test data
    y_pred_nn = model_nn.predict(X_test)

    y_pred_nn = y_pred_nn.flatten()

    # Evaluate the model's performance
    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    # print("Mean Absolute Error (NN):", mae_nn)
    rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
    # print("Root Mean Squared Error (NN):", rmse_nn)
    r2_nn = r2_score(y_test, y_pred_nn)
    # print("R-squared (NN):", r2_nn)

    prediction_plus_metrics = [y_pred_nn, mae_nn, rmse_nn, r2_nn]

    return prediction_plus_metrics

num_clusters = 3

start_dtes = [1,2,3,4]

hyp_dict = {}
hyp_dict['NIFTY'] = {'learning_rate': 5.144475412729949e-05, 'epochs': 200, 'l2_regularization': 0.00011455292256091957}
hyp_dict['FINNIFTY'] = {'learning_rate': 0.0002464337661335635, 'epochs': 350, 'l2_regularization': 0.0005913099708706237}
hyp_dict['BANKNIFTY'] = {'learning_rate': 0.0002210896804967545, 'epochs': 250, 'l2_regularization': 0.00011700255002918199}
hyp_dict['SENSEX'] = {'learning_rate': 0.0001, 'epochs': 300, 'l2_regularization': 0.1}
hyp_dict['MIDCPNIFTY'] = {'learning_rate': 0.0001, 'epochs': 300, 'l2_regularization': 0.1}
hyp_dict['BANKEX'] = {'learning_rate': 0.0001, 'epochs': 300, 'l2_regularization': 0.1}



for index in indices:
    
    df = df_dict[index]

    for dte in start_dtes:
        
        df_ = df[[f'open_dte{dte}', f'close_dte{dte}', f'open_dte{dte+1}', f'close_dte{dte+1}', f'vix_dte{dte}', f'vix_dte{dte+1}']]

        X_train, X_test, y_train, y_test = create_train_test_data(df_, dte, index=index)

        X_train, X_test = train_k_means(X_train, X_test, num_clusters, index=index, dte=dte)

        # cluster_counts = X_train['cluster'].value_counts()
        # size = len(X_train['cluster'])

        X_train_clusters, X_test_clusters, y_train_clusters, y_test_clusters = structure_data(X_train, X_test, y_train, y_test, dte)

        y_pred_clusters = {}

        hyp = hyp_dict[index]
        
        model_nn = model(X_train_clusters[0],  y_train_clusters[0], lr=hyp['learning_rate'], epochs=hyp['epochs'], l2r=hyp['l2_regularization'], index=index, dte=dte, cluster=0)
        if 0 in X_test_clusters:
          prediction_plus_metrics = predict(model_nn, X_test_clusters[0], y_test_clusters[0])
          y_pred_clusters[0] = prediction_plus_metrics

        model_nn = model(X_train_clusters[1],  y_train_clusters[1], lr=hyp['learning_rate'], epochs=hyp['epochs'], l2r=hyp['l2_regularization'], index=index, dte=dte, cluster=1)
        if 1 in X_test_clusters:
          prediction_plus_metrics = predict(model_nn, X_test_clusters[1], y_test_clusters[1])
          y_pred_clusters[1] = prediction_plus_metrics

        model_nn = model(X_train_clusters[2],  y_train_clusters[2], lr=hyp['learning_rate'], epochs=hyp['epochs'], l2r=hyp['l2_regularization'], index=index, dte=dte, cluster=2)
        if 2 in X_test_clusters:
          prediction_plus_metrics = predict(model_nn, X_test_clusters[2], y_test_clusters[2])
          y_pred_clusters[2] = prediction_plus_metrics