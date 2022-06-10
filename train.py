#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 17:59:27 2022

@author: curro
"""

import os
import csv
import math
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import TensorBoard

from model_utils import test_forecast
from plot_utils import  plot_training
from model_utils import build_model, predict

# Plot settings
plt.style.use('seaborn')
plt.rcParams["figure.figsize"] = (16, 8)
pd.options.plotting.backend = "plotly"


def parser_opt():
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument('-m','--model_name', type=str, required=True, help='Models available: CNN_LSTM,Simple_RNN,LSTM,LSTM_stacked,Bidirectional_LSTM,Simple_ANN')
    parser.add_argument('-w','--window_size', type=int, required=True, help='Input sequence length, 96 or 168 values (4-7 days)')
    parser.add_argument('-e','--epochs', type=int, required=True, help = 'Number of training epochs')
    parser.add_argument('-b','--batch_size', type=int, required=False, default = 16, help = 'Batch size for training' )

    opt = parser.parse_args()
    return opt

def main(opt):
    # Input args
    model_name = opt.model_name 
    window_size = opt.window_size
    epochs = opt.epochs
    batch_size = opt.batch_size

    # Load the dataset
    file_path = "data_processing/processed_data/multivariate_data_filtered.json"
    df = pd.read_json(file_path)

    times = df.filter(['Index'])

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    df_scaled = scaler.fit_transform(df)

    # Get the output and input columns
    y_data = df_scaled[:,0]
    x_data = df_scaled[:,:-1] # drop desviation column (unknown)

    #%% Split training set
    split_time = int(len(df_scaled) * 0.66)

    x_train = x_data[:split_time] 
    y_train = y_data[:split_time]

    x_test = x_data[split_time:] 
    y_test = y_data[split_time:]

    # Create the data generators for training and test
    num_features = x_data.shape[1]

    train_generator = TimeseriesGenerator(x_train,
                                        y_train,
                                        length=window_size,
                                        batch_size=batch_size)
    validation_generator = TimeseriesGenerator(x_test,
                                            y_test,
                                            length=window_size,
                                            batch_size=batch_size)

    # Delete previous session to obtain static results
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)

    # Build the model
    model = build_model(model_name,
                    window_size,
                    num_features)

    # Create new dir to save results
    try:
        os.mkdir(f'Results/train_{model_name}_ws{window_size}_epochs{epochs}_results')
    except:
        pass

    # Configure Tensorboard callback
    callbacks = []
    callbacks.append(TensorBoard(
        log_dir=f"Results/train_{model_name}_ws{window_size}_epochs{epochs}_results/tf_logs",
        histogram_freq=0,  # How often to log histogram visualizations
        embeddings_freq=0,  # How often to log embedding visualizations
        update_freq="epoch",
    ))

    # Print model summary
    model.summary()

    # Compilate the model  
    optimizer = 'adam'
    metrics = ['acc']
    model.compile(loss=tf.keras.losses.Huber(),
                optimizer=optimizer,
                metrics=metrics)

    # Train model
    history = model.fit_generator(generator=train_generator,
                                verbose=1,
                                epochs=epochs,
                                validation_data=validation_generator,
                                callbacks = callbacks)

    # Save the trained model 
    model.save(f'Results/train_{model_name}_ws{window_size}_epochs{epochs}_results/{model_name}_ws{window_size}_epochs{epochs}_model.h5')

    # Plot the training and validation accuracy and loss at each epoch
    plot_training(history)
    plt.savefig(f'Results/train_{model_name}_ws{window_size}_epochs{epochs}_results/history.png')

    # Run predictions on training and validation set
    train_predict = predict(model,
                            train_generator,
                            scaler, 
                            num_features = df_scaled.shape[1])

    test_predict = predict(model,
                        validation_generator,
                        scaler,
                        num_features = df_scaled.shape[1])


    # Get ground truth values
    y_gt = df.iloc[:,0]
    # Split train and validation ground truth
    train_Y_gt = y_gt[:split_time]
    test_Y_gt = y_gt[split_time:]

    # Plot predictions on train data
    df_train_predict = pd.DataFrame(train_predict, index = times.index[window_size:split_time])
    plt.figure()
    plt.plot(train_Y_gt)
    plt.plot(df_train_predict)
    plt.title('Predictions on train')
    plt.legend(['Ground truth', 'Predictions'])
    plt.savefig(f'Results/train_{model_name}_ws{window_size}_epochs{epochs}_results/pred_train.png')

    # Plot predictions on test data
    df_test_predict = pd.DataFrame(test_predict, index = times.index[split_time + window_size:])
    plt.figure()
    plt.plot(test_Y_gt)
    plt.plot(df_test_predict)
    plt.title('Predictions on test')
    plt.legend(['Ground truth', 'Predictions'])
    plt.savefig(f'Results/train_{model_name}_ws{window_size}_epochs{epochs}_results/pred_test.png')

    # Plot zoomed train and test data
    start = 50
    end = 100
    plt.figure()
    plt.plot(train_Y_gt[window_size + start: window_size + end])
    plt.plot(df_train_predict.iloc[start:end])
    plt.legend(['Ground truth', 'Predictions'])
    plt.title('Predictions on train (zoomed sample)')
    plt.savefig(f'Results/train_{model_name}_ws{window_size}_epochs{epochs}_results/pred_train_zoom.png')

    start = 50
    end = 100
    plt.figure()
    plt.plot(test_Y_gt[window_size + start: window_size + end])
    plt.plot(df_test_predict.iloc[start:end])
    plt.legend(['Ground truth', 'Predictions'])
    plt.title('Predictions on test (zoomed sample)')
    plt.savefig(f'Results/train_{model_name}_ws{window_size}_epochs{epochs}_results/pred_test_zoom.png')

    # Validate the model using RMSE and MAE
    trainScore = math.sqrt(mean_squared_error(df_train_predict , train_Y_gt[window_size:]))
    testScore = math.sqrt(mean_squared_error(df_test_predict , test_Y_gt[window_size:]))
    print('Train Score: %.2f RMSE' % (trainScore))
    print('Test Score: %.2f RMSE' % (testScore))

    train_mae = mean_absolute_error(df_train_predict , train_Y_gt[window_size:])
    test_mae = mean_absolute_error(df_test_predict , test_Y_gt[window_size:])
    print('Train Score: %.2f MAE' % (train_mae))
    print('Test Score: %.2f MAE' % (test_mae))

    # Write validation results on csv
    csv_file = open(f'Results/train_{model_name}_ws{window_size}_epochs{epochs}_results/validation_metrics.txt', 'w')
    csv_writer = csv.writer(csv_file, delimiter = '\n')
    csv_writer.writerow(['Train Score: %.2f RMSE' % (trainScore), 'Test Score: %.2f RMSE' % (testScore)])
    csv_writer.writerow(['Train Score: %.2f MAE' % (train_mae), 'Test Score: %.2f MAE' % (test_mae)])
    csv_file.close()


if __name__ =='__main__':
    opt = parser_opt()
    main(opt)