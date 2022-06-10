#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 19:06:09 2022

@author: curro
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def build_model(name, window_size, num_features):
    if name == 'CNN_LSTM':
        return get_CNN_LSTM(window_size, num_features)
    elif name == 'Simpler_RNN':
        return get_Simpler_RNN(window_size, num_features)
    elif name == 'Simple_RNN':
        return get_Simple_RNN(window_size, num_features)
    elif name == 'LSTM':
        return get_LSTM(window_size, num_features)
    elif name == 'LSTM_stacked':
        return get_LSTM_stacked(window_size, num_features)
    elif name == 'Bidirectional_LSTM':
        return get_Bidirectional_LSTM(window_size, num_features)
    elif name == 'Simple_ANN':
        return get_Simple_ANN(window_size, num_features)
    
#%% Model architectures
def get_CNN_LSTM(window_size, num_features):
    # CNN + LSTM
    model = tf.keras.models.Sequential([
      # tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
      #                     input_shape=[None,1]),
      tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                          strides=1, padding="causal",
                          activation="relu", input_shape=[window_size,num_features]),
      tf.keras.layers.LSTM(64, return_sequences=True),
      tf.keras.layers.LSTM(64, return_sequences=False),
      tf.keras.layers.Dense(30, activation="relu"),
      tf.keras.layers.Dense(10, activation="relu"),
      tf.keras.layers.Dense(1, activation = 'relu'),
    ])
    return model

def get_Simple_RNN(window_size, num_features):
    # Simple RNN
    model = tf.keras.models.Sequential([
      # tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                          # input_shape=[None]),
      tf.keras.layers.SimpleRNN(40, return_sequences=True, input_shape=[window_size,num_features]),
      tf.keras.layers.SimpleRNN(40),
      tf.keras.layers.Dense(1, activation = 'relu'),
    ])
    
    # =============================================================================
    #Simpler RNN (YT)
    # model = tf.keras.models.Sequential([
    #   tf.keras.layers.SimpleRNN(64, activation='relu', input_shape = [window_size, num_features]),
    #   tf.keras.layers.Dense(1),
    # ])
    # =============================================================================
    return model

def get_Simpler_RNN(window_size, num_features):
    # Simpler RNN (YT)
    model = tf.keras.models.Sequential([
      tf.keras.layers.SimpleRNN(64, activation='relu', input_shape = [window_size, num_features]),
      tf.keras.layers.Dense(1,  activation = 'relu'),
    ])
    return model

def get_LSTM(window_size, num_features):
    # LSTM 1 * 64
    model = tf.keras.models.Sequential([
      tf.keras.layers.LSTM(64, return_sequences=False, input_shape = [window_size, num_features]),
      tf.keras.layers.Dense(1, activation = 'relu')
      ])
    return model

def get_LSTM_stacked(window_size, num_features):
    # LSTM 2 * 32
    model = tf.keras.models.Sequential([
      tf.keras.layers.LSTM(32, return_sequences=True, input_shape = [window_size, num_features]),
      tf.keras.layers.LSTM(32),
      tf.keras.layers.Dense(1),
      ])
    return model

def get_Bidirectional_LSTM(window_size, num_features):
    # Bidirectional LSTM
    model = tf.keras.models.Sequential([
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True), input_shape = [window_size, num_features]),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
      tf.keras.layers.Dense(1, activation = 'relu'),
    ])  
    return model

def get_Simple_ANN(window_size, num_features):
    # ANN single 
    reshape = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1),
                         input_shape=[window_size,num_features])
    l0 = tf.keras.layers.Dense(1, activation = 'relu')
    model = tf.keras.models.Sequential([reshape,
                                        l0])
    # Post training: print("Layer weights {}".format(l0.get_weights()))    
    return model


#%% Configurate callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

def configure_callbacks(early_stopping = False,patience=100, checkpoint= False, model_name= "Model.h5", lr_scheduler=False ):
    callbacks = []
    if checkpoint == True:
        checkpoint_filepath = '/Model_checkpoints/' + model_name + '.h5'
    
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='loss',
            mode='min',
            save_best_only=True)
        callbacks.append(model_checkpoint_callback)

    if early_stopping == True:
        callbacks.append(EarlyStopping(patience=patience, monitor='loss'))
        
    return callbacks


#%% Run inferences
def predict(model,generator,scaler, num_features):
    # Run inferences on dataset generator
    predictions = model.predict(generator)
    # Rescale predicted values
    predict_copies = np.repeat(predictions, num_features ,axis=-1)
    predictions = scaler.inverse_transform(predict_copies)[:,0]

    return predictions

#%% Run forecasting inferences
def test_forecast(test_Y_gt, x_test, model, scaler, window_size, num_features, future, num_inferences, start_all, verbose=0 ):
    start = start_all
    for i in range(num_inferences):
        prediction=[]
        current_batch = x_test[start : start + window_size]
        current_batch = current_batch.reshape(1, window_size, num_features) #Reshape
        
        # Predict future
        for j in range(future):
            current_pred = model.predict(current_batch)[0]
            prediction.append(current_pred)
            current_batch = np.append(current_batch[:,1:,:],[[x_test[start + window_size + j]]],axis=1)
            #current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
            current_batch[:,-1,0] = current_pred
        
        # Inverse transform to before scaling so we get actual values
        predict_copies = np.repeat(prediction, num_features + 1 , axis=-1)
        rescaled_prediction = scaler.inverse_transform(predict_copies)[:,0]

        
        if verbose == 1:
            plt.figure()
            plt.plot(x_test[start + window_size: start + window_size + future])
            plt.plot(rescaled_prediction)
            plt.show()
    
        if i == 0:
            all_predictions = rescaled_prediction
        else:
            all_predictions = np.concatenate((all_predictions,rescaled_prediction))
        
        
        start = start + future
    
    
    plt.figure(figsize=(12,6), dpi=100)
    plt.plot(test_Y_gt[start_all + window_size: start_all + window_size + future*num_inferences].array)
    plt.plot(all_predictions)
    plt.title('Predictions using 24h horizon')
    plt.legend(['Ground truth', 'Predictions'])
    plt.show()


    

    
    



