#!/usr/bin/env python

import numpy as np
import keras
from keras import Sequential
from keras.layers import Conv2D, Dropout, UpSampling2D, Activation, Flatten, Dense, Reshape
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();

def my_data():
	tr = np.load('train.npy')
	te = np.load('test.npy')
	va = np.load('valid.npy')
	return (tr[:1000], tr[:1000]), (va, va), (te, te)

def my_model(train_data,activ, nstart,repeat_filts,filt_size,add_reductive,nfilters,dropout,regularization,model_name,lr, hidden_space):
    from keras import backend as K; K.clear_session()
    sess = tf.Session()
    model = Sequential()
    model_shape = []
    for i_step, step in enumerate(range(nstart,int(np.log(train_data[0].shape[1])/np.log(2)))[::-1]):
        for repeat_filt in range(repeat_filts):
            model.add(Conv2D(nfilters, filt_size, strides=(1,1), padding='same',input_shape=list(train_data[0][0].shape),kernel_regularizer=l2(regularization)))
            model.add(Dropout(dropout)); model.add(Activation(activ))
            model_shape.append(model.output_shape)
        if add_reductive == True:
            model.add(Conv2D(nfilters, filt_size, strides=(2,2), padding='same',input_shape=list(train_data[0][0].shape),kernel_regularizer=l2(regularization)))
            model.add(Dropout(dropout)); model.add(Activation(activ))
            model_shape.append(model.output_shape)
    model.add(Flatten())
    model_shape.append( model.output_shape)
    model.add(Dense(hidden_space))
    model_shape.append(model.output_shape)
    decoder_shapes = model_shape[::-1][1:]
    # model.add(Activation(activ))
    model.add(Dense(decoder_shapes[0][1]))
    model.add(Reshape((decoder_shapes[1][1:])))
    counter = 0; max_layers = len(range(nstart,int(np.log(train_data[0].shape[1])/np.log(2))))*repeat_filts
    for i_step, step in enumerate(range(nstart,int(np.log(train_data[0].shape[1])/np.log(2)))[::-1]):
        if add_reductive == True:
            model.add(Conv2D(nfilters, filt_size, strides=(1,1), padding='same',input_shape=list(train_data[0][0].shape),kernel_regularizer=l2(regularization)))
            model.add(Dropout(dropout)); model.add(Activation(activ))
            model_shape.append(model.output_shape)
            model.add(UpSampling2D(size=(2,2)))
        for repeat_filt in range(repeat_filts):
            counter+=1
            if counter==max_layers:
                break
            model.add(Conv2D(nfilters, filt_size, strides=(1,1), padding='same',input_shape=list(train_data[0][0].shape),kernel_regularizer=l2(regularization)))
            model.add(Dropout(dropout)); model.add(Activation(activ))
            model_shape.append(model.output_shape)
    model.add(Conv2D(model.input_shape[-1], filt_size, strides=(1,1), padding='same',input_shape=list(train_data[0][0].shape),kernel_regularizer=l2(regularization)))
    model.add(Dropout(dropout))
    model.add(Activation(activ))
    checkpoint = ModelCheckpoint('model_{0}.hdf5'.format(model_name), monitor='val_acc', verbose=1, save_best_only=False, mode='auto')
    callbacks_list = [checkpoint, PlotLosses()]
    opt = keras.optimizers.adam(lr=1e-4)
    model_serial = model
    #model_serial.compile(loss= K.sum(K.square(model.output - model.input)) , optimizer=opt, metrics=None)
    #model_serial.compile(loss= custom_loss , optimizer=opt, metrics=None)
    model_serial.compile(loss= 'mean_squared_error' , optimizer=opt, metrics=None)
    model_serial.summary()
    return model_serial

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
hyper_params = {
    "data":{},
    "model": {
        "repeat_filts": hp.choice("m_repeat_filts", (2,3,4)),
        "lr": hp.loguniform("m_lr", np.log(1e-4), np.log(1e-2)), # 0.0001 - 0.01
        "activ": hp.choice("m_activ", ('relu','selu')),
        "add_reductive": hp.choice("m_add_reductive", (True,False)),
        "filt_size": hp.choice("m_filt_size", ((2,2),(3,3))),
        "nstart": hp.choice("m_nstart", (1,2)),
        "nfilters": hp.choice("m_nfilters", (16, 32)),
        "hidden_space":hp.choice("m_hidden_space", (2,4,8,16)),
        "dropout": hp.choice("m_dropout", (0.0, 0.2,0.4)),
        "regularization": hp.choice("m_regularization", (1e-4, 1e-3,1e-2)),
        "model_name": "m_my_test_model",
        },
    "fit": {
        "epochs": 200,
        "patience": 10,
        "batch_size": 32,
    }
}

from kopt import CompileFN, KMongoTrials, test_fn
import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging
#keras_model.fit(train[0],train[1])
objective = CompileFN(db_name="mydb", exp_name="motif_initialization",  # experiment name
                      data_fn=my_data,
                      model_fn=my_model,
                      add_eval_metrics=["mse"], # metrics from concise.eval_metrics, you can also use your own
                      optim_metric="mse", # which metric to optimize for
                      optim_metric_mode="min", # maximum should be searched for
                      valid_split=None, # use valid from the data function
                      #cv_b_folds=None,
                      save_model='best', # checkpoint the best model
                      save_results=True, # save the results as .json (in addition to mongoDB)
                      save_dir="./saved_models_vae/")  # place to store the models
#data = get_data(objective.data_fn, hyper_params)
#test_fn(objective, hyper_params)
trials = Trials()
best = fmin(objective, hyper_params, trials=trials, algo=tpe.suggest, max_evals=100)