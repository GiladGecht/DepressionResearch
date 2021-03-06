# -*- coding: utf-8 -*-
"""Keras_Sequence_depression_classifier_CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TIrRmuU0HZvuMQ5YdykMsIxeOvEwtVeH
"""


# =============== IMPORTS =============== #
# TODO: Remove prints and change to python's logging library

import re
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Create_Data.UtilFunctions as utils

from keras.optimizers import Adam
from keras.models import Sequential
from Create_Data.Logging import Logger
from keras.preprocessing.text import Tokenizer
from keras_models.preprocessing import Preprocessing
from Create_Data.UtilFunctions import confusion_matrix, classification_report
from keras_models.keras_util_functions import split_user_train_test, train_validation_split
from keras.layers import Embedding, Dropout, Dense, Flatten, Activation, Conv1D, MaxPooling1D

warnings.simplefilter("ignore")
filename = utils.os.path.basename(__file__)[:-3]
logger = Logger(filename=filename)

# =============== LOAD DATA =============== #
url1 = 'https://github.com/GiladGecht/DepressionResearch/raw/master/Data/depression_lstm.csv'  # 200 samples
url2 = 'https://github.com/GiladGecht/DepressionResearch/raw/master/Data/neutral_lstm_gen2.csv'  # 200 samples
url3 = 'https://github.com/GiladGecht/DepressionResearch/raw/master/Data/neutral_group_neutral_posts_gen2.csv'  # 2000 samples
url4 = 'https://github.com/GiladGecht/DepressionResearch/raw/master/Data/depression_lstm_gen2.csv'  # 2000 samples

depressed_neutral = pd.read_csv(url4, nrows=1000)
neutral_neutral = pd.read_csv(url3, nrows=1000)

# =============== DATA PRE-PROCESSING =============== #

class_map = {1: "DEPRESSION",
             0: "NEUTRAL"}

# remove special characters in the text & update the new length of each post
depressed_neutral['post_text'] = depressed_neutral['post_text'].apply(lambda x: re.sub('[^a-zA-Z.,!?]+', " ", x))
depressed_neutral['post_length_updated'] = depressed_neutral['post_text'].apply(lambda x: len(x))
neutral_neutral['post_text'] = neutral_neutral['post_text'].apply(lambda x: re.sub('[^a-zA-Z.,!?]+', " ", x))
neutral_neutral['post_length_updated'] = neutral_neutral['post_text'].apply(lambda x: len(x))

depressed_neutral_copy = depressed_neutral.copy()
neutral_neutral_copy = neutral_neutral.copy()

# Create the train and test so that each user appears either in the training set or the test set
depression_train, depression_test = split_user_train_test(depressed_neutral_copy, 0.8)
neutral_train, neutral_test       = split_user_train_test(neutral_neutral_copy, 0.8)
assert len(set(depression_train).intersection(set(depression_test))) == 0, "Depression Train and Test sets are not homogeneous"
assert len(set(neutral_train).intersection(set(neutral_test))) == 0, "Neutral Train and Test sets are not homogeneous"

'''
Collect the posts from all the users at each group
Create a single string comprised of these users's posts
'''
Depression_train = depressed_neutral_copy[depressed_neutral_copy['user_name'].isin(depression_train)]['post_text']
Depression_test  = depressed_neutral_copy[depressed_neutral_copy['user_name'].isin(depression_test)]['post_text']

Neutral_train = neutral_neutral_copy[neutral_neutral_copy['user_name'].isin(neutral_train)]['post_text']
Neutral_test  = neutral_neutral_copy[neutral_neutral_copy['user_name'].isin(neutral_test)]['post_text']

logger.log("Depression train shape: {}, Depression test shape:{}".format(Depression_train.shape, Depression_test.shape))
logger.log("Neutral train shape: {}, Neutral test shape:{}".format(Neutral_train.shape, Neutral_test.shape))

depressed_neutral_copy_cat = depressed_neutral_copy['post_text'].str.cat()
neutral_neutral_copy_cat   = neutral_neutral_copy['post_text'].str.cat()
Depression_train           = Depression_train.str.cat()
Depression_test            = Depression_test.str.cat()
Neutral_train              = Neutral_train.str.cat()
Neutral_test               = Neutral_test.str.cat()

logger.log("Neutral train text length: ", len(Neutral_train))
logger.log("Neutral test text length: ", len(Neutral_test))
logger.log("Depressed train text length: ", len(Depression_train))
logger.log("Depressed test text length: ", len(Depression_test))

val_scores, acc_scores = [], []
SEQUENCE_LENGTHS = [100]
BATCH_SIZE       = 4096
EMBEDDING_SIZE   = 150
EPOCHS           = 10
CVS              = 5
FLAG             = 1

depressed_train_name_list, depressed_valid_name_list = train_validation_split(train_names=depression_train,
                                                                              data=depressed_neutral_copy,
                                                                              CVS=CVS)

neutral_train_name_list, neutral_valid_name_list = train_validation_split(train_names=neutral_train,
                                                                          data=neutral_neutral_copy,
                                                                          CVS=CVS)
tokenizer = Tokenizer(char_level=True)

# =============== MODEL FITTING =============== #

for cv in range(CVS):

    Depression_train = depressed_neutral_copy[depressed_neutral_copy['user_name'].isin(depressed_train_name_list[cv])]['post_text']
    Depression_val   = depressed_neutral_copy[depressed_neutral_copy['user_name'].isin(depressed_valid_name_list[cv])]['post_text']
    Depression_test  = depressed_neutral_copy[depressed_neutral_copy['user_name'].isin(depression_test)]['post_text']

    Neutral_train = neutral_neutral_copy[neutral_neutral_copy['user_name'].isin(neutral_train_name_list[cv])]['post_text']
    Neutral_val   = neutral_neutral_copy[neutral_neutral_copy['user_name'].isin(neutral_valid_name_list[cv])]['post_text']
    Neutral_test  = neutral_neutral_copy[neutral_neutral_copy['user_name'].isin(neutral_test)]['post_text']

    Depression_train = Depression_train.str.cat()
    Depression_test  = Depression_test.str.cat()
    Depression_val   = Depression_val.str.cat()
    Neutral_train    = Neutral_train.str.cat()
    Neutral_test     = Neutral_test.str.cat()
    Neutral_val      = Neutral_val.str.cat()

    if flag:
        tokenizer.fit_on_texts(Depression_train + Neutral_train)
        flag = 0

    logger.log("Performing Fold #{}".format(cv))
    for seq_len in SEQUENCE_LENGTHS:
        preprocessing = Preprocessing(tokenizer, Neutral_train, Neutral_test, Neutral_val, Depression_train,
                                      Depression_test, Depression_val, seq_len)

        # Create the sets
        X_train, X_test, X_val, y_train, y_test, y_val = preprocessing.create_train_test_subsequences()

        # Combine them so we can shuffle and keep the targets corresponding to their sequence
        Train = np.hstack((X_train, y_train))
        Test = np.hstack((X_test, y_test))
        Val = np.hstack((X_val, y_val))
        np.random.shuffle(Train)
        np.random.shuffle(Test)
        np.random.shuffle(Val)

        # Split back the shuffled data
        X_train, y_train = Train[:, :-1], Train[:, -1]
        X_test, y_test = Test[:, :-1], Test[:, -1]
        X_val, y_val = Val[:, :-1], Val[:, -1]

        t1 = time.time()

        # Network

        # Convolution Layers
        model = Sequential()
        model.add(Embedding(len(tokenizer.word_index) + 1, EMBEDDING_SIZE, input_length=seq_len))
        model.add(Dropout(.8))
        model.add(Conv1D(filters=64, kernel_size=3))
        model.add(Activation("relu"))
        model.add(MaxPooling1D(pool_size=3))
        model.add(Dropout(.7))
        model.add(Conv1D(filters=64, kernel_size=3))
        model.add(Activation("relu"))
        model.add(Dropout(.6))
        model.add(Flatten())

        # Hidden Layers
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(Dropout(.4))
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(.2))

        # Output layer
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        # Fit
        history = model.fit(X_train,
                            y_train,
                            batch_size=BATCH_SIZE,
                            epochs=10,
                            verbose=True,
                            validation_data=(X_val, y_val))
        # Evaluate fold results on test sets
        score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=True)
        score_val = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=True)
        val_scores.append(score_val[1])
        acc_scores.append(score[1])
        logger.log("It took {:.2f} Seconds, Test Accuracy with SEQUENCE LENGTH={}: {}, Validation Accuracy: {}".format(
            time.time() - t1, seq_len, score[1], score_val[1]))

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel("Epoch")
    plt.savefig("Acc{}.png".format(cv))
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.savefig("Loss{}.png".format(cv))
    plt.legend()

    logger.log("-" * 30)
# =============== EVALUATION METRICS =============== #

y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
X_test_pred = np.column_stack((X_test, y_pred))
y_pred_output = [1 if p >= 0.5 else 0 for p in y_pred]
logger.log(confusion_matrix(y_pred=y_pred_output, y_true=y_test))
logger.log(classification_report(y_true=y_test, y_pred=y_pred_output))
logger.log(model.summary())

acc      = history.history['acc']
val_acc  = history.history['val_acc']
loss     = history.history['loss']
val_loss = history.history['val_loss']

# =============== LOSS & ACCURACY VISUALIZATIONS =============== #

epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel("Epoch")
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()



