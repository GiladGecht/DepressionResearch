import numpy as np
import random


def train_validation_split(train_names, data, CVS=5):
    """
    :param train_names:

    :param data:

    :param CVS: Integer
     Default 5
     Number of cross validation folds to perform
    :return:
    """

    # TH is the amount of posts we can take for our validation set, putting the rest into the train set
    data = data[data['user_name'].isin(train_names)]
    TH = int(data[data['user_name'].isin(train_names)].shape[0] / CVS)

    # Train and validation names to return
    train_names_to_send = []
    valid_names_to_send = []

    flag = 1
    for cv in range(CVS):
        # counter to check if we exceeded our TH
        total_names = 0

        # Lists to hold our current train & val names
        train_names = []
        val_names = []

        # if first iteration
        if flag == 1:
            for name in data.user_name.value_counts().keys():
                # if exceeded append to train, else val
                if total_names > TH:
                    train_names.append(name)
                else:
                    val_names.append(name)
                    total_names += data.user_name.value_counts()[name]
            # concat the lists so we can take the next names as validation next iteration
            cv_names = train_names + val_names
            train_names_to_send.append(train_names)
            valid_names_to_send.append(val_names)
            flag = 0
        else:

            for name in cv_names:
                if total_names > TH:
                    train_names.append(name)
                else:
                    val_names.append(name)
                    total_names += data.user_name.value_counts()[name]
            train_names_to_send.append(train_names)
            valid_names_to_send.append(val_names)
            cv_names = train_names + val_names

    return train_names_to_send, valid_names_to_send


def shuffle_forward(data):

    """
    This method is to shuffle the given data while retaining the original indices for each row vector

    :param data: Matrix
     The sequences matrix to shuffle
    :return: Matrix, Array
     A new matrix shuffled
     Array containing the original indices of the shuffled matrix
    """
    order = np.arange(len(data))
    random.shuffle(order)

    data = list(np.array(data)[order])

    t = np.zeros((len(list(data)), len(data[0])))
    for arr in range(len(list(data))):
        t[arr] = list(data)[arr]

    return t, order


def shuffle_backward(data, order):
    """
    This method is to un-shuffle the given data using the indices retained by the shuffle method

    :param data: Matrix
     Shuffled sequence matrix
    :param order: Array
     Array of the unshuffled indices
    :return: Matrix
     Return the shuffled matrix in its original order
    """

    data_out = [0] * len(data)
    for i, j in enumerate(order):
        data_out[j] = data[i]

    t = np.zeros((len(list(data_out)), len(data_out[0])))
    for arr in range(len(list(data_out))):
        t[arr] = list(data_out)[arr]

    return t


def split_user_train_test(data, ratio):
    """
    We calculate how much exactly ids make up for the ratio we want in our train set & test set
    If the num of rows exceeds the ratio threshold, we append the following ids to the test set keeping a close ratio.

    :param data:
    :param ratio: Float
     The ratio we wish to maintain with our train and test sets
    :return:
    """
    TH = int(data.shape[0] * ratio)
    total_names = 0

    train_names = []
    test_names = []

    for name in data.user_name.value_counts().keys():
        if total_names > TH:
            test_names.append(name)
        else:
            train_names.append(name)
            total_names += data.user_name.value_counts()[name]

    return train_names, test_names
