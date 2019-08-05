import numpy as np


class Preprocessing:
    def __init__(self, tokenizer, neutral_train, neutral_test, neutral_val, depression_train, depression_test, depression_val, SEQ_LEN):
        """

        :param tokenizer: Object
        Tokenizer object pre-fit on the text

        :param neutral_train: String
         Concatenated string comprised from the neutral group train set users

        :param neutral_test: String
         Concatenated string comprised from the neutral group test set users

        :param neutral_val: String
         Concatenated string comprised from the neutral group validation set users

        :param depression_train: String
         Concatenated string comprised from the depression group train set users

        :param depression_test: String
         Concatenated string comprised from the depression group test set users

        :param depression_val: String
         Concatenated string comprised from the depression group validation set users

        :param SEQ_LEN: Integer
         The length to which we will split each sequence from the concatenated string
        """

        self.Neutral_train    = neutral_train
        self.Neutral_test     = neutral_test
        self.Neutral_val      = neutral_val
        self.Depression_test  = depression_test
        self.Depression_train = depression_train
        self.Depression_val   = depression_val
        self.SEQ_LEN          = SEQ_LEN
        self.tokenizer        = tokenizer


def make_sequences(self):

    """
    Convert every data set into a sequence matrix
    """

    self.neutral_long_sequence_train = self.tokenizer.texts_to_sequences([self.Neutral_train])[0]
    self.neutral_long_sequence_test = self.tokenizer.texts_to_sequences([self.Neutral_test])[0]
    self.neutral_long_sequence_val = self.tokenizer.texts_to_sequences([self.Neutral_val])[0]
    self.depressed_long_sequence_train = self.tokenizer.texts_to_sequences([self.Depression_train])[0]
    self.depressed_long_sequence_test = self.tokenizer.texts_to_sequences([self.Depression_test])[0]
    self.depressed_long_sequence_val = self.tokenizer.texts_to_sequences([self.Depression_val])[0]


def make_subsequences(self, long_sequence, label):
    """

    :param self:
    :param long_sequence: String
     The concatenated string initialized in our init method according to each set and group
    :param label: Integer
     The true class for this specific sequence
    :return: Matrix comprised of sequences from the entire group's concatenated text with the according class
    """

    len_sequences = len(long_sequence)
    X = np.zeros(((len_sequences - self.SEQ_LEN) + 1, self.SEQ_LEN))
    y = np.zeros((X.shape[0], 1))

    for i in range(X.shape[0]):
        X[i] = long_sequence[i:i + self.SEQ_LEN]
        y[i] = label
    return X, y


def create_train_test_subsequences(self):
    """
    Driver function to activate the entire preprocessing class and create our data sets to feed to the network

    :param self:
    :return:
    """
    self.make_sequences()
    self.X_neutral_train, self.y_neutral_train       = self.make_subsequences(self.neutral_long_sequence_train, 0)
    self.X_neutral_test, self.y_neutral_test         = self.make_subsequences(self.neutral_long_sequence_test, 0)
    self.X_neutral_val, self.y_neutral_val           = self.make_subsequences(self.neutral_long_sequence_val, 0)
    self.X_depression_train, self.y_depression_train = self.make_subsequences(self.depressed_long_sequence_train, 1)
    self.X_depression_test, self.y_depression_test   = self.make_subsequences(self.depressed_long_sequence_test, 1)
    self.X_depression_val, self.y_depression_val     = self.make_subsequences(self.depressed_long_sequence_val, 1)

    self.X_train = np.vstack((self.X_depression_train, self.X_neutral_train[:self.X_depression_train.shape[0]]))
    self.X_test  = np.vstack((self.X_depression_test, self.X_neutral_test[:self.X_depression_test.shape[0]]))
    self.X_val   = np.vstack((self.X_depression_val, self.X_neutral_val[:self.X_depression_val.shape[0]]))

    self.X_test_copy = self.X_test.copy()
    self.y_train     = np.vstack((self.y_depression_train, self.y_neutral_train[:self.y_depression_train.shape[0]]))
    self.y_test      = np.vstack((self.y_depression_test, self.y_neutral_test[:self.y_depression_test.shape[0]]))
    self.y_val       = np.vstack((self.y_depression_val, self.y_neutral_val[:self.X_depression_val.shape[0]]))

    return self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val
