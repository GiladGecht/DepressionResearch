import re
import requests
import json
import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")

from sklearn.decomposition import PCA
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, normalize, Normalizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def TitleClassifier(io_df):
    target = 'subreddit'
    cols = 'title'

    X = io_df[cols]
    y = io_df[target]

    count_vect = CountVectorizer(stop_words='english', lowercase=True, analyzer='word')
    X = count_vect.fit_transform(X)
    tfidf_transformer = TfidfTransformer()
    X = tfidf_transformer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svc = LinearSVC(random_state=42, penalty='l2', dual=True, tol=0.0001, C=1,
                    fit_intercept=True, intercept_scaling=1.0, class_weight=None)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    score = svc.score(X_test, y_test)

    print("First Classifier - Title (with SVM)\n")
    print("Accuracy Score:", score)
    print(confusion_matrix(y_pred=y_pred, y_true=y_test))
    print("AUC Score:", np.mean(cross_val_score(svc, X_train, y_train, cv=10, scoring='roc_auc')))

    feature_names = np.array(count_vect.get_feature_names())
    sorted_coef_index = svc.coef_[0].argsort()

    print('\nSmallest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:10]]))
    print('Largest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:-11:-1]]))

    return svc, count_vect

def CleanData(dataset):
    dataset['post_text'] = dataset['post_text'].fillna('')
    dataset = dataset[dataset['post_text'] != '[removed]']
    dataset['post_text'] = dataset['post_text'].apply(lambda x: x.replace('\r', ''))
    dataset['post_text'] = dataset['post_text'].apply(lambda x: x.replace('\n', ''))
    dataset['post_text'] = dataset['post_text'].apply(lambda x: x.replace("\'", ""))
    dataset = dataset.dropna()
    dataset = dataset.reset_index()

    return dataset

def GetRegularExpressions(io_FullDF):
    # Find by using regular expressions all the sentences
    # which are built in the following way: "i ...... anxi/ety/ous/olytic and so on"
    keywordToFilterBy = input("Enter keyword to run regular expressions on\n")
    myRegEx = r'\bi\s.*\b' + keywordToFilterBy + r'[\w]*\b'
    count = 0
    sentences = []
    post = []
    subreddits = []
    for row in io_FullDF.iterrows():

        sentence = row[1]['post_text']
        sentencesContainingRegEx = re.findall(myRegEx, sentence)
        if len(sentencesContainingRegEx) > 0:
            post.append(row[1]['submission_id'])
            subreddits.append(row[1]['subreddit'])
            sentences.append(sentence)
            count += 1
    print("Amount of posts containing the regular expression: ", count)
    return post

def GetDepressionGroupUsersNeutralPosts(i_RegularExpressionsPosts, io_FullDF):

    # Take n largest subreddit by appreance in the filtered dataset
    n_largest = list(i_RegularExpressionsPosts['subreddit'].value_counts().nlargest(7).keys())

    # Create the final depressed testing group to be compared with neutral people
    # by taking the depressed test group user id's, we can create the group's neutral posts
    depressed_group_depressed_posts = i_RegularExpressionsPosts[i_RegularExpressionsPosts['subreddit'].isin(n_largest)]
    depression_group_users = list(set(depressed_group_depressed_posts['user_name']))
    depression_group_users_indices = list(set(depressed_group_depressed_posts['user_name'].index))

    # Create a list of all the possible neutral predicted posts which contain our regular expression
    temp_list = list(
        depressed_group_depressed_posts[depressed_group_depressed_posts['predicted'] == 0]['submission_id'].index)

    # First, create the dataset comprised of the same users we have in our depression dataset
    # Second, take only the neutral related posts of these users
    # Third, drop out the posts which were filtered by the regular expression and are now considered depression wise
    # Fourth, Filter out empty posts and keep only the ones above 50 words, this leaves us with an almost similar in size dataset
    depression_group_users_neutral_posts = io_FullDF[io_FullDF['user_name'].isin(depression_group_users)]
    depression_group_users_neutral_posts = depression_group_users_neutral_posts[
        depression_group_users_neutral_posts['predicted'] == 0]
    depression_group_users_neutral_posts = depression_group_users_neutral_posts.drop(temp_list, axis=0)
    depression_group_users_neutral_posts = depression_group_users_neutral_posts[
        depression_group_users_neutral_posts['num_words_post'] > 50]

    # Create a dataset comprised of all the other users who weren't classified as depressed by our regular expression
    # next, we only want those who we classified by our original classifier, who were predicted as neutral => predicted = 0
    non_depressed_people = io_FullDF.drop(depression_group_users_indices, axis=0).copy()
    non_depressed_people = non_depressed_people[non_depressed_people['predicted'] == 1]
    non_depressed_people = non_depressed_people[non_depressed_people['num_words_post'] > 50]

    depression_group_users_neutral_posts = depression_group_users_neutral_posts.reset_index().drop('index', axis=1)
    neutral_total_subreddits = set(depression_group_users_neutral_posts['subreddit'].value_counts().keys())

    filtered_neutral_subreddits = list(set(n_largest) ^ neutral_total_subreddits)

    depression_group_users_neutral_posts = depression_group_users_neutral_posts[
        depression_group_users_neutral_posts['subreddit'].isin(filtered_neutral_subreddits)]

    # Print how many unique users we have for each group:
    print("Number of Unique depressed posts users:", len(list(set(depressed_group_depressed_posts['user_name']))))
    print("Number of Unique depressed neutral posts users:",
          len(list(set(depression_group_users_neutral_posts['user_name']))))
    print("Number of Unique neutral posts users", len(list(set(non_depressed_people['user_name']))))

    return depression_group_users_neutral_posts

def ConvertInputToListOfStrings(io_Subreddits):
    io_Subreddits = io_Subreddits.replace("'", "")
    io_Subreddits = io_Subreddits.split(',')
    return  io_Subreddits

def GetNeutralAndDepressionSubreddits(io_Whole_data, i_Subreddits):
    neutralSubreddits = []
    depression_subreddits = []
    for i in i_Subreddits:
        values = io_Whole_data[io_Whole_data['subreddit'] == i]['predicted'].value_counts().values
        sum_values = np.sum(io_Whole_data[io_Whole_data['subreddit'] == i]['predicted'].value_counts().values)
        values_perc = values / sum_values
        # value1 = io_Whole_data[io_Whole_data['subreddit'] == i]['predicted'].value_counts().values[0]
        if io_Whole_data[io_Whole_data['subreddit'] == i]['predicted'].value_counts().keys()[0] == 1:
            if values_perc[0] >= 0.7:
                neutralSubreddits.append(i)
        else:
            if values_perc[0] >= 0.7:
                depression_subreddits.append(i)

    print("Distribution of depression subreddits\n")
    print(io_Whole_data[io_Whole_data['subreddit'].isin(depression_subreddits)]['subreddit'].value_counts())

    print("Distribution of neutral subreddits\n")
    print(io_Whole_data[io_Whole_data['subreddit'].isin(neutralSubreddits)]['subreddit'].value_counts())
    return neutralSubreddits, depression_subreddits

def GetNeutralDepressionUsers(io_WholeData, i_AnxietySubreddits, i_NeutralSubreddits):
    # Split the dataframe to neutral and depressed by the filtered subreddits
    depression_df = io_WholeData[io_WholeData['subreddit'].isin(i_AnxietySubreddits)]
    neutral_df = io_WholeData[io_WholeData['subreddit'].isin(i_NeutralSubreddits)]

    print("Anxiety group size:\n\n", depression_df.shape)
    print(20 * "-")
    print("neutral group size:\n\n", neutral_df.shape)

    # Get the list of all unique users for each type of dataset
    depression_names = list(set(depression_df['user_name']))
    neutral_names = list(set(neutral_df['user_name']))

    # Merge them back to a single dataframe
    full_df = pd.concat([depression_df, neutral_df], axis=0)
    full_df.shape

    # Filter out people who havn't posted in both types of subreddits (Depression/Neutral) in the current dataset
    both = []
    for i in depression_names:
        if i in neutral_names:
            both.append(i)
    print("Amount of unique users who are in both groups: ", len(both))

    anxietyGroupSize = 0
    neutralGroupSize = 0
    for user in depression_names:
        anxietyGroupSize += io_WholeData[io_WholeData['user_name'] == user].shape[0]

    for user in neutral_names:
        neutralGroupSize += io_WholeData[io_WholeData['user_name'] == user].shape[0]

    print("Posts taken from anxious users: ", anxietyGroupSize)
    print(20 * "-")
    print("Posts taken from neutral users: ", neutralGroupSize)

    full_df = full_df[full_df['user_name'].isin(both)]
    full_df = full_df.sort_values(by=['user_name', 'date_created'], ascending=False)
    full_df['num_distinct_words'] = full_df['post_text'].apply(lambda x: len(set(x.split())))

    return full_df

