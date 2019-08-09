import re
import pandas as pd
import numpy as np
import warnings
import Create_Data.UtilFunctions as utils


from pipelineHelperFunctions import *
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from Create_Data.Logging import Logger
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

warnings.simplefilter("ignore")


def DataFilter(DF):
    """

    :param DF: The current DataFrame
    :return: The DataFrame after filtering posts who were removed, empty or too short.
    """

    DF = shuffle(DF)
    encoder = LabelEncoder()
    DF['subreddit'] = encoder.fit_transform(DF['subreddit'])
    DF['post_text'] = DF['post_text'].fillna('')
    DF = DF[DF['post_text'] != '[removed]']
    DF = DF[DF['title_length'] >= 20]
    DF = DF.dropna()
    return DF

def FilterWholeData(PartialData, WholeData):
    """

    :param PartialData: The DataFrame with posts only from
    AskReddit and the relevant subreddit for the desired type of depression

    :param WholeData: The whole DataFrame

    :return: A DataFrame, after cleaning irrelevant posts
             and taking only popular subreddits from the whole DataFrame
    """

    svc, count_vect = TitleClassifier(PartialData)
    WholeData = CleanData(WholeData)
    WholeData['predicted'] = svc.predict(count_vect.transform(WholeData['title']))
    # Filter out the data by noise
    # Subreddits with less than 50 appearances are dropped out
    counts = WholeData['subreddit'].value_counts()
    popular_subreddits = counts[counts.values >= 50].keys()
    WholeData = WholeData[(WholeData['subreddit'].isin(popular_subreddits))]

    return WholeData



def Pipeline():
    df = pd.read_csv(r'/home/ohad/Desktop/Studies/Year3/Project/Updated_Data/anxietyTemp.csv')
    df = DataFilter(df)

    whole_data = pd.read_csv(r'/home/ohad/Desktop/Studies/Year3/Project/Updated_Data/SubmissionsDF.csv', index_col=0)
    whole_data = FilterWholeData(df, whole_data)

    # Number of UNIQUE subreddits left after being filtered
    subreddits = set(whole_data['subreddit'])
    len(subreddits)

    neutral_subreddits, anxiety_subreddits = GetNeutralAndDepressionSubreddits(whole_data, subreddits)

    logger.log("The Filtered Neutral Subreddits Are:\n\n", neutral_subreddits)
    logger.log(20 * "-")
    logger.log("The Filtered Anxiety Subreddits are:\n\n", anxiety_subreddits)

    anxiety_subreddit_filtered_list = ['Anxiety']

    # Split the dataframe to neutral and depressed by the filtered subreddits
    depression_df = whole_data[whole_data['subreddit'].isin(anxiety_subreddit_filtered_list)]
    neutral_df = whole_data[whole_data['subreddit'].isin(neutral_subreddits)]

    logger.log("Anxiety group size:\n\n", depression_df.shape)
    logger.log(20 * "-")
    logger.log("neutral group size:\n\n", neutral_df.shape)

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
    logger.log("Amount of unique users who are in both groups: ", len(both))

    anxietyGroupSize = 0
    neutralGroupSize = 0
    for user in depression_names:
        anxietyGroupSize += whole_data[whole_data['user_name'] == user].shape[0]

    for user in neutral_names:
        neutralGroupSize += whole_data[whole_data['user_name'] == user].shape[0]

    logger.log("Posts taken from anxious users: ", anxietyGroupSize)
    logger.log(20 * "-")
    logger.log("Posts taken from neutral users: ", neutralGroupSize)

    full_df = full_df[full_df['user_name'].isin(both)]
    full_df = full_df.sort_values(by=['user_name', 'date_created'], ascending=False)
    full_df['num_distinct_words'] = full_df['post_text'].apply(lambda x: len(set(x.split())))

    post = GetRegularExpressions(full_df)

    # Find out how many unique users we found who match our regular expressions - by submission id
    # this is done to get only their depression related posts and not their entire posts
    # Later on, we'll take the rest of their post and categorize them as neutral based
    # this will be our compare group
    users_filtered_by_re = list(set(post))
    len(list(set(post)))

    # Get all the unique users found in the previous step
    filtered_by_re = full_df[full_df['submission_id'].isin(users_filtered_by_re)].copy()

    # Take n largest subreddit by appreance in the filtered dataset
    n_largest = list(filtered_by_re['subreddit'].value_counts().nlargest(7).keys())

    logger.log(filtered_by_re['subreddit'].value_counts())

    # Create the final depressed testing group to be compared with neutral people
    # by taking the depressed test group user id's, we can create the group's neutral posts
    depressed_group_depressed_posts = filtered_by_re[filtered_by_re['subreddit'].isin(n_largest)]
    depression_group_users = list(set(depressed_group_depressed_posts['user_name']))
    depression_group_users_indices = list(set(depressed_group_depressed_posts['user_name'].index))

    # Create a list of all the possible neutral predicted posts which contain our regular expression
    temp_list = list(
        depressed_group_depressed_posts[depressed_group_depressed_posts['predicted'] == 1]['submission_id'].index)

    # First, create the dataset comprised of the same users we have in our depression dataset
    # Second, take only the neutral related posts of these users
    # Third, drop out the posts which were filtered by the regular expression and are now considered depression wise
    # Fourth, Filter out empty posts and keep only the ones above 50 words, this leaves us with an almost similar in size dataset
    depression_group_users_neutral_posts = full_df[full_df['user_name'].isin(depression_group_users)]
    depression_group_users_neutral_posts = depression_group_users_neutral_posts[
        depression_group_users_neutral_posts['predicted'] == 1]
    depression_group_users_neutral_posts = depression_group_users_neutral_posts.drop(temp_list, axis=0)
    depression_group_users_neutral_posts = depression_group_users_neutral_posts[
        depression_group_users_neutral_posts['num_words_post'] > 50]

    # Create a dataset comprised of all the other users who weren't classified as depressed by our regular expression
    # next, we only want those who we classified by our original classifier, who were predicted as neutral => predicted = 0
    non_depressed_people = full_df.drop(depression_group_users_indices, axis=0).copy()
    non_depressed_people = non_depressed_people[non_depressed_people['predicted'] == 1]
    non_depressed_people = non_depressed_people[non_depressed_people['num_words_post'] > 50]

    logger.log(non_depressed_people.head())

    depression_group_users_neutral_posts = depression_group_users_neutral_posts.reset_index().drop('index', axis=1)
    neutral_total_subreddits = set(depression_group_users_neutral_posts['subreddit'].value_counts().keys())

    filtered_neutral_subreddits = list(set(n_largest) ^ neutral_total_subreddits)

    depression_group_users_neutral_posts = depression_group_users_neutral_posts[
        depression_group_users_neutral_posts['subreddit'].isin(filtered_neutral_subreddits)]

    # Print how many unique users we have for each group:
    logger.log("Number of Unique depressed posts users:", len(list(set(depressed_group_depressed_posts['user_name']))))
    logger.log("Number of Unique depressed neutral posts users:",
          len(list(set(depression_group_users_neutral_posts['user_name']))))
    logger.log("Number of Unique neutral posts users", len(list(set(non_depressed_people['user_name']))))

    logger.log("Number of depression Neutral posts: ", depression_group_users_neutral_posts.shape)

    depression_group_users_neutral_posts.to_csv('neutralPosts.csv')


Pipeline()