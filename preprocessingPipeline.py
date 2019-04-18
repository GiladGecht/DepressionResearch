import pandas as pd
import numpy as np
import warnings
from pipelineHelperFunctions import *
warnings.simplefilter("ignore")


from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder


def main():
    partialData = pd.read_csv()
    wholeData = pd.read_csv()
    GenerateNeutralAndDepressionSubreddits(partialData, wholeData)

    print("Enter desired neutral subreddits:")
    neutralSubredditsInput = input("Enter the neutral subreddits")
    neutralSubreddits = neutralSubredditsInput.split(',')

    print("Enter desired depression subreddits:")
    depressedSubredditsInput = input("Enter the depression subreddits")
    depressedSubreddits = depressedSubredditsInput.split(',')

    neutralPosts = ContinueScript(neutralSubreddits, depressedSubreddits, wholeData)
    pd.to_csv('neutralPosts.csv')




def GenerateNeutralAndDepressionSubreddits(i_PartialData, i_WholeData):
    df = pd.read_csv(i_PartialData)
    df = shuffle(df)
    encoder = LabelEncoder()
    df['subreddit'] = encoder.fit_transform(df['subreddit'])
    df['post_text'] = df['post_text'].fillna('')
    df = df[df['post_text'] != '[removed]']
    df = df[df['title_length'] >= 20]
    df = df.dropna()

    svc, count_vect = TitleClassifier(df)

    whole_data = pd.read_csv(i_WholeData)
    whole_data = CleanData(whole_data)
    whole_data['predicted'] = svc.predict(count_vect.transform(whole_data['title']))

    # Filter out the data by noise
    # Subreddits with less than 50 appearences are dropped out
    counts = whole_data['subreddit'].value_counts()
    popular_subreddits = counts[counts.values >= 50].keys()
    whole_data = whole_data[(whole_data['subreddit'].isin(popular_subreddits))]

    # Number of UNIQUE subreddits left after being filtered
    subreddits = set(whole_data['subreddit'])
    print("Amount of subredditss: ",len(subreddits))

    neutralSubreddits, depression_subreddits = GetNeutralAndDepressionSubreddits(whole_data, subreddits)
    print("The Filtered Neutral Subreddits Are:\n\n", neutralSubreddits)
    print(20 * "-")
    print("The Filtered Anxiety Subreddits are:\n\n", depression_subreddits)




def GetNeutralAndDepressionSubreddits(whole_data, subreddits):
    neutralSubreddits = []
    depression_subreddits = []
    for i in subreddits:
        values = whole_data[whole_data['subreddit'] == i]['predicted'].value_counts().values
        sum_values = np.sum(whole_data[whole_data['subreddit'] == i]['predicted'].value_counts().values)
        values_perc = values / sum_values
        value1 = whole_data[whole_data['subreddit'] == i]['predicted'].value_counts().values[0]
        if whole_data[whole_data['subreddit'] == i]['predicted'].value_counts().keys()[0] == 1:
            if values_perc[0] >= 0.7:
                neutralSubreddits.append(i)
        else:
            if values_perc[0] >= 0.7:
                depression_subreddits.append(i)
    return neutralSubreddits, depression_subreddits



def ContinueScript(i_NeutralSubreddits, i_DepressedSubreddits, wholeData):
    # Split the dataframe to neutral and depressed by the filtered subreddits
    depression_df = wholeData[wholeData['subreddit'].isin(i_DepressedSubreddits)]
    neutral_df = wholeData[wholeData['subreddit'].isin(i_NeutralSubreddits)]

    print(depression_df.shape)
    print("\n\n")
    print(neutral_df.shape)

    # Get the list of all unique users for each type of dataset
    depression_names = list(set(depression_df['user_name']))
    neutral_names = list(set(neutral_df['user_name']))

    # Merge them back to a single dataframe
    full_df = pd.concat([depression_df, neutral_df], axis=0)
    print("Full DF shape: ", full_df.shape)

    # Filter out people who havn't posted in both types of subreddits (Depression/Neutral) in the current dataset
    both = []
    for i in depression_names:
        if i in neutral_names:
            both.append(i)

    full_df = full_df[full_df['user_name'].isin(both)]
    full_df = full_df.sort_values(by=['user_name', 'date_created'], ascending=False)
    full_df['num_distinct_words'] = full_df['post_text'].apply(lambda x: len(set(x.split())))

    postsByRegEx = GetRegularExpressions(full_df)

    # Find out how many unique users we found who match our regular expressions - by submission id
    # this is done to get only their depression related posts and not their entire posts
    # Later on, we'll take the rest of their post and categorize them as neutral based
    # this will be our compare group
    users_filtered_by_re = list(set(postsByRegEx))
    print("Amount of posts after regular expressions filter", len(list(set(postsByRegEx))))

    # Get all the unique users found in the previous step
    filtered_by_re = full_df[full_df['submission_id'].isin(users_filtered_by_re)].copy()

    depressionGroupNeutralPosts = GetDepressionGroupUsersNeutralPosts(filtered_by_re, full_df)

    return  depressionGroupNeutralPosts



