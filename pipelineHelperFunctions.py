import re
import pandas as pd
import numpy as np
import warnings
import Create_Data.UtilFunctions as utils

from sklearn.svm import LinearSVC
from Create_Data.Logging import Logger
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

warnings.simplefilter("ignore")

filename = utils.os.path.basename(__file__)[:-3]
logger = Logger(filename=filename)
def TitleClassifier(df):
    """
    First classifier - classifying title with SVM

    :param df: The DataFrame
    :return: weights of the algorithms
    """
    target = 'subreddit'
    cols = 'title'

    X = df[cols]
    y = df[target]

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

    logger.log("First Classifier - Title (with SVM)\n")
    logger.log("Accuracy Score:{}".format(score))
    logger.log(confusion_matrix(y_pred=y_pred, y_true=y_test))
    logger.log("AUC Score:{}".format(np.mean(cross_val_score(svc, X_train, y_train, cv=10, scoring='roc_auc'))))

    feature_names = np.array(count_vect.get_feature_names())
    sorted_coef_index = svc.coef_[0].argsort()

    logger.log('\nSmallest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:10]]))
    logger.log('Largest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:-11:-1]]))

    return svc, count_vect


def CleanData(dataset):
    """

    :param dataset: The DataFrame
    :return: The DataFrame, after removing posts who were removed, and deleting new lines.S
    """
    dataset['post_text'] = dataset['post_text'].fillna('')
    dataset = dataset[dataset['post_text'] != '[removed]']
    dataset['post_text'] = dataset['post_text'].apply(lambda x: x.replace('\r', ''))
    dataset['post_text'] = dataset['post_text'].apply(lambda x: x.replace('\n', ''))
    dataset['post_text'] = dataset['post_text'].apply(lambda x: x.replace("\'", ""))
    dataset = dataset.dropna()
    dataset = dataset.reset_index()

    return dataset


def GetRegularExpressions(FullDF):
    """
    Search by using regular expressions
    which are built in the following way: "i ...... " + keywordToFilterBy + "..."
    Example: "i ...... anxi/ety/ous/olytic..."
    """
    keywordToFilterBy = input("Enter keyword to run regular expressions on\n")
    myRegEx = r'\bi\s.*\b' + keywordToFilterBy + r'[\w]*\b'
    count = 0
    sentences = []
    post = []
    subreddits = []
    for row in FullDF.iterrows():

        sentence = row[1]['post_text']
        sentencesContainingRegEx = re.findall(myRegEx, sentence)
        if len(sentencesContainingRegEx) > 0:
            post.append(row[1]['submission_id'])
            subreddits.append(row[1]['subreddit'])
            sentences.append(sentence)
            count += 1
    logger.log("Amount of posts containing the regular expression:{} ".format(count))
    return post

def GetDepressionGroupUsersNeutralPosts(RegularExpressionsPosts, FullDF):
    """

    :param RegularExpressionsPosts: DataFrame containing the posts filtered by Regular Expression
    :param FullDF: The complete data
    :return: A DataFrame containing the neutral posts from the depression users
    """

    # Take n largest subreddit by appreance in the filtered dataset
    n_largest = list(RegularExpressionsPosts['subreddit'].value_counts().nlargest(7).keys())

    # Create the final depressed testing group to be compared with neutral people
    # by taking the depressed test group user id's, we can create the group's neutral posts
    depressed_group_depressed_posts = RegularExpressionsPosts[RegularExpressionsPosts['subreddit'].isin(n_largest)]
    depression_group_users = list(set(depressed_group_depressed_posts['user_name']))
    depression_group_users_indices = list(set(depressed_group_depressed_posts['user_name'].index))

    # Create a list of all the possible neutral predicted posts which contain our regular expression
    temp_list = list(
        depressed_group_depressed_posts[depressed_group_depressed_posts['predicted'] == 0]['submission_id'].index)

    # First, create the dataset comprised of the same users we have in our depression dataset
    # Second, take only the neutral related posts of these users
    # Third, drop out the posts which were filtered by the regular expression and are now considered depression wise
    # Fourth, Filter out empty posts and keep only the ones above 50 words, this leaves us with an almost similar in size dataset
    depression_group_users_neutral_posts = FullDF[FullDF['user_name'].isin(depression_group_users)]
    depression_group_users_neutral_posts = depression_group_users_neutral_posts[
        depression_group_users_neutral_posts['predicted'] == 0]
    depression_group_users_neutral_posts = depression_group_users_neutral_posts.drop(temp_list, axis=0)
    depression_group_users_neutral_posts = depression_group_users_neutral_posts[
        depression_group_users_neutral_posts['num_words_post'] > 50]

    # Create a dataset comprised of all the other users who weren't classified as depressed by our regular expression
    # next, we only want those who we classified by our original classifier, who were predicted as neutral => predicted = 0
    non_depressed_people = FullDF.drop(depression_group_users_indices, axis=0).copy()
    non_depressed_people = non_depressed_people[non_depressed_people['predicted'] == 1]
    non_depressed_people = non_depressed_people[non_depressed_people['num_words_post'] > 50]

    depression_group_users_neutral_posts = depression_group_users_neutral_posts.reset_index().drop('index', axis=1)
    neutral_total_subreddits = set(depression_group_users_neutral_posts['subreddit'].value_counts().keys())

    filtered_neutral_subreddits = list(set(n_largest) ^ neutral_total_subreddits)

    depression_group_users_neutral_posts = depression_group_users_neutral_posts[
        depression_group_users_neutral_posts['subreddit'].isin(filtered_neutral_subreddits)]

    # Print how many unique users we have for each group:
    logger.log("Number of Unique depressed posts users:{}".format(len(list(set(depressed_group_depressed_posts['user_name'])))))
    logger.log("Number of Unique depressed neutral posts users:{}".format(
          len(list(set(depression_group_users_neutral_posts['user_name'])))))
    logger.log("Number of Unique neutral posts users:{}".format(len(list(set(non_depressed_people['user_name'])))))

    return depression_group_users_neutral_posts


def ConvertInputToListOfStrings(Subreddits):
    Subreddits = Subreddits.replace("'", "")
    Subreddits = Subreddits.split(',')
    return Subreddits


def GetNeutralAndDepressionSubreddits(Whole_data, Subreddits):
    """

    :param Whole_data: The complete DataFrame
    :param Subreddits:  a set of all the subreddits
    :return: a list of neutral subreddits and a list of depression subreddits
    """

    neutralSubreddits = []
    depression_subreddits = []
    for i in Subreddits:
        values = Whole_data[Whole_data['subreddit'] == i]['predicted'].value_counts().values
        sum_values = np.sum(Whole_data[Whole_data['subreddit'] == i]['predicted'].value_counts().values)
        values_perc = values / sum_values
        # value1 = Whole_data[Whole_data['subreddit'] == i]['predicted'].value_counts().values[0]
        if Whole_data[Whole_data['subreddit'] == i]['predicted'].value_counts().keys()[0] == 1:
            if values_perc[0] >= 0.7:
                neutralSubreddits.append(i)
        else:
            if values_perc[0] >= 0.7:
                depression_subreddits.append(i)

    logger.log("Distribution of depression subreddits\n")
    logger.log(Whole_data[Whole_data['subreddit'].isin(depression_subreddits)]['subreddit'].value_counts())

    logger.log("Distribution of neutral subreddits\n")
    logger.log(Whole_data[Whole_data['subreddit'].isin(neutralSubreddits)]['subreddit'].value_counts())
    return neutralSubreddits, depression_subreddits


def GetNeutralDepressionUsers(WholeData, AnxietySubreddits, NeutralSubreddits):
    """
    :param WholeData: The complete DataFrame
    :param AnxietySubreddits: A list of Anxiety subreddits
    :param NeutralSubreddits: A list of neutral subreddits
    :return: A DataFrame of users who posted in both depression and neutral subreddits
    """

    # Split the dataframe to neutral and depressed by the filtered subreddits
    depression_df = WholeData[WholeData['subreddit'].isin(AnxietySubreddits)]
    neutral_df = WholeData[WholeData['subreddit'].isin(NeutralSubreddits)]

    logger.log("Anxiety group size:{}\n\n".format(depression_df.shape))
    logger.log(20 * "-")
    logger.log("neutral group size:{}\n\n".format(neutral_df.shape))

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
    logger.log("Amount of unique users who are in both groups:{} ".format(len(both)))

    anxietyGroupSize = 0
    neutralGroupSize = 0
    for user in depression_names:
        anxietyGroupSize += WholeData[WholeData['user_name'] == user].shape[0]

    for user in neutral_names:
        neutralGroupSize += WholeData[WholeData['user_name'] == user].shape[0]

    logger.log("Posts taken from anxious users:{} ".format(anxietyGroupSize))
    logger.log(20 * "-")
    logger.log("Posts taken from neutral users:{} ".format(neutralGroupSize))

    full_df = full_df[full_df['user_name'].isin(both)]
    full_df = full_df.sort_values(by=['user_name', 'date_created'], ascending=False)
    full_df['num_distinct_words'] = full_df['post_text'].apply(lambda x: len(set(x.split())))

    return full_df

