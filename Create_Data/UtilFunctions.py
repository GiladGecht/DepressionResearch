'''

HIT - 2018/2019 -  Academic Research Project

Title - Analyzing depression symptoms in social media

@Author - Gilad Gecht, Ohad Valtzer, Aline Eliovich
'''

# ----------- Imports ----------- #
import time
import praw
import json
import requests
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


from time import time
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from urllib3.exceptions import HTTPError
from datetime import datetime as dt
from elasticsearch import Elasticsearch

# ----------- Helper Functions ----------- #

# Load The Data
def loadData():
    submissionDF = pd.read_csv(r'/home/ohad/PycharmProjects/DepressionResearch/Create_Data/SubmissionsDF.csv')
    return submissionDF


# Connect to reddit's API using Praw
def connectToAPI():

    reddit = praw.Reddit(client_id="v2iaLX1281gABQ",
                         client_secret="lYSXF__Z1ZlbUYqF_8F2ZwYOoNo",
                         user_agent="Gilad",
                         username="GiladGe",
                         password="gilad123")

    return reddit


# Connect to desired subreddit's new section
def getNewSubreddit(redditInstance, limit):
    subreddit = redditInstance.subreddit('showerthoughts')
    new_subreddit = subreddit.new(limit=limit)

    return new_subreddit


# Iterate through the already assigned user names in the DataFrame,
# disregard existing user names and search for new not ones.
# If none found, process sleep for "X" not time.
# Otherwise, create a temporary DataFrame and concat to existing one.

def getNames(submissions, new_subreddit):
    '''

    :param submissions: The complete DataFrame
    :param new_subreddit: n new posts of a given subreddit
    :return: A list of users who exist only in new posts
    '''
    list_of_names = list(set(submissions['user_name'].tolist()))

    new_names = {"names": []}

    for submission in new_subreddit:  # works - clear
        if not submission.stickied:
            new_names["names"].append(submission.author)

    new_names = pd.DataFrame(data=new_names)
    new_names = new_names['names'].apply(lambda x: str(x))
    names_list = np.asarray(new_names)
    names_list = names_list.tolist()  # works - clear

    unique_new_names = []

    for nname in range(len(new_names)):
        curr_name = new_names[nname]
        if curr_name not in list_of_names:
            unique_new_names.append(curr_name)  # works - clear

    return unique_new_names


# Create new features and alter existing ones for better usage
def createMoreFeatures(submissionDF):
    submissionDF['title_length'] = submissionDF['title'].apply(lambda x: len(x))
    submissionDF['num_words_title'] = submissionDF['title'].apply(lambda x: len(x.split()))
    submissionDF['post_length'] = submissionDF['post_text'].apply(lambda x: len(x))
    submissionDF['num_words_post'] = submissionDF['post_text'].apply(lambda x: len(x.split()))
    submissionDF['date_created'] = submissionDF['date_created'].apply(lambda x: dt.fromtimestamp(x))
    submissionDF['title'] = submissionDF['title'].apply(lambda x: x.lower())
    submissionDF['post_text'] = submissionDF['post_text'].apply(lambda x: x.lower())

    return submissionDF


# Text Processing - Count Vectorizer and TfIdf Transformer
def vector_transformers(text_column):

    count_vect = CountVectorizer()
    post_text = count_vect.fit_transform(text_column)

    tfidf_transformer = TfidfTransformer()
    post_text_tfidf = tfidf_transformer.fit_transform(post_text)

    return post_text_tfidf


# Clean dataset
def CleanData(dataset):
    dataset = dataset[dataset['subreddit'] != 'depression']
    dataset = dataset[dataset['subreddit'] != 'AskReddit']
    dataset['post_text'] = dataset['post_text'].fillna('')
    dataset = dataset[dataset['post_text'] != '[removed]']
    dataset = dataset.dropna()
    dataset = dataset.reset_index()#.drop('index', axis=1)

    return dataset


def load_json_data():
    '''
    Get a csv file created by the current iteration of "createMoreData".
    Convert that csv file to a temporary json file, "temp_json.json".
    open and re-parse it in a way we can upload to elastic => "temp.json"
    '''

    with open('temp_json.json') as json_data:
        d = json.load(json_data)

    with open("temp.json", "w") as json_file:
        json.dump(d, json_file, indent=4)
        json_file.write("\n")

    json_data = open(r'/home/ohad/PycharmProjects/DepressionResearch/Create_Data/temp.json').read()
    data = json.loads(json_data)

    return data

def load_to_elastic(data,index,doc_type,es, counter):
    '''
    Iterate through the data file, if the length of the data is larger than 1
    iterate through the json object and upload each object to elastic.
    Indexing to the given index with the given doc type
    '''


    count = counter['count']
    if len(data) > 1:
        for d in data.items():
            temp_list = d[1]
            es.index(index=index, doc_type=doc_type, id=count, body=temp_list)
            count += 1
    else:
        es.index(index=index, doc_type='doc_type', id=count, body=data[1])
        count += 1

    es.indices.refresh(index=index)


def init_elastic(index, doc_type, elastic_address, index_counter):
    '''
    Create the Elasticsearch instance
    check if the given index exists, if not create one
    load the given data
    '''

    es = Elasticsearch(elastic_address) # TODO: Switch localhost to elastic address in the future
    object = load_json_data()

    if es.indices.exists(index=index):
        load_to_elastic(data=object, index=index, doc_type=doc_type, es=es, counter=index_counter)
    else:
        es.indices.create(index=index, ignore=400)
        load_to_elastic(data=object, index=index, doc_type=doc_type, es=es, counter=index_counter)


def addNewFeature(submissionDF):
    # adds a new column with appearance
    # Assign new columns to a DataFrame, returning a new object(a copy!) with the new columns added to the original ones
    submissionDF = submissionDF.assign(text_changed=pd.Series(data=np.zeros(submissionDF['submission_id'].shape[0])))
    submissionDF = submissionDF.assign(appearance=pd.Series(data=np.zeros(submissionDF['submission_id'].shape[0])))

    submissionDF.to_csv('SubmissionsDF.csv', index=False)

def update_data(dict, df):
    '''
    :param dict: all of the user's updated posts from reddit
    :param df: all of the user's posts from the dataframe
    :return: dictionary with all of the posts from both sources, updated.
    '''

    indices_to_remove = []
    updated_posts = 0
    new_posts = 0
    for i in range(len(dict['submission_id'])):
        if dict['submission_id'][i] in list(df['submission_id']):  # post already exist in DF
            posts = df[df['submission_id'] == dict['submission_id'][i]]  # all posts with submission_id
            max_appearance = np.max(list(posts['appearance']))
            df_row = posts.loc[posts['appearance'] == max_appearance].index[0]  # row of last appearance
            # check if last version of the post was changed and not removed
            changed = post_changed(dict, posts, i, df_row)
            if changed[0] and \
                    dict['post_text'][i] != '[removed]':
                dict['appearance'][i] = max_appearance + 1  # new appearance
                updated_posts += 1
                if changed[1]:
                    dict['text_changed'][i] = 1


            else:  # save indices to delete later - nothing changed
                indices_to_remove.append(i)
        else:
            dict['appearance'][i] = 0
            new_posts += 1

    # delete unnecessary posts
    for i in reversed(indices_to_remove):
        for key in dict:
            del dict[key][i]

    updated_posts -= len(indices_to_remove)
    print("Posts updated: ", updated_posts, ".  New posts: ", new_posts)

    return dict


def post_changed(dict, post, i, post_row):
    '''
    :param dict: dictionary with the users posts
    :param post: last version of the post
    :param i: index of the current post in dictionary
    :param post_row: index of the current post in df
    :return: True if post had been changed, False otherwise
    '''

    # remove keys that don't need checking
    keys = list(dict.keys())
    keys.remove('post_text')
    keys.remove('text_changed')
    keys.remove('appearance')
    keys.remove('date_created')

    text_changed = False
    if type(dict['post_text'][i]) is str and type(post.loc[post_row]['post_text']) is str:
        dict_post = (dict['post_text'][i]).replace("\r", "")
        df_post = (post.loc[post_row]['post_text']).lower().lstrip().replace("\r", "")
        if dict_post != df_post:
            print("post_text changed: ")
            print("New: ", dict_post, " Old: ", df_post)
            text_changed = True
            return True, text_changed

    for key in keys:
        if type(dict[key][i]) is str and type(post.loc[post_row][key]) is str:
            if (dict[key][i].lower().lstrip().replace("\r", "")) != (post.loc[post_row][key]).lower().lstrip().replace("\r", ""):
                print(key, " changed: ")
                print("New: ", dict[key][i].lower().lstrip().replace("\r", ""), " Old: ", (post.loc[post_row][key]).lower().lstrip().replace("\r", ""))
                return True, text_changed

        elif dict[key][i] != post.loc[post_row][key]:
            print(key," changed: ")
            print("New: ", dict[key][i], " Old: ", post.loc[post_row][key])
            return True, text_changed

    return False, text_changed
