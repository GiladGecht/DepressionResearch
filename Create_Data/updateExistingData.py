# ----------- Imports ----------- #
import Create_Data.UtilFunctions as utils
from datetime import datetime as dt
# import pandas as pd

# Changed from a recursive function to an infinite loop.
# thus, no extra memory required.
#
# index = 'reddit'
# doc_type = 'submission'
# es = utils.Elasticsearch("http://localhost:9200")
# if es.indices.exists(index=index):
#     index_counter = es.count(index=index)
# else:
#     es.indices.create(index=index, ignore=400)
#     index_counter = es.count(index=index)
#


reddit = utils.connectToAPI()
submissionDF = utils.loadData()
print("Current DataFrame Shape:{}".format(submissionDF.shape))

unique_names = submissionDF['user_name'].unique()
print("Number of unique users:{}".format(len(unique_names)))

'''
    The updater takes a certain amount of users from the DataFrame, 
    Download all of those users posts, cross reference them with the posts in DataFrame
    And creates new lines if there is a new post or if the post was changed, it creates
    a new line with a new index.
'''

users_list = []
amountOfUsers = 2
for i in range(0, len(unique_names), amountOfUsers):
    start = i
    if (i+amountOfUsers) < len(unique_names):
        end = i + amountOfUsers
    else:
        end = len(unique_names) + 1

    # Dataframe columns
    topics_dict = {
        "submission_id": [],
        "title": [],
        "score": [],
        "num_comments": [],
        "title_length": [],
        "subreddit": [],
        "post_text": [],
        "comment_karma": [],
        "link_karma": [],
        "upvote_ratio": [],
        "date_created": [],
        "user_name": [],
        "appearance": [],
        "text_changed": [],
        'num_words_title': [],
        'post_length': [],
        'num_words_post': [],
    }

    # ----------- Getting the current data from reddit ----------- #
    print("Entering Part 1\n")
    for curr_id in unique_names[start:end]:
        if type(curr_id) is str:
            print(curr_id)
            users_list.append(curr_id)
            print("Number of posts: ", submissionDF['user_name'].value_counts()[curr_id])
            try:
                for submission in reddit.redditor(str(curr_id)).submissions.new():
                    userName = str(submission.author)
                    topics_dict['submission_id'].append(submission.id)
                    topics_dict['title'].append(submission.title.lower())
                    topics_dict['score'].append(submission.score)
                    topics_dict['num_comments'].append(submission.num_comments)
                    topics_dict['title_length'].append(len(submission.title))
                    topics_dict['subreddit'].append(submission.subreddit)
                    topics_dict['post_text'].append(submission.selftext.lower().lstrip())
                    topics_dict['link_karma'].append(reddit.redditor(userName).link_karma)
                    topics_dict['upvote_ratio'].append(submission.upvote_ratio)
                    topics_dict['date_created'].append(dt.fromtimestamp(submission.created_utc))
                    topics_dict['user_name'].append(submission.author)
                    topics_dict['comment_karma'].append(reddit.redditor(userName).comment_karma)

                    # ------- Additions ------- #
                    topics_dict['appearance'].append(-1)
                    topics_dict['text_changed'].append(0)
                    topics_dict['num_words_title'].append(len(topics_dict['title'][-1].split()))
                    topics_dict['post_length'].append(len(topics_dict['post_text'][-1].replace("\r", "")))
                    topics_dict['num_words_post'].append(len(topics_dict['post_text'][-1].replace("\r", "").split()))

            except Exception as e:
                print("Error occurred with id:{}".format(str(curr_id)))
                print(e)
                pass

    topics_dict = utils.update_data(topics_dict, submissionDF)
    topics_dict = utils.pd.DataFrame(data=topics_dict)

    print("Entering Part 2")
    topics_dict = topics_dict[['submission_id', 'title', 'score', 'num_comments',
                               'title_length', 'subreddit', 'post_text', 'comment_karma',
                               'link_karma', 'upvote_ratio', 'date_created', 'user_name', 'appearance', 'text_changed',
                               'num_words_title','post_length' , 'num_words_post']]

    # print("Loading to Elasticsearch")
    # print(len(topics_dict))
    # topics_dict.to_csv('temp_json.csv',index=False)
    # topics_dict = pd.read_csv('temp_json.csv')
    # topics_dict.to_json('temp_json.json', orient='index')
    #
    # utils.init_elastic(index=index, doc_type=doc_type, elastic_address="http://localhost:9200", index_counter=index_counter)
    # index_counter = es.count(index=index)
    print("Saving")
    topics_dict = utils.pd.concat([topics_dict, submissionDF], sort=False)
    topics_dict = topics_dict.fillna('')

    topics_dict.to_csv('SubmissionsDF.csv', index=False)
    print("Saved\n")
    submissionDF = utils.loadData()  # Reload data to work with the new DF we just saved


print("Finished!")
