'''
@Authors: Gilad Gecht, Ohad Valtzer, Aline Eliovich
Date Created: December 10th, 2018
'''

# ------- Imports ------- #
import Create_Data.UtilFunctions as utils

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC


# ------- Load Data ------- #
df = utils.pd.read_csv('https://raw.githubusercontent.com/GiladGecht/DepressionResearch/master/depression_neutral_df.csv')

# ------- Data Preprocessing ------- #

'''
Encoding the target label => Subreddit Name to 0/1, Neutral or Depression
For the current moment, we're keeping empty posts, this might be a key feature,
perhaps depressed people tend to write way more than a non-depressed person.
Since removed doesn't imply anything, we'll remove rows which contain such behaviour.
Eventually, we'll remove any other signs of Nulls in the dataframe.
'''

df = shuffle(df)
encoder = LabelEncoder()
df['_subreddit'] = encoder.fit_transform(df['_subreddit'])
df['_post_text'] = df['_post_text'].fillna('')
df = df[df['_post_text'] != '[removed]']
df = df.dropna()

# ------- Splitting Data to Features & Label ------- #

target = '_subreddit'
cols = '_title'

X = df[cols]
y = df[target]

count_vect = utils.CountVectorizer(stop_words='english', lowercase=True,analyzer='word')
X = count_vect.fit_transform(X)
tfidf_transformer = utils.TfidfTransformer()
X = tfidf_transformer.fit_transform(X)


# ------- Splitting to Train & Test ------- #

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------- Model Training ------- #

svc = LinearSVC(random_state=42, penalty='l2', dual= True, tol=0.0001, C = 1,
                fit_intercept= True, intercept_scaling=1.0, class_weight= None)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
score = svc.score(X_test, y_test)

# ------- Model Evaluation ------- #
print("Accuracy Score:",score)
print(confusion_matrix(y_pred=y_pred,y_true=y_test))
print("AUC Score:", utils.np.mean(cross_val_score(svc, X_train, y_train, cv=5, scoring='roc_auc')))

# ------- Predict Real Data ------- #
whole_data = utils.pd.read_csv(r'/home/ohad/PycharmProjects/DepressionResearch/Create_Data/SubmissionsDF2.csv',index_col=0)
whole_data = utils.CleanData(whole_data)
whole_data['predicted'] = svc.predict(count_vect.transform(whole_data['title']))

# ------- Check Results ------- #

'''
Filter out subreddits with less than 50 appearances.
After filtering, for each subreddit, check the distribution in the classification.
If the classification tends to fit more to the depression side (over 70% of the classification was 1)
add to the depression list.
if the classification tends to be more neutral, i.e over 70% was 0, send to neutral.

CURRENT THRESHOLD = 0.7 

'''
predicted = whole_data[['title','subreddit','predicted']]
counts = whole_data['subreddit'].value_counts()
popular_subreddits = counts[counts.values >= 50].keys()
whole_data = whole_data[(whole_data['subreddit'].isin(popular_subreddits))]
subreddits = set(whole_data['subreddit'])

neutral_subreddits = []
depression_subreddits = []
THRESHOLD = 0.7
for i in subreddits:
    values = whole_data[whole_data['subreddit'] == i]['predicted'].value_counts().values
    sum_values = utils.np.sum(whole_data[whole_data['subreddit'] == i]['predicted'].value_counts().values)
    values_perc = values/sum_values
    value1 = whole_data[whole_data['subreddit'] == i]['predicted'].value_counts().values[0]
    if whole_data[whole_data['subreddit'] == i]['predicted'].value_counts().keys()[0] == 0:
        if values_perc[0] >= THRESHOLD:
            neutral_subreddits.append(i)
    else:
        if values_perc[0] >= THRESHOLD:
            depression_subreddits.append(i)
print("Depression Subreddits:\n", depression_subreddits)
print("Neutral Subreddits:\n", neutral_subreddits)
