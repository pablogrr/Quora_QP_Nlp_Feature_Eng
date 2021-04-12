
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
from sklearn import svm
import xgboost as xgb
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

import nltk
import pandas as pd
import scipy
import sklearn
from sklearn import *
import numpy as np
from nltk import *
import os
import statistics
import editdistance
import itertools  
import re

def cast_list_as_strings(mylist):
    """
    return a list of strings
    """
    #assert isinstance(mylist, list), f"the input mylist should be a list it is {type(mylist)}"
    mylist_of_strings = []
    for x in mylist:
        mylist_of_strings.append(str(x))

    return mylist_of_strings


def common_words_transformation_remove_punctuation(text):
    
    text = text.lower()
    
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"when's", "when is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"there's", "there is", text)

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"\'s", " ", text)  # 除了上面的特殊情况外，“\'s”只能表示所有格，应替换成“ ”
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" uk ", " england ", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" ds ", " data science ", text)
    text = re.sub(r" ee ", " electronic engineering ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iphone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the us", "america", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ", text)
    text = re.sub(r"\+", " ", text)
    text = re.sub(r"\-", " ", text)
    text = re.sub(r"\=", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " ", text)
    text = re.sub(r"\0s", "0", text)
    
    punctuation="?:!.,;"

    text = "".join([c for c in text if c not in punctuation])
    
    return text


def lemm(sentence):
    wordnet_lemmatizer = WordNetLemmatizer()

    token_words = word_tokenize(sentence)
    sentence_stemmed = []
    for word in token_words:
        sentence_stemmed.append(wordnet_lemmatizer.lemmatize(word))
        sentence_stemmed.append(" ")
    return "".join(sentence_stemmed)


def preprocess(df, save=False):
    df = df.copy()
    df['question1'] = cast_list_as_strings(list(df["question1"]))
    df['question2'] = cast_list_as_strings(list(df["question2"]))
        
    df['question1_cleaned'] = [common_words_transformation_remove_punctuation(quest) for quest in df["question1"]]
    df['question2_cleaned'] = [common_words_transformation_remove_punctuation(quest) for quest in df["question2"]]
        
    if save==True:
        df.to_csv('preprocessed_Quora_full_train_data.csv')
        
    return df

#Counts the number of words in each string
def cont_len(question):
    num_words = len(question.split())
    return num_words

#Adding len Words in common feat
def len_common(q1, q2):
    q1 = set(word_tokenize(q1)) ; q2 = set(word_tokenize(q2))
    return len(q1.intersection(q2))

#Adding len common words in common feat
def len_not_common(q1,q2):
    q1 = set(word_tokenize(q1)) ; q2 = set(word_tokenize(q2))
    return len(q1 ^ q2)

#Adding mean distance between common words 
def mean_dist_not_com(q1,q2):
    q1 = set(word_tokenize(q1)) ; q2 = set(word_tokenize(q2))
    not_comm1 = (q1 ^ q2) - q1
    if len(not_comm1)==0 : not_comm1={''}
    not_comm2 = (q1 ^ q2) - q2
    if len(not_comm2)==0 : not_comm2={''}
    return statistics.mean([editdistance.eval(i[0],i[1]) for i in itertools.product(not_comm1, not_comm2)])

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString)) 
def both_number(q1,q2):
    return hasNumbers(q1) *  hasNumbers(q2) 

# get average number of words
def average_len(question):
    num_words = len(question.split())
    return len(question)/num_words   

# levenshtein distance (for strings of unequal length)
def levenshtein(q1, q2): 
    #create initial array (two for loops since q1 and q2 can differ in length)
    dist_array = []
    for i in range(len(q1)+1):
        dist_array.append([0]*(len(q2)+1))
        dist_array[i][0] = i
    for j in range(len(q2)+1):
        dist_array[0][j] = j

    dist = [0]*3
    for i in range(1,len(q1)+1):
        for j in range(1,len(q2)+1):
            dist[0] = dist_array[i-1][j-1] if q1[i-1]==q2[j-1] else dist_array[i-1][j-1]+1
            dist[1] = dist_array[i][j-1]+1
            dist[2] = dist_array[i-1][j]+1
            dist_array[i][j]=min(dist)
    
    return dist_array[i][j]

# self implemented jaccard similarity
#from math import*
def jaccard_similarity(vector1,vector2):
    jacc_num = 0 
    jacc_den = 0 
    for index in enumerate(vector1): 
        if vector1[index] != 0 or vector2[index] != 0: 
            jacc_den += max(vector1[index], vector2[index]) 
            jacc_num += min(vector1[index], vector2[index]) 
    return jacc_num / jacc_den

def common_tokens(string_1,string_2):
    """
    counts common word types. options are NOUN,VERB,ADV,ADJ
    """
    
    string_1 = str(string_1)
    string_2 = str(string_2)
    
    x = nltk.word_tokenize(string_1)
    y = nltk.word_tokenize(string_2)   
 
    common_tokens = len(list(set(x).intersection(y)))
    
    return(common_tokens)

def common_count(string_1,string_2, word_type = "NOUN"):
    """
    counts common word types. options are NOUN,VERB,ADV,ADJ
    """
    string_1 = str(string_1)
    string_2 = str(string_2)
    
    tagged_1 = nltk.pos_tag(nltk.word_tokenize(string_1),tagset='universal')
    tagged_2 = nltk.pos_tag(nltk.word_tokenize(string_2),tagset='universal')
    
    x = list([])
    for word in tagged_1:
        if word[1] == word_type:
            x.append(word[0])
                     
    y = list([])
    for word in tagged_2:
        if word[1] == word_type:
            y.append(word[0])
 
    common = len(list(set(x).intersection(y)))
    
    return(common)

def length_diff_characters(x,y):
    """
    find absolute difference in character length of two questions
    """
    x = str(x)
    y = str(y)
    return abs(len(y)-len(x))

def length_diff_tokens(x,y):
    """
    find absolute difference in number of tokens in two questions
    """
    return abs(len(nltk.word_tokenize(y))-len(nltk.word_tokenize(x)))

def common_numbers(x,y):
    """
    count numbers present in both tokens
    """

    x_numbers = re.findall(r'\b\d+\b', x)
    y_numbers = re.findall(r'\b\d+\b', y)
    
    common_numbers = len(list(set(x_numbers).intersection(y_numbers)))
    return(common_numbers)

def first_word(x,y):
    #first word
    word_list_x = x.split()  # list of words
    word_list_y = y.split()  # list of words
    return word_list_x[0] == word_list_y[0]

def last_word(x,y):
    #last word
    word_list_x = x.split()  # list of words
    word_list_y = y.split()  # list of words
    return word_list_x[-1] == word_list_y[-1]


def Add_features(df, scaler, col1, col2, save=False):
    df = df.copy()
    col1_to_transform = list(df.columns).index(col1)
    col2_to_transform = list(df.columns).index(col2)

    df['len_q1'] = [len(s) for s in df[col1]] 
    df['len_q2'] = [len(s) for s in df[col2]]
    df['len_q1'] = scaler.fit_transform(df[['len_q1']])
    df['len_q2'] = scaler.fit_transform(df[['len_q2']])
    print('len done')
    
    df['len_common'] = [len_common(df.iloc[i,col1_to_transform],df.iloc[i,col2_to_transform]) for i in range(len(df))]
    df['len_common'] = scaler.fit_transform(df[['len_common']])
    print('len_common done')

    df['len_not_common'] = [len_not_common(df.iloc[i,col1_to_transform],df.iloc[i,col2_to_transform]) for i in range(len(df))]
    df['len_not_common'] = scaler.fit_transform(df[['len_not_common']])
    print('len_not_common done')

    df['mean_dist_not_com'] = [mean_dist_not_com(df.iloc[i,col1_to_transform],df.iloc[i,col2_to_transform]) for i in range(len(df))]
    df['mean_dist_not_com'] = scaler.fit_transform(df[['mean_dist_not_com']])
    print('mean_dist_not_com done')

    df['both_number'] = [both_number(df.iloc[i,col1_to_transform],df.iloc[i,col2_to_transform]) for i in range(len(df))]
    print('both_number done')

    df["avg_len_q1"]= df[col1].apply(lambda x: average_len(x))
    df["avg_len_q2"]= df[col2].apply(lambda x: average_len(x))
    df['avg_len_q1'] = scaler.fit_transform(df[['avg_len_q1']])
    df['avg_len_q2'] = scaler.fit_transform(df[['avg_len_q2']])
    print('avg_len done')

    df['edit_distance'] = [editdistance.eval(df.iloc[i,col1_to_transform],df.iloc[i,col2_to_transform]) for i in range(len(df))]
    df['edit_distance'] = scaler.fit_transform(df[['edit_distance']])
    print('edit_distance done')

    df['levenshtein_distance'] = [levenshtein(df.iloc[i,col1_to_transform],df.iloc[i,col2_to_transform]) for i in range(len(df))]
    df['levenshtein_distance'] = scaler.fit_transform(df[['levenshtein_distance']])
    print('levenshtein_distance done')

    df["length_diff_characters"] = df.apply(lambda df: length_diff_characters(df[col1], df[col2]), axis=1)
    df['length_diff_characters'] = scaler.fit_transform(df[['length_diff_characters']])
    print('length_diff_characters done')

    df["length_diff_tokens"] = df.apply(lambda df: length_diff_tokens(df[col1], df[col2]), axis=1)
    df['length_diff_tokens'] = scaler.fit_transform(df[['length_diff_tokens']])
    print('length_diff_tokens done')

    df["common_numbers"] = df.apply(lambda df: length_diff_characters(df.question1, df.question2), axis=1)
    df['common_numbers'] = scaler.fit_transform(df[['common_numbers']])
    print('common_numbers done')

    df['first_word'] = df.apply(lambda df: first_word(df[col1], df[col2]), axis=1)*1
    df['last_word'] = df.apply(lambda df: last_word(df[col1], df[col2]), axis=1)*1
    print('first_word last_word done')

    if save==True:
        df.to_csv('features_added_Quora_full_train_data.csv')
    
    return df


def fit_on_q1_q2(df, model, documents_type1, documents_type2):
    q_list1 = list(df[documents_type1])
    q_list2 = list(df[documents_type2])
    all_questions = q_list1 + q_list2 
    model.fit(all_questions)
    return

def get_features_from_df(df, count_vectorizer):
    """
    returns a sparse matrix containing the features build by the count vectorizer.
    Each row should contain features from question1 and question2.
    """
    q1_casted =  cast_list_as_strings(list(df["question1_cleaned"]))
    q2_casted =  cast_list_as_strings(list(df["question1_cleaned"]))
    
    ############### Begin exercise ###################
    # what is kaggle                  q1
    # What is the kaggle platform     q2
    X_q1 = count_vectorizer.transform(q1_casted)
    X_q2 = count_vectorizer.transform(q2_casted)    
    X_q1q2 = scipy.sparse.hstack((X_q1,X_q2))
    ############### End exercise ###################

    return X_q1q2


def get_tfidf(df, tfidf, documents_type1, documents_type2, sim=True):
    q1_casted =  cast_list_as_strings(list(df[documents_type1]))
    q2_casted =  cast_list_as_strings(list(df[documents_type2]))
    
    tfidf_q1 = tfidf.transform(q1_casted)
    tfidf_q2 = tfidf.transform(q2_casted)
    tfidf_q1q2 = scipy.sparse.hstack((tfidf_q1,tfidf_q2))
    if sim == True:
        sims = []
        for i in range(len(q1_casted)):
            sims.append(cosine_similarity(tfidf_q1[i,:],tfidf_q2[i,:]))
        sims = np.reshape(sims, (len(q1_casted), 1))

        return scipy.sparse.hstack((tfidf_q1q2,sims)).tocsr() 
    else:
        return tfidf_q1q2.tocsr()


def vectorize_train_val_test(df, get_feat_model, vectorizer_func, documents_type1, documents_type2):
    
    train_df, test_df = sklearn.model_selection.train_test_split(df, test_size=0.05,random_state=123)
    train_df, val_df  = sklearn.model_selection.train_test_split(train_df, test_size=0.05,random_state=123)
    X_train_q1q2      = get_feat_model(train_df, vectorizer_func, documents_type1, documents_type2)
    X_val_q1q2        = get_feat_model(val_df, vectorizer_func, documents_type1, documents_type2)
    X_test_q1q2       = get_feat_model(test_df, vectorizer_func, documents_type1, documents_type2)
    
    return  train_df, val_df, test_df, X_train_q1q2, X_val_q1q2, X_test_q1q2


def add_a_column_feat(data, col, sparse_matrix):
    feat_q = data[col].to_numpy().reshape(len(data[col]),1)
        
    return scipy.sparse.hstack((sparse_matrix,feat_q)).tocsr()        

def train_models_plus_feat(X_train_q1q2, X_val_q1q2, X_test_q1q2, train_df, val_df, test_df, col_list):    
    
    logistic = sklearn.linear_model.LogisticRegression(solver="liblinear",
                                                       random_state=123)

    xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, 
                              learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, 
                              objective='binary:logistic', eta=0.3, silent=1, subsample=0.8, random_state=123)

    
    for col in col_list:
        X_train_q1q2      = add_a_column_feat(train_df, col, X_train_q1q2)
        X_val_q1q2        = add_a_column_feat(val_df, col, X_val_q1q2)
        X_test_q1q2       = add_a_column_feat(test_df, col, X_test_q1q2)
    print('...features added...')

    y_train           = train_df["is_duplicate"].values
    y_val             = val_df["is_duplicate"].values
    y_test            = test_df["is_duplicate"].values
    
    logistic.fit(X_train_q1q2, y_train)
    print('...logistic model fitted...')

    xgb_model.fit(X_train_q1q2, y_train) 
    print('...model fitted...')

    logistic_train_acc = roc_auc_score(y_train, logistic.predict_proba(X_train_q1q2)[:, 1])                                                       
    logistic_val_acc   = roc_auc_score(y_val, logistic.predict_proba(X_val_q1q2)[:, 1])
    logistic_test_acc  = roc_auc_score(y_test, logistic.predict_proba(X_test_q1q2)[:, 1])
    print('logistic_train_acc:{}, logistic_val_acc:{}, logistic_test_acc:{}'.format(logistic_train_acc, logistic_val_acc, logistic_test_acc))
                                                   
    xgb_train_acc      = roc_auc_score(y_train, xgb_model.predict_proba(X_train_q1q2)[:, 1])
    xgb_val_acc        = roc_auc_score(y_val, xgb_model.predict_proba(X_val_q1q2)[:, 1])
    xgb_test_acc       = roc_auc_score(y_test, xgb_model.predict_proba(X_test_q1q2)[:, 1])
    print('xgb_train_acc:{}, xgb_val_acc:{}, xgb_test_acc:{}'.format(xgb_train_acc, xgb_val_acc, xgb_test_acc))
                                                   
    return logistic, xgb_model, [logistic_train_acc, logistic_val_acc, logistic_test_acc], [xgb_train_acc, xgb_val_acc, xgb_test_acc]



    



