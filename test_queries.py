from termcolor import colored
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from collections import Counter
import math
import json
import time

import fasttext
import fasttext.util

import numpy as np
from numpy import dot
from numpy.linalg import norm
from autocorrect import Speller
spell = Speller(lang='en')

def query_pre_process(query):
    '''
    Used to tokenize the query

    input: Raw query string 
            
    output: Query tokens
    '''
    
    punc = [ch for ch in string.punctuation if not (ch=='\'' or ch=='-' or ch=='%' or ch==':')]
    punc = ''.join(punc)
    table = str.maketrans('', '', punc)
    new_query = word_tokenize(query)
    new_query = [term.translate(table) for term in new_query]
    new_query = [term.lower() for term in new_query]
    new_query = [ch for ch in new_query if not (ch=="''" or ch=='' or ch=="' '")]
    return new_query

def get_query_terms(query):
    """
    Returns a dictionary with keys as query tokens and values as term frequency in the query
    """
    return Counter(query)


def get_normalized_query_scores(query_terms, freq_list, inverted_index):
    '''
    returns query score for each query term (l.t.c)
    '''

    tf_weights = {}
    #Calculating the tf weights
    for term in query_terms:
        tf_weights[term] = 1 + math.log10(query_terms[term])

    idf = {}
    # Calculating idf weights
    # Total documents in corpus
    N = len(freq_list)
    for term in query_terms:
        if term in inverted_index.keys():
            idf[term] = math.log10( N / len(inverted_index[term]))
        else:
            idf[term] = 0

    query_tf_idf = {}
    
    # Calculating Tf-Idf
    for term in query_terms:
        query_tf_idf[term] = idf[term]*tf_weights[term]

    # Calculating the norm for the denominator of cosine
    normalize = norm(list(query_tf_idf.values()))
    if normalize != 0:
        normalize= 1/normalize

    #Normalising the query weights
    for term in query_tf_idf:
        query_tf_idf[term] = normalize * query_tf_idf[term]

    return query_tf_idf


def get_normalized_doc_weights(freq_list, inverted_index):
    '''
    returns document score (l.n.c)
    '''
    N = len(freq_list)
    doc_weights = [[] for i in range(N)]

    for i in range(len(freq_list)):
        for term in freq_list[i].keys():
            val = freq_list[i][term]
            doc_weights[i].append([term, 1 + math.log10(val)])
            # doc_weights[i] is term and its unigram score for ith document in the log and both of them are stored as a pair

    normalized_doc_weights = [[] for i in range(N)]

    for i in range(N):
        normalize = math.sqrt(sum( [v[1]**2 for v in doc_weights[i]]))
        if normalize != 0:
            normalize = 1 / normalize

        for j in range(len(doc_weights[i])):
            normalized_doc_weights[i].append([doc_weights[i][j][0], doc_weights[i][j][1]*normalize])
    
    return normalized_doc_weights


def get_query_term_weight(term, term_weights):
    """
    Returns the weight of query terms
    """
    if term in term_weights.keys():
        return term_weights[term]
    else:
        return 0

def compute_normal_scores(query_wt, document_wt):
    """
    Returns the score of a given query on the whole corpus of documents (lnc.ltc)
    """
    scores = [[i, 0] for i in range(len(document_wt))]

    for i in range(len(document_wt)):
        doc_tf = document_wt[i]
        score = 0
        for j in range(len(doc_tf)):
            term = doc_tf[j][0]
            term_weight = get_query_term_weight(term, query_wt)
            score += term_weight*doc_tf[j][1]

        scores[i] = [i, score]
    # Sort the score in decreasing order
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores

def compute_BM25_scores(queryTerms, freq_list, inverted_index, k=0.5, b=0.5):
    """
    Returns the BM_25 score of a given query on the whole corpus of documents
    """
    N  = len(freq_list)
    length_av = 0
    for doc in freq_list:
        l = 0
        for key in doc:
            l += doc[key]
        length_av += l
    #Calculated the average length of documents
    length_av /= N
    rsv = [[i, 0] for i in range(0, N)]
    for i in range(0, len(freq_list)):
        doc_freq = dict(freq_list[i])
        score = 0
        length_doc = 0
        for key in doc_freq:
            length_doc += doc_freq[key]
        
        for term in queryTerms:
            if term in doc_freq:
                df = len(inverted_index[term])
                tf = doc_freq[term]
                temp_score = math.log10(N/df) * (k+1) * tf
                temp_score /= k*((1-b) +b*length_doc/length_av) + tf 
                score+= temp_score
        rsv[i]=[i, score]
        
    rsv = sorted(rsv, key=lambda x: x[1], reverse=True)
    return rsv


def search(query, inverted_index, freq, title_list):
    """
    Calculating document score using normal scoring system (lnc.ltc)
    """
    query_processed = query_pre_process(query)
    query_terms = get_query_terms(query_processed)
    print("_"*50)
    print("Query Terms: ", query_terms)

    query_wt = get_normalized_query_scores(query_terms, freq, inverted_index)
    document_wt = get_normalized_doc_weights(freq, inverted_index)

    scores = compute_normal_scores(query_wt, document_wt)

    print("\nPART1 NORMAL: The top 10 documents matching with the query " + query + " are:\n")
    
    for i in range(10):
        if i == len(title_list):
            break
        print(colored(str(i+1) + ". DocumentID: " + (str(scores[i][0])).ljust(8) + " Score: " + (str(round(scores[i][1], 3))).ljust(8) + " Title: " + str(title_list[scores[i][0]]),'blue')) 
    print("_"*50)
    print('\n')


def bm25_improved1(query, inverted_index, freq, title_list):
    '''
    BM25
    '''
    query_processed = query_pre_process(query)
    query_terms = get_query_terms(query_processed)
    print("_"*50)
    print("Query Terms: ", query_terms)
    scores = compute_BM25_scores(query_terms, freq,  inverted_index)
    print("\nBM25: The top 10 documents matching with the query " + query +" are:\n")
    
    for i in range(10):
        if i == len(title_list):
            break
        print(colored(str(i+1) + ". DocumentID: " + (str(scores[i][0])).ljust(5) + ", Score: " + (str(round(scores[i][1], 3))).ljust(5) + ", Title: " + str(title_list[scores[i][0]]),'red')) 
    print("_"*50)
    print('\n')



def fasttext_improved2(query, inverted_index, freq, title_list):
    '''
    Fasttext to enrich the query with similar words 
    input query: 
    	inverted_index: 
    	freq: freq of words 
    	title_list: dictionary mapping document id to its title
    '''

    # Here we are using Fasttest to get most similar word of a query term to enrich the query vector and give better results. 

    fasttext.util.download_model('en', if_exists='ignore')  # English
    ft = fasttext.load_model('cc.en.300.bin')
    queryL = Counter()
    n = 11	# Hyper parameter, we have chosen 10 most related words + the query term itself. 
    """ It cannot be more than 11. """
    # We don't want the model to unnecessarily find similar word for stop words so we remove them
    query_processed = query_pre_process(query)
    query_terms = get_query_terms(query_processed)
    eng_stopwords = stopwords.words()
    count=1
    for word in query_terms:
        if not word in eng_stopwords:
            temp_list = set([word])
            for item in ft.get_nearest_neighbors(word):
                if item[1].isalpha():
                    # as fasttext also gives words with incorrect spellings so we correct these spellings using autocorrect library
                    correct_item = spell(item[1]).lower()
                    if correct_item not in temp_list:
                        count+=1
                        temp_list.add(correct_item)
                        if count==n:
                            count=1
                            break

            for i in list(temp_list):
                queryL[i]=1
        queryL[word] = query_terms[word]

    query_terms = queryL
    print("_"*50)
    print("Query Terms: "+ " ".join(query_terms))

    query_wt = get_normalized_query_scores(query_terms, freq, inverted_index)
    document_wt = get_normalized_doc_weights(freq, inverted_index)

    scores = compute_normal_scores(query_wt, document_wt)

    print("\nFasttext: The top 10 documents matching with the query " + query + " are:\n")
    
    for i in range(10):
        if i == len(title_list):
            break
        print(colored(str(i+1) + ". DocumentID: " + (str(scores[i][0])).ljust(5) + ", Score: " + (str(round(scores[i][1], 3))).ljust(5) + ", Title: " + str(title_list[scores[i][0]]),'green')) 
    print("_"*50)
    print('\n')

            
def champion_list(query_terms, inverted_index,championLists, freq, top_k,title_list): # doc_lnc_df,query_ltc_df
    '''
    Function implementing Improvement #1: Champion Lists
    Returns 'top_k' Document ID's based on cosine similarity.
    '''

    query_terms  = [ word for word in query_terms if word in inverted_index.keys()] #List of terms that are in query as well as posting lists
    query_terms = get_query_terms(query_terms)
    cl_doc_ids = set()

    for word in query_terms:
        for k in championLists[word]:
            cl_doc_ids.add(k[0]) #we take the union of the champion lists for each of the terms comprising the query. 
            #fWe now restrict cosine computation to only these documents
    #print(cl_doc_ids)
    cl_doc_ids = list(cl_doc_ids)
    cl_freq = [freq[i] for i in cl_doc_ids]

    query_wt = get_normalized_query_scores(query_terms, cl_freq, championLists)
    document_wt = get_normalized_doc_weights(cl_freq, championLists)

    scores = compute_normal_scores(query_wt, document_wt)
    print("\nCHAMPION LIST: The top 10 documents matching with the query")
    for i in range(top_k):
        if i == len(title_list):
            break
        print(colored(str(i+1) + ". DocumentID: " + (str(cl_doc_ids[scores[i][0]])).ljust(5) + ", Score: " + (str(round(scores[i][1], 3))).ljust(5) + ", Title: " + str(title_list[cl_doc_ids[scores[i][0]]]),'yellow'))

def fasttext_bm25(query, inverted_index, freq, title_list):
    """ Fasttext + BM25 """
    fasttext.util.download_model('en', if_exists='ignore')  # English
    ft = fasttext.load_model('cc.en.300.bin')
    queryL = Counter()
    n = 11	# Hyper parameter, we have chosen 10 most related words + the query term itself. 
    """ It cannot be more than 11. """
    # We don't want the model to unnecessarily find similar word for stop words so we remove them
    
    query_processed = query_pre_process(query)
    query_terms = get_query_terms(query_processed)
    eng_stopwords = stopwords.words()
    count=1
    for word in query_terms:
        if not word in eng_stopwords:
            temp_list = set([word])
            for item in ft.get_nearest_neighbors(word):
                if item[1].isalpha():
                    correct_item = spell(item[1]).lower()
                    if correct_item not in temp_list:
                        count+=1
                        temp_list.add(correct_item)
                        if count==n:
                            count=1
                            break

            for i in list(temp_list) :
                queryL[i]=1
        queryL[word] = query_terms[word]
    
    query_terms = queryL
    print("_"*50)
    print("Query Terms: "+ " ".join(query_terms))
    scores = compute_BM25_scores(query_terms, freq,  inverted_index)
    print("\nBM25+Fasttext: The top 10 documents matching with the query " +query+ " are:\n")
    
    for i in range(10):
        if i == len(title_list):
            break
        print(str(i+1) + ". DocumentID: " + (str(scores[i][0])).ljust(5) + ", Score: " + (str(round(scores[i][1], 3))).ljust(5) + ", Title: " + str(title_list[scores[i][0]])) 
    print("_"*50)
    print('\n')

    
def main():
    
    folder = input('<Enter folder storing the index files (ex- indexFiles)>:\n')
    # default folder name is indexFiles which stores all the indices created by indexing.py
    inverted_index = {}
    freq = []
    title_list = []
    championLists = {}
    # Data Structures will be filled by reading the index files

    with open(folder+'/inverted_index_dict.json') as f1:
        inverted_index = json.load(f1)

    with open(folder+'/freq_list.json') as f2:
        freq = json.load(f2)

    with open(folder+'/titles_list.json') as f3:
        title_list = json.load(f3)

    with open(folder+'/champ_list.json') as f4:
        championLists = json.load(f4)

    while 1:
        query = input('<Enter your query:>\n')		   
        option = input('<Enter Option:- \n\t1:Normal Part1 retrieval, \n\t2:Improvement1 BM25, \n\t3:Improvement2 Fasttext, \n\t4:championLists, \n\t5:BM25 + Fasttext, \n\t6:All five \n\t0:exit>\n')
        startTime = time.time()	# to check total time taken
        if option=='1':
            search(query, inverted_index, freq, title_list)		# model1  retrieval model (tf-idf)
        elif option=='2':
            bm25_improved1(query, inverted_index, freq, title_list)	# BM25
        elif option=='3':
            fasttext_improved2(query, inverted_index, freq, title_list)	# Fasttext            
        elif option=='4': 
            champion_list(query_pre_process(query), inverted_index, championLists, freq, 10,title_list) #doc_lnc_df,query_ltc_df,10)
                	
        elif option=='5':                                      #Improved 3 model BM25 + Fasttext
            fasttext_bm25(query, inverted_index, freq, title_list)
        
        elif option=='6':
            search(query, inverted_index, freq, title_list)		# all 5 tables of 10 docs each retrieved above, together
            bm25_improved1(query, inverted_index, freq, title_list)
            fasttext_improved2(query, inverted_index, freq, title_list)
            fasttext_bm25(query, inverted_index, freq, title_list)
            champion_list(query_pre_process(query), inverted_index, championLists, freq, 10, title_list)

        elif option=='0' :
            break
        
        
        print("Time Taken= %s seconds" %(time.time()-startTime))
        reply = input('\nDo you want to search something else (y/n)\n')
        if reply == 'n' or reply=="N":
            break   

if __name__ == "__main__":
    main()	# start the program