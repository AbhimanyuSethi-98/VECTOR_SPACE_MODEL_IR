
from __future__ import unicode_literals
import spacy
from bs4 import BeautifulSoup
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from collections import Counter
import math
import numpy
import json
import time
import pandas as pd
from scipy import spatial

import fasttext
import fasttext.util

import numpy as np
from numpy import dot
from numpy.linalg import norm
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet
# nltk.download('punkt')
# nltk.download('wordnet')
# from nltk.util import ngrams
import pickle

########################################  install SPACY   ########################################
######################################## python -m spacy download en_core_web_sm ########################################
######################################## python -m spacy download en_core_web_lg ########################################

def query_pre_process(query):
    '''
    input: string of raw query terms. 
            
    output: 
    '''
    
    punc = [w for w in string.punctuation if not (w=='\'' or w=='-' or w=='%' or w==':')]
    punc = ''.join(punc)
    table = str.maketrans('', '', punc)
    new_query = word_tokenize(query)
    new_query = [w.translate(table) for w in new_query]
    new_query = [x.lower() for x in new_query]
    new_query = [w for w in new_query if not (w=="''" or w=='' or w=="' '")]

    return new_query

def get_query_terms(query):
    return Counter(query)


def get_normalized_query_scores(query_terms, freq_list, inverted_index):
    '''
    returns query score for each query term (l.t.c)
    '''

    tf_weights = {}

    for term in query_terms:
        tf_weights[term] = 1 + math.log10(query_terms[term])

    idf = {}
    # idf has the weights corresponding to query temrs their frequency in documents
    N = len(freq_list)    # no of document in corpus
    for term in query_terms:
        if term in inverted_index.keys():
            idf[term] = math.log10( N / len(inverted_index[term]))
        else:
            idf[term] = 0

    query_tf_idf = {}
    

    for term in query_terms:
        query_tf_idf[term] = idf[term]*tf_weights[term]

    cos_factor = math.sqrt(sum([x**2 for x in query_tf_idf.values()]))
    
    if cos_factor != 0:
        cos_factor= 1/cos_factor

    for term in query_tf_idf:
        query_tf_idf[term] = cos_factor * query_tf_idf[term];

    return query_tf_idf



def get_normalized_doc_weights(query_terms, freq_list, inverted_index):
    
    doc_weights = [[] for i in range(len(freq_list))]

    for i in range(len(freq_list)):
        for term in freq_list[i].keys():
            val = freq_list[i][term]
            doc_weights[i].append([term, 1 + math.log10(val)])
            # doc_weights[i] is term and its unigram score for ith document in the log and both of them are stored as a pair

    normalized_doc_weights = [[] for i in range(len(doc_weights))]

    for i in range(len(doc_weights)):
        doc_tf = doc_weights[i]

        square_sum = math.sqrt(sum( [v[1]**2 for v in doc_tf]))
        if square_sum != 0:
            factor = 1 / square_sum

        for j in range(len(doc_tf)):
            normalized_doc_weights[i].append([doc_tf[j][0], doc_tf[j][1]*factor])
    
    return normalized_doc_weights




def get_query_term_weight(term, term_weights):
    if term in term_weights.keys():
        return term_weights[term]
    else:
        return 0

def compute_scores(query_wt, document_wt):

    scores = [[i, 0] for i in range(len(document_wt))]

    for i in range(len(document_wt)):
        
        doc_tf = document_wt[i]

        score = 0

        for j in range(len(doc_tf)):
            term = doc_tf[j][0]
            term_weight = get_query_term_weight(term, query_wt)

            score += term_weight*doc_tf[j][1]

        scores[i] = [i, score]

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # print(scores[:10])

    return scores

def compute_BM25_scores(queryTerms, freq_list, inverted_index, k, b):
    N  = len(freq_list)
    length_av = 0
    for doc in freq_list:
        l = 0
        for key in doc:
            l += doc[key]
        length_av += l
    length_av /= N	# average lenth of a document
    RSV = [[i, 0] for i in range(0, N)]
    for i in range(0, len(freq_list)):
        # doc_freq = {}
        doc_freq = dict(freq_list[i])
        score = 0
        length_doc = 0
        for key in doc_freq:
            length_doc += doc_freq[key]
        
        for term in queryTerms:
            if term in doc_freq:
                df = len(inverted_index[term])
                tf = doc_freq[term]
                temp_score = math.log10(N/df)* (k+1)*tf
                # length_doc = len(processed_text[i])
                temp_score /= k*((1-b) +b*length_doc/length_av) + tf 
                score+= temp_score
        RSV[i]=[i, score]
        
    RSV = sorted(RSV, key=lambda x: x[1], reverse=True)
    return RSV




def search(query, inverted_index, freq, title_list):
    query_processed = query_pre_process(query)
    query_terms = get_query_terms(query_processed)
    print("-"*50)
    print("Query Terms: ", query_terms)

    query_wt = get_normalized_query_scores(query_terms, freq, inverted_index)
    document_wt = get_normalized_doc_weights(query_terms, freq, inverted_index)

    scores = compute_scores(query_wt, document_wt)

    print("\nPART1: The top 10 documents matching with the query '", query, "' are:\n")
    
    for i in range(10):
        if i == len(title_list):
            break
        print(str(i+1) + ". DocumentID: " + (str(scores[i][0])).ljust(8) + " Score: " + (str(round(scores[i][1], 3))).ljust(8) + " Title: " + str(title_list[scores[i][0]])) 
    print("-"*50)
    print('\n')


def improved1(query, inverted_index, freq, title_list):
    '''
    BM25
    '''
    query_processed = query_pre_process(query)
    query_terms = get_query_terms(query_processed)
    print("-"*50)
    print("Query Terms: ", query_terms)
    
    k = 0.5
    b = 0.5
    
    scores = compute_BM25_scores(query_terms, freq,  inverted_index, k, b)
    print("\nIMPROVEMENT 1: The top 10 documents matching with the query '",query, "' are:\n")
    
    for i in range(10):
        if i == len(title_list):
            break
        print(str(i+1) + ". DocumentID: " + (str(scores[i][0])).ljust(5) + ", Score: " + (str(round(scores[i][1], 3))).ljust(5) + ", Title: " + str(title_list[scores[i][0]])) 
    print("-"*50)
    print('\n')







# cosine similarity
def cosine(v1, v2):
    if norm(v1) > 0 and norm(v2) > 0:
        return dot(v1, v2) / (norm(v1) * norm(v2))
    else:
        return 0.0


def improved2(query, inverted_index, freq, title_list):
    '''
    input query: 
    	inverted_index: 
    	freq: freq of words 
    	title_list: dictionary mapping document id to its title
    '''

    # Here we are using Fasttest to get most similar word of a query term to enrich the query vector and give better results. 

    fasttext.util.download_model('en', if_exists='ignore')  # English
    ft = fasttext.load_model('cc.en.300.bin')
    queryL = Counter()
    n = 3	# Hyper parameter, we have chosen 2 most related words in corpus + the query term itself.
    # # thus total no of query terms now are: 3*M if M was initial query length
    
    query_processed = query_pre_process(query)
    query_terms = get_query_terms(query_processed)	# a dictionary of counter
    eng_stopwords = stopwords.words()
    count=1
    for word in query_terms:
        if not word in eng_stopwords:
            temp_list = set([word])
            for item in ft.get_nearest_neighbors(word):
                if item[1].isalpha() and item[1].lower() not in temp_list:
                    print(item)
                    count+=1
                    temp_list.add(item[1].lower())
                    if count==n:
                        count=1
                        break

            for i in list(temp_list) :
                queryL[i]=1
            queryL[word] = query_terms[word]
    
    '''
    search(queryL, inverted_index, freq, title_list)
    '''
    query_terms = queryL
    print("-"*50)
    print("Query Terms: "+ " ".join(query_terms))

    query_wt = get_normalized_query_scores(query_terms, freq, inverted_index)
    document_wt = get_normalized_doc_weights(query_terms, freq, inverted_index)

    scores = compute_scores(query_wt, document_wt)

    print("\nIMPROVEMENT2 [with model1 search()]: The top 10 documents matching with the query '", query, "' are:\n")
    
    for i in range(10):
        if i == len(title_list):
            break
        print(str(i+1) + ". DocumentID: " + (str(scores[i][0])).ljust(5) + ", Score: " + (str(round(scores[i][1], 3))).ljust(5) + ", Title: " + str(title_list[scores[i][0]])) 
    print("-"*50)
    print('\n')

    
    '''
    improved1(queryL, inverted_index, freq, title_list)   
    '''
    '''
    print("-"*50)
    print("Query Terms: ", query_terms)
    
    k = 0.5
    b = 0.5
    
    scores = compute_BM25_scores(query_terms, freq,  inverted_index, k, b)
    print("\nIMPROVEMENT2 with improvement1(): The top 10 documents matching with the query '",query, "' are:\n")
    
    for i in range(10):
        if i == len(title_list):
            break
        print(str(i+1) + ". DocumentID: " + (str(scores[i][0])).ljust(5) + ", Score: " + (str(round(scores[i][1], 3))).ljust(5) + ", Title: " + str(title_list[scores[i][0]])) 
    print("-"*50)
    print('\n')
    '''
            
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
    document_wt = get_normalized_doc_weights(query_terms, cl_freq, championLists)

    scores = compute_scores(query_wt, document_wt)

    for i in range(top_k):
        if i == len(title_list):
            break
        print(str(i+1) + ". DocumentID: " + (str(cl_doc_ids[scores[i][0]])).ljust(5) + ", Score: " + (str(round(scores[i][1], 3))).ljust(5) + ", Title: " + str(title_list[cl_doc_ids[scores[i][0]]]))

def improved2Robust(query, inverted_index, freq, title_list, pklFileName):
    # robust version of improved2() Please check that, only dictionary d is new and enforced
    
    tokens = inverted_index.keys()
    queryL = Counter()
    #n = 3	# Hyper parameter, we have chosen 2 most related words in corpus + the query term itself.
    # thus total no of query terms now are: 3*M if M was initial query length
    
    query_processed = query_pre_process(query)
    query_terms = get_query_terms(query_processed)	# a dictionary of counter 
    
    #######
    '''
    d = {}
    d['me'] =  ['me', 'me—and', 'myself']
    d['enlighten'] =  ['enlighten', 'expound', 'inspire']
    d['viral'] = ['viral', 'virus', 'infection']
    d['foreigner'] = ['foreigner', 'foreigners', 'nationality']
    d['main'] = ['main', 'main-sequence', "'main"]
    d['cause'] = ['cause', 'causes', 'causing']
    d['of'] = ['of', "'of", 'the']
    d['poverty'] = ['poverty', 'poverty…by','inequality' ]
    '''
    ## OR
    
    with open(pklFileName, 'rb') as f:
        d1 = pickle.load(f)
    
    #######
    
    
    for str1 in query_terms :
        ltemp = []
        #print(str1)
        ltemp = d1[str1] 
        for i in ltemp :
            queryL[i]=1
        queryL[str1] = query_terms[str1]
    '''
    search(queryL, inverted_index, freq, title_list)
    '''
    query_terms = queryL
    print("-"*50)
    print("Query Terms: ", query_terms)

    query_wt = get_normalized_query_scores(query_terms, freq, inverted_index)
    document_wt = get_normalized_doc_weights(query_terms, freq, inverted_index)

    scores = compute_scores(query_wt, document_wt)

    print("\nIMPROVEMENT2 'robust' with model1 search(): The top 10 documents matching with the query '", query, "' are:\n")
    
    for i in range(10):
        if i == len(title_list):
            break
        print(str(i+1) + ". DocumentID: " + (str(scores[i][0])).ljust(5) + " Score: " + (str(round(scores[i][1], 3))).ljust(5) + " Title: " + str(title_list[scores[i][0]])) 
    print("-"*50)
    print('\n')


    
def main():
    
    folder = input('<Enter folder storing the index files (ex- indexFiles)>:\n')
    # default folder name is indexFiles which stores all the indices created by indexing.py
    inverted_index = {}
    freq = []
    title_list = []
    championLists = {}
    # these datastructures will be filled by reading the index files

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
        #print(query)
        # takes query as a string
    
        
        # this file stores the dictionary(like invertd index) but instead of posting list as values, its values are list of similar words
        # ex: dict['enlighten'] =  ['enlighten', 'expound', 'inspire']
        #pklFileName = folder+'/relatedWords.pickle'
        
        # which out of 5+1 options to be executed on the query  		   
        option = input('<Enter Option:- \n\t1:Normal Part1 retreival, \n\t2:Improvement1, \n\t3:Improvement2, \n\t4:All three, \n\t5:All three but Lengthy \n\t6:championLists\n\t0:exit>\n')
        startT = time.time()	# to check total time taken
        if option=='1' :
            search(query, inverted_index, freq, title_list)		# model1  retreival model (tf-idf)
        elif option=='2' :
            improved1(query, inverted_index, freq, title_list)	# imporvement 1
        elif option=='3' :
            improved2(query, inverted_index, freq, title_list)	# improvement 2 (robust), quicker with the help of index created
            
        elif option=='4' :
            search(query, inverted_index, freq, title_list)		# all 3 tables of 10 docs each retrieved above, together
            improved1(query, inverted_index, freq, title_list)
            improved2Robust(query, inverted_index, freq, title_list, pklFileName)
                	
        elif option=='5' :						# this option uses spacy library to find in real time related terms to query terms from the corpus itself
            								# It is slower (as calculates every time) (better version is Option 4)
            								# uses GLovE vector representation					
            print('Are you sure? This will take around 9 minutes per query word Y/N')
            x = input()
            while(x!='Y' and x!='N' and x=='y' and x=='n') :
                print('Please input either "Y" or "N"')          
            if(x=='Y' or x=='y') :					# giving last chance to avoid lengthy alternative
                search(query, inverted_index, freq, title_list)
                improved1(query, inverted_index, freq, title_list)	
                improved2(query, inverted_index, freq, title_list)
            elif(x=='N' or x=='n') :
                search(query, inverted_index, freq, title_list)
                improved1(query, inverted_index, freq, title_list)	
        
        elif option=='6' :
            #res = 
            champion_list(query_pre_process(query), inverted_index, championLists, freq, 10,title_list)#doc_lnc_df,query_ltc_df,10)
            # for doc in res:
            #     #print(doc)
            #     print("DocumentID: " + str(doc[0]).ljust(8) + " Score: " + (str(round(doc[1], 3))).ljust(8) + " Title: " + str(title_list[doc[0]]))

        elif option=='0' :
            break
        
        
        print("Time Taken= %s seconds" %(time.time()-startT))
        reply = input('\nDo you want to search something else (y/n)\n')
        if reply == 'n' or reply=="N":
            break





   


if __name__ == "__main__":
    
    main()	## call the main()





