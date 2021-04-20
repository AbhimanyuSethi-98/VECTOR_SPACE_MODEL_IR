from bs4 import BeautifulSoup
# import nltk
import json
import math
import numpy
import os
import sys
import pickle
import string
from collections import Counter
from nltk.tokenize import word_tokenize

def extract(filename = ''):
    '''
    returns 
    doc_list: list of wikipedia documents where doc_list[i] is the extracted text of ith document
    and
    titles_list: where titles_list[i] is title of ith document in the file(say in wiki_00)
    '''
    file = open(filename, encoding="utf8")
    soup = BeautifulSoup(file, features="html.parser")
    docs=soup.findAll("doc")	#Each document in the dataset is present under </doc> tags
    titles_list = []
    titles_list=[x["title"] for x in docs]
    doc_list = [BeautifulSoup(str(doc), features="html.parser").get_text() for doc in docs]
    return doc_list,titles_list


def pre_processing(docs):
    '''
    Function to take care of preprocessing on the input documents from the wikipedia dataset
    input: list of raw text parsed by the html parser. 
           docs[i] corresponding to the raw text of the ith document 
    returns: tokens[i] is a list of tokens of the ith document, and
            tokens[i][j] is jth token of ith document
    '''
    tokens = []
    punct = [w for w in string.punctuation if not (w=='\'' or w=='%' or w==':' or w=='-')]	# to remove all standard punctuations except '-', etc
    punct = ''.join(punct)
    mapping = str.maketrans('', '', punct)	# mapping from each punctuation to be removed to None, in order to replace them with None in the text
    for doc in docs:
        doc_tokens = word_tokenize(doc)	# tokenizing each document into a list of tokens
        doc_tokens = [w.translate(mapping) for w in doc_tokens]	# punctuations removal
        doc_tokens = [x.lower() for x in doc_tokens]	# converting all tokens to lower case
        doc_tokens = [w for w in doc_tokens if not (w=="''" or w=='' or w=="' '")]	# removing spaces and empty quotes
        tokens.append(doc_tokens)
    return tokens



def generate_term_frequencies(docs):
    '''
    input: list of (list of tokens)
    output: list of dictionaries keeping tf count per document
    termFrequency[0]['in']=43 // 0th doc contains 'in' 43 times, or term frequency of "in" for doc 0 is 43
    '''
    termFrequencies = [Counter(doc) for doc in docs] # doc is 1 document , docs conains multiple docs
    # "Counter creates a dictionary for each document, with
    # KEY: term in the document, and VALUE: term frequency(tf) in that document
    # termFrequencies is list of such dictionaries 
    return termFrequencies


def generate_inverted_index(termFrequencies):
    #create set from list of tokenized docs    
    '''
    input: list of dictionaries keeping tf count per document [tf(t,d)] 
    outputs: a dictionary with 
    KEYS: Unique terms in the corpus, and VALUE: list of [Document_ID,Term frequency] lists for each document ID
    '''
    inverted_index = {}
    for i in range(len(termFrequencies)): # i.e., number of documents
        doc = termFrequencies[i] # A single document
        for key in doc.keys(): #denoting a term in the voacbulary
            if key in inverted_index.keys(): 
                inverted_index[key].append((i, doc[key]))	# if term already added to index, then append this new [doc_id,tf] valueto the key
            else:
                inverted_index[key] = []
                inverted_index[key].append((i, doc[key]))	# if its a new term not seen before, create an empty list and then append

    return inverted_index

def generate_champion_lists(inverted_index,top_r): # doc_lnc_df,query_ltc_df
    '''
    Function implementing Improvement #1: Champion Lists
    Returns 'top_r' Document ID's based on cosine similarity.
    '''
    championLists = {}
    for word in inverted_index.keys():
        championLists[word] = sorted(inverted_index[word], key=lambda x:x[1])[::-1] #In decreasing order of tf
        # print(word + ": ")
        # print(championLists[word])
        # print("\n")
        championLists[word] = championLists[word][:top_r]
    #print(championLists)

    return championLists




def main(arg1, arg2):
    # Optional functionality to provide 2 command line arguments together. If none, then defaults are used as input
    # arg1 contains relative path of corpus file (input data). Default : ./data/wiki_00
    # arg2 contains name of the directory that will be created(if doesn't exist already) to store the 
    # JSON files containing the inverted index, frequency list, and titles' list describing the corpus. Default : ./genFiles
    
    filepath = ''
    filename = filepath + arg1
    docs_list,titles_list = extract(filename)	# docs_list: where docs_list[i] is the extracted text of ith document
    #titles_list: where titles_list[i] is the extracted title of ith document
    processed_docs = pre_processing(docs_list) # processed_docs[i] is a list of tokrns of extracted ith document after preprocessing,
            						 # processed_docs[i][j] is jth token of ith document
    tfs = generate_term_frequencies(processed_docs) #tfs denotes a list of document-wise frequencies of unigrams as dictionaries (term frequencies) 
    inverted_index = generate_inverted_index(tfs) # the constructed inverted index dictionary with 
    #KEY: Term itself, and VALUE:list of [Document_ID,Term frequency] lists | ex: "finance": [[0, 3], [60, 1], .....]
    championLists = generate_champion_lists(inverted_index,15)

    
    if not os.path.exists(arg2):	# create the directory if not exists
        os.makedirs(arg2)
    
    '''
    docs_list = []
    titles_list = []
    processed_docs = []
    tfs = []
    
    '''
    # writing all lists and dictionaries generated into a target directory specified in arguments or default
    with open(arg2+'/inverted_index_dict.json', 'w') as f:
        json.dump(inverted_index, f)
    print('Created successfully: inverted_index_dict.json')
    
    with open(arg2+'/freq_list.json', 'w') as f:
        json.dump(tfs, f)
    print('Created successfully: freq_list.json')

    with open(arg2+'/champ_list.json', 'w') as f:
        json.dump(championLists, f)
    print('Created successfully: champ_list.json')
    
    with open(arg2+'/titles_list.json', 'w') as f:
        json.dump(titles_list, f)
    print('Created successfully: titles_list.json \nCompleted!')
    
    ## this is the hardcoded version
    
    dict = {}
    dict['me'] =  ['me', 'me-and', 'myself']
    dict['enlighten'] =  ['enlighten', 'expound', 'inspire']
    dict['viral'] = ['viral', 'virus', 'infection'] 
    dict['foreigner'] = ['foreigner', 'foreigners', 'nationality']
    dict['main'] = ['main', 'main-sequence', "'main"]
    dict['poverty'] = ['poverty', 'poverty…by','inequality' ]
    dict['of'] = ['of', "'of", 'the']
    dict['cause'] = ['cause', 'causes', 'causing']
    
    with open(arg2+ '/relatedWords.pickle', 'wb') as f:
        pickle.dump(dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    #with open('relatedWords.pickle', 'rb') as f:
    #    d = pickle.load(f)

    
    


if __name__ == "__main__":
    try:
        arg1 = sys.argv[1]
        
    except:
        arg1 = './data/wiki_00'
	    
    try:
        arg2 = sys.argv[2]
    except:
        arg2 = './genFiles'
    
    print('Input data file: ',arg1,'\nOutput directory: ',arg2)
    main(arg1, arg2)