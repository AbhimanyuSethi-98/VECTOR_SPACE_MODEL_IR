# VECTOR_SPACE_MODEL_IR
The objective of this project is to build a vector space-based information retrieval system, and run user-input queries on it with certain improvements implemented on top.
## Directory Structure
```bash
.
├── indexing.py
├── test_queries.py
├── requirements.txt
├── output.txt
├── README.md
├── Data
│   ├── wiki_00
├── genFiles
│   ├── inverted_index_dict.json
│   ├── freq_list.json
│   ├── champ_list.json
│   ├── titles_list.json
```
## Instruction to run the code

Optionally, create a virtual environment on your system and open it.
To run the application, first clone the repository by typing the command in git bash.
```
git clone https://github.com/AbhimanyuSethi-98/VECTOR_SPACE_MODEL_IR.git
```

Alternatively, you can download the code as .zip and extract the files.

To install the requirements, run the following command:
```
pip install -r requirements.txt
```
NOTE: One of the additional (3rd) improvements we implemented was using FastText to generate similar query terms to augment input query. Running this requires an ~8 GB download when running for the first time.
Install FastText using the following line of code.
```
git clone https://github.com/facebookresearch/fastText.git

cd fastText

sudo pip install
```

First run the indexing.py file to generate the required JSON files for building a vector space model for information retrieval, as follows:
```
python indexing.py <Optional argument 1> <Optional argument 2>
```

Argument 1 contains the relative path of corpus file (input data). Default : ./data/wiki_00

Argument 2 contains the name of the directory that will be created (if doesn't exist already) to store the JSON files containing the:

1) Inverted Index 

2) Frequency List 

3) Titles' List, and
4) Champion Lists, describing the corpus. Default : ./genFiles

To test your queries on the built IR System, run the test_queries.py file as follows:
```
python  test_queries.py
```

following which you will be asked to enter your query, the name of the directory containing the generated indexes and lists, and the corresponding option for which method of retrieval you want to run on your query, like so:

1. Normal tf-idf retrieval
2. Champion Lists
3. BM25 improvement
4. FastText improvement
5. BM25 + FastText
6. All 5 above

## Team Members
1. [Abhimanyu Sethi](https://github.com/AbhimanyuSethi-98)
2. [Ayush Agrawal](https://github.com/ayush1801)
3. [Bhavya Gera](https://github.com/bhavyagera10)
4. [Dhruv Patel](https://github.com/dppatel99)
5. [Pratyush Goel](https://github.com/GoelPratyush)
