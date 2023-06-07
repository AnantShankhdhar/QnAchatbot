# QnAchatbot

## Setting up the Repository
1. Clone the repository
```sh
git clone git@github.com:AnantShankhdhar/QnAchatbot.git
```
2. Install the required dependencies in your python environment
```sh
pip install requirements.txt
```

## File structure
1. truefoundry_qna.ipynb :- Contains the notebook showcasing the work flow of the solution
2. qna_model.py :- Contains the my_model class which is the pipeline for the retrieval method
3. app.py :- The main service file containing FastAPI 
4. deploy.py :- Truefoundry deployment file
5. make_segments.py :- Code for preprocessing documents to generate segments which can answer the questions
6. output.txt :- contains the text output for the questions given for the task

## Solution Approach
We implement a three step solution for the problem:-
### 1. Breaking the document collection into smaller segments :- 
This has to be done in such a way that every segment contains a group of sentences that have similar context.
In order to do this we first break the documents into sentences seperated by a "." token. Then we create embeddings for every sentences using all-mpnet-base-v2 model. We iterate over all sentences and calculate the similarity between the current and previous sentences. Whenever we find a relative minima (i.e similarity is high before the point as well as after the point) we identify that as a split point and split sentences accordingly. A bigger insight can be by reading this [blog](https://medium.com/@npolovinkin/how-to-chunk-text-into-paragraphs-using-python-8ae66be38ea6)

### 2. Identifying Potential Segments that can answer the questions:-
Our database is too large for us to use the entire document for searching the answer to the questions. To make our search faster we need to narrow it down to a few segments that may possibly contain the answer. To do this, we use DPR for searching the best segments that can answer the question. We use the pretrained dpr-ctx_encoder-single-nq-base for context and dpr-question_encoder-single-nq-base for the questions. We generate indexes using FAISS and search for the top 5 segments that are best for answering our question. We concatenate all the segments and use this as the context for answering the final question.

### 3. Retreival Augmented Generation for question answering:-
Now that we have our question and context we finally perform retrieval augmented generation to get the answer for the question. The model used is gpt4all-j-v.1.3 which is a small model that can be run on colab and kaggle. We can use better models too but under the constraints this was the best possible choice. We use prompt engineering which passes the context and question from the model and asks it to give the answer from the context
