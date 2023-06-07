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
