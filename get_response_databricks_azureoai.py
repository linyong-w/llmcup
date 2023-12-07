from dotenv import load_dotenv
import os
import openai
import pandas as pd
import numpy as np
import json
import requests
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

load_dotenv()
chat_history=[]

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://dbc-eb788f31-6c73.cloud.databricks.com/serving-endpoints/openai_03/invocations'
  headers = {'Authorization': f'Bearer {os.getenv("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json, verify=False)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

def ask_local_question(question):
    global chat_history

    badanswer_phrases = [ # phrases that indicate model produced non-answer
    "no information", "no context", "don't know", "no clear answer", "sorry", 
    "no answer", "no mention", "reminder", "context does not provide", "no helpful answer", 
    "given context", "no helpful", "no relevant", "no question", "not clear",
    "don't have enough information", " does not have the relevant information", "does not seem to be directly related"
    ]

    user_prompt = f"This is the user question: {question}. \n Answer the question above based and use the following chat history as a reference for previous conversation. \nChat History: {chat_history}."

    queries = pd.DataFrame({'question':[
    user_prompt
    ]})
    
    formatted_user_question = pd.DataFrame({'question':[
    question
    ]})

    result = score_model(queries)
    # Print or process the result as needed
    predictions = result.get('predictions')[0]["answer"]
    # chat_history.append((question, predictions))
    
    for phrase in badanswer_phrases:
        if phrase in predictions.lower():
          chat_history=[]
          result = score_model(formatted_user_question)
          predictions = result.get('predictions')[0]["answer"]
          print("\n BAD ANSWER DETECTED \n")
          # break
    
    chat_history = (f"User: {question}", f"Answer: {predictions}")

    print("Result:", predictions)
    print(f"Chat history: {chat_history}")
    return predictions

if __name__ == "__main__":
    while True:
       user_input = input("User:")
       ai_response = ask_local_question(user_input)
       print(f"AI: {ai_response} \n\n\n")
