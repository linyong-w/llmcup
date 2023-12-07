from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from flask_login import LoginManager

from flask_login import login_required, current_user


from flask_login import UserMixin

from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from get_response import ask_local_question, ask_openai_question

import uuid
import time
import json
import os
import logging
import requests
import openai
import pandas as pd
from flask import Flask, Response, request, jsonify, send_from_directory
from dotenv import load_dotenv
from utils import auth_required

load_dotenv()


# init SQLAlchemy so we can use it later in our models
db = SQLAlchemy()

application = Flask(__name__, static_folder="static")

application.config['SECRET_KEY'] = 'secret-key-goes-here'
application.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
db.init_app(application)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))

with application.app_context():
    db.create_all()

login_manager = LoginManager()
# login_manager.login_view = 'auth.login'
login_manager.login_view = 'login'
login_manager.init_app(application)
##########MODELS

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

#MAIN
@application.route('/')
def index():
    return render_template('login.html')

# Static Files
@application.route("/profile")
@login_required
def profile():
    return application.send_static_file("index.html")

@application.route("/favicon.ico")
@login_required
def favicon():
    return application.send_static_file('favicon.ico')

@application.route("/assets/<path:path>")
@login_required
def assets(path):
    return send_from_directory("static/assets", path)
# ACS Integration Settings
AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
AZURE_SEARCH_USE_SEMANTIC_SEARCH = os.environ.get("AZURE_SEARCH_USE_SEMANTIC_SEARCH", "false")
AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG = os.environ.get("AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG", "default")
AZURE_SEARCH_TOP_K = os.environ.get("AZURE_SEARCH_TOP_K", 5)
AZURE_SEARCH_ENABLE_IN_DOMAIN = os.environ.get("AZURE_SEARCH_ENABLE_IN_DOMAIN", "true")
AZURE_SEARCH_CONTENT_COLUMNS = os.environ.get("AZURE_SEARCH_CONTENT_COLUMNS")
AZURE_SEARCH_FILENAME_COLUMN = os.environ.get("AZURE_SEARCH_FILENAME_COLUMN")
AZURE_SEARCH_TITLE_COLUMN = os.environ.get("AZURE_SEARCH_TITLE_COLUMN")
AZURE_SEARCH_URL_COLUMN = os.environ.get("AZURE_SEARCH_URL_COLUMN")

# AOAI Integration Settings
AZURE_OPENAI_RESOURCE = os.environ.get("AZURE_OPENAI_RESOURCE")
AZURE_OPENAI_MODEL = os.environ.get("AZURE_OPENAI_MODEL")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE", 0)
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P", 1.0)
AZURE_OPENAI_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS", 1000)
AZURE_OPENAI_STOP_SEQUENCE = os.environ.get("AZURE_OPENAI_STOP_SEQUENCE")
AZURE_OPENAI_SYSTEM_MESSAGE = os.environ.get("AZURE_OPENAI_SYSTEM_MESSAGE", "You are an AI assistant that helps people find information.")
AZURE_OPENAI_PREVIEW_API_VERSION = os.environ.get("AZURE_OPENAI_PREVIEW_API_VERSION", "2023-06-01-preview")
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM", "true")
AZURE_OPENAI_MODEL_NAME = os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-35-turbo") # Name of the model, e.g. 'gpt-35-turbo' or 'gpt-4'

SHOULD_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False

# Define a global variable to track conversation state
conversation_state = {
    'collecting_details': True,
    'user_details': {}
}

def start_collecting_details():
    global conversation_state
    conversation_state['collecting_details'] = True

def stop_collecting_details():
    global conversation_state
    conversation_state['collecting_details'] = False

def reset_user_details():
    global conversation_state
    conversation_state['user_details'] = {}

def extract_user_content(interactions):
    user_contents = [interaction['content'] for interaction in interactions if interaction['role'] == 'user']
    return user_contents

def collect_user_details(request_messages):
    global conversation_state

    if 'username' not in conversation_state['user_details']:
        conversation_state['user_details']['username'] = request_messages[0].get('content')
        # next_response = ask_openai_question(f"Paraphrase the follow 'Hi {conversation_state['user_details']['username']}, What is your working position?'")
        return f"Hi {conversation_state['user_details']['username']}, what is your current role?"

    elif 'working_position' not in conversation_state['user_details']:
        conversation_state['user_details']['working_position'] = request_messages[3].get('content')
        # next_response = ask_openai_question(f"Paraphrase the follow 'Where are you going to work at?'")
        return 'Thank you! Where is the location of the job?'

    elif 'working_location' not in conversation_state['user_details']:
        conversation_state['user_details']['working_location'] = request_messages[6].get('content')
        # next_response = ask_openai_question(f"Paraphrase the follow 'What is the purpose of this visit?'")
        return 'And the purpose of your visit?'

    elif 'working_purpose' not in conversation_state['user_details']:
        conversation_state['user_details']['working_purpose'] = request_messages[9].get('content')
        username = conversation_state['user_details']['username']
        working_position = conversation_state['user_details']['working_position']
        print(working_position)
        working_location = conversation_state['user_details']['working_location']
        working_purpose = conversation_state['user_details']['working_purpose']
        stop_collecting_details()

        initial_prompt = f"""You are a smart chatbot that help in providing professional advice for working before entering a workspace. Advise the below situation:
        {username} is a {working_position} that is going to {working_location} for {working_purpose}.
        Based on the information above, give 10 specific safety measurements that he should be aware of based on his working purpose. Start the message with Hi {username}!"""
        
        print(f"This is the initial prompt: {initial_prompt}")

        predictions = ask_local_question(initial_prompt)
        # body, headers = prepare_body_headers_with_data(initial_prompt)
        # #Make request to Databricks
        # r = requests.post(headers=headers, url=url, data=body, verify=False)
        # r = r.json()

        # predictions = r.get('predictions')
        # predictions = predictions[0]
        # request_messages.json["messages"] = []
        # request_messages.clear()
        # print(f"This is the predictions \n {predictions}")

        return predictions

    else:
        return "Something went wrong. Please try again."

def is_chat_model():
    if 'gpt-4' in AZURE_OPENAI_MODEL_NAME.lower() or AZURE_OPENAI_MODEL_NAME.lower() in ['gpt-35-turbo-4k', 'gpt-35-turbo-16k']:
        return True
    return False

def should_use_data():
    if AZURE_SEARCH_SERVICE and AZURE_SEARCH_INDEX and AZURE_SEARCH_KEY:
        return True
    return False

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def prepare_body_headers_with_data(request):

    request_body = pd.DataFrame({
        'prompt': [f"{request}"],
        'temperature': [0.5],
        'max_new_tokens': [500]
    })

    ds_dict = {'dataframe_split': request_body.to_dict(orient='split')} if isinstance(request_body, pd.DataFrame) else create_tf_serving_json(request_body)
    body = json.dumps(ds_dict, allow_nan=True)
    # response = requests.request(method='POST', headers=headers, url=url, data=data_body, verify=False)
    headers = {'Authorization': f'Bearer {os.getenv("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
    
   
    return body, headers

def format_prediction(prediction_text):
    # Replace this with your logic to extract citations and intent from the prediction_text
    citations = []
    intent = []

    # Generate a unique ID
    unique_id = str(uuid.uuid4())

    # Get the current timestamp
    timestamp = int(time.time())

    response_json = {
        "id": unique_id,
        "model": "gpt-35-turbo",
        "created": timestamp,
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "messages": [
                    {
                        "index": 0,
                        "role": "tool",
                        "content": "{\"citations\": [], \"intent\": \"[]\"}",
                        "end_turn": False
                    },
                    {
                        "index": 1,
                        "role": "assistant",
                        "content": prediction_text,
                        "end_turn": True
                    }
                ]
            }
        ]
    }


    return response_json

def conversation_with_data(request):
    request_messages = request.json["messages"]
    user_messages = [msg['content'] for msg in reversed(request_messages) if msg.get('role') == 'user']

    # Get the last 'content' from the filtered messages
    last_user_content = user_messages[0] if user_messages else None

    prompt_message = f"This is the response of the user. {last_user_content} \n\n If the answer is too long, try to put in point form."

    if 'working_purpose' not in conversation_state['user_details']:
        formatted_response = format_prediction(collect_user_details(request_messages))
        return jsonify(formatted_response)

    else:
        predictions = ask_local_question(prompt_message)
        formatted_response = format_prediction(predictions)
            
        return jsonify(formatted_response)

# Clear user details when clearing the chat
@application.route("/conversation", methods=["GET", "POST"])
@login_required
def conversation():
    try:
        action = request.args.get('action', '')
        if action == 'clear':
            # Clear user details if action is 'clear'
            reset_user_details()

        return conversation_with_data(request)

    except Exception as e:
        logging.exception("Exception in /conversation")
        return jsonify({"error": str(e)}), 500


##AUTH
@application.route('/login')
def login():
    return render_template('login.html')

@application.route('/login', methods=['POST'])
def login_post():
    # login code goes here
    email = request.form.get('email')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False

    user = User.query.filter_by(email=email).first()

    # check if the user actually exists
    # take the user-supplied password, hash it, and compare it to the hashed password in the database
    if not user or not check_password_hash(user.password, password):
        flash('Please check your login details and try again.')
        return redirect(url_for('login')) # if the user doesn't exist or password is wrong, reload the page

    # if the above check passes, then we know the user has the right credentials
    login_user(user,remember=remember)
    return redirect(url_for('profile'))

if __name__ == "__main__":
    application.run(port=3000)