from dotenv import load_dotenv
import os
import openai
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import os
import openai
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

#load environment variables
load_dotenv()

OPENAI_API_KEY = 'db0ceebd87844220b070af75fcd744a3'
OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")

OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = os.getenv("OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME")
OPENAI_ADA_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_ADA_EMBEDDING_MODEL_NAME")

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

openai.api_type = "azure"
openai.api_base = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = AZURE_OPENAI_KEY

chat_history=[]

def append_message(existing_message, new_message):
    """
    Appends a new message to an existing message.

    Parameters:
    - existing_message (str): The original message.
    - new_message (str): The message to append.

    Returns:
    - str: The combined message.
    """
    combined_message = existing_message + " " + new_message
    return combined_message

# Example usage:
# original_message = "Hello"
# new_message = "world!"
# result = append_message(original_message, new_message)
# print(result)



def ask_openai_question(question):
    messages = [{"role": "user", "content": question}]

    response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_MODEL,
            messages=messages,
            temperature=float(AZURE_OPENAI_TEMPERATURE),
            max_tokens=int(AZURE_OPENAI_MAX_TOKENS),
            top_p=float(AZURE_OPENAI_TOP_P),
            stop=AZURE_OPENAI_STOP_SEQUENCE.split("|") if AZURE_OPENAI_STOP_SEQUENCE else None,
            stream=False
        )
    
    result = response.choices[0].message.content
    return result


def ask_question(qa, question):
    result = qa({"query": question})
    print("Question:", question)
    print("Answer:", result["result"])

def ask_local_question(question):
    global chat_history
    llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
                      model_name=OPENAI_MODEL_NAME,
                      openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                      openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                      openai_api_key=OPENAI_API_KEY,
                      openai_api_type="azure")
    
    embeddings=OpenAIEmbeddings(deployment=OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
                                model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
                                openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                                openai_api_type="azure",
                                chunk_size=1)


    # Initialize gpt-35-turbo and our embedding model
    #load the faiss vector store we saved into memory
    vectorStore = FAISS.load_local("./dbs/documentation/faiss_index", embeddings)

    #use the faiss vector store we saved to search the local document
    retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k":5})

    INITIAL_PROMPT = PromptTemplate.from_template(
        """
        You are a smart chatbot that help in providing professional advice for working before entering a workspace. Advise the below situation:

        {name} is a {working_position} that is going to {location} for {work_purpose}.
        Based on the information above, give 10 safety measurements that he should be aware of before entering the workspace.
        Chat History:
        {chat_history}""")

    INITIAL_QA = ConversationalRetrievalChain.from_llm(llm=llm,
                                            retriever=retriever,
                                            condense_question_prompt=INITIAL_PROMPT,
                                            return_source_documents=True,
                                            verbose=False)
    
    QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
                                                    Chat History:
                                                    {chat_history}
                                                    Follow Up Input: {question}""")
    
    qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                        retriever=retriever,
                                        condense_question_prompt=QUESTION_PROMPT,
                                        return_source_documents=True,
                                        verbose=False)

    result = qa({"question": question, "chat_history": chat_history})

    # chat_history.append((question, result["answer"]))
    chat_history = [(question, result["answer"])]
    # chat_history = chat_history + "\n" + chat_history
    
    print(f"This is the chat history:{chat_history}")
    return result["answer"]

def ask_question_with_context(qa, question, chat_history):
    query = "what is Azure OpenAI Service?"
    result = qa({"question": question, "chat_history": chat_history})
    print("PASH:", result["answer"])
    print("\n")
    
    chat_history = [(question, result["answer"])]
    return chat_history


if __name__ == "__main__":
    print(ask_local_question("What is hydrogen sulfide?"))
    # print("Welcome to PASH - PASHionate about your safety at workspace.")
    # # Configure OpenAI API
    # openai.api_type = "azure"
    # openai.api_base = os.getenv('OPENAI_API_BASE')
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    # openai.api_version = os.getenv('OPENAI_API_VERSION')
    # llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
    #                   model_name=OPENAI_MODEL_NAME,
    #                   openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
    #                   openai_api_version=OPENAI_DEPLOYMENT_VERSION,
    #                   openai_api_key=OPENAI_API_KEY,
    #                   openai_api_type="azure")
    
    # embeddings=OpenAIEmbeddings(deployment=OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
    #                             model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
    #                             openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
    #                             openai_api_type="azure",
    #                             chunk_size=1)


    # # Initialize gpt-35-turbo and our embedding model
    # #load the faiss vector store we saved into memory
    # vectorStore = FAISS.load_local("./dbs/documentation/faiss_index", embeddings)

    # #use the faiss vector store we saved to search the local document
    # retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k":2})

    # INITIAL_PROMPT = PromptTemplate.from_template(
    #     """
    #     You are a smart chatbot that help in providing professional advice for working before entering a workspace. Advise the below situation:

    #     {name} is a {working_position} that is going to {location} for {work_purpose}.
    #     Based on the information above, give 10 safety measurements that he should be aware of before entering the workspace.
    #     Chat History:
    #     {chat_history}""")

    # INITIAL_QA = ConversationalRetrievalChain.from_llm(llm=llm,
    #                                         retriever=retriever,
    #                                         condense_question_prompt=INITIAL_PROMPT,
    #                                         return_source_documents=True,
    #                                         verbose=False)
    
    # QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    #                                                 Chat History:
    #                                                 {chat_history}
    #                                                 Follow Up Input: {question}
    #                                                 Standalone question:""")
    
    # # QUESTION_PROMPT = PromptTemplate.from_template(
    # #     """
    # #     You are a smart chatbot that help in providing professional advice for working before entering a workspace. Advise the below situation:

    # #     Chat History:
    # #     {chat_history}

    # #     Follow Up Input: {question}
    # #     """)

    # qa = ConversationalRetrievalChain.from_llm(llm=llm,
    #                                         retriever=retriever,
    #                                         condense_question_prompt=QUESTION_PROMPT,
    #                                         return_source_documents=True,
    #                                         verbose=False)                           
    
    # # while True:


    # chat_history = []
    # name = input('Your name: ')
    # working_position = input('Working Position: ')
    # location = input('Location: ')
    # work_purpose = input('Work Purpose: ')
    # # name = "Steven"
    # # working_position = "Mechanical Technician"
    # # location = "Rotating Machine is Gas Pipeline"
    # # work_purpose = "Perform maintenance on the rotary part of the generator"

    # first_response = f"""You are a smart chatbot that help in providing professional advice for working before entering a workspace. Advise the below situation:
    #                     {name} is a {working_position} that is going to {location} for {work_purpose}.
    #                     Based on the information above, give 10 safety measurements that he should be aware of before entering the workspace.
    #                     Start your sentence with "Hi, {name}!"""
    
    # chat_history = ask_question_with_context(qa, first_response, chat_history)
    # # query = input(f'{name}: ')
    # # if query == 'q':
    # #     break

    # # chat_history = ask_question_with_context(qa, query, chat_history)

    # while True:
        
    #     query = input(f'{name}: ')
    #     if query == 'q':
    #         break

    #     # user_input = f"You are a smart chatbot that help in providing professional advice for working before entering a workspace. Advise the below situation: {query}"
        
    #     chat_history = ask_question_with_context(qa, query, chat_history)
