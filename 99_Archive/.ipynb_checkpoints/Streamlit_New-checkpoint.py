import streamlit as st
import pandas as pd
import time
import gcsfs
import asyncio
import os
import math
import chromadb
from chromadb.utils import embedding_functions

import langchain
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import SimpleMemory

#CHROMA
import chromadb
from chromadb.utils import embedding_functions
from langchain.vectorstores import Chroma

import nltk
from nltk.tokenize import word_tokenize
import string

import ast  # Import the ast module for literal evaluation

def get_threads(sender,
                replier,
                subject,
                df,
                num_emails_past):
    relevant_df = df[((df.sender==sender) & (df.replier==replier) & (df.subject == subject))]
    
    if (len(relevant_df)==0):
        relevant_df = df[((df.sender==replier) & (df.replier==sender) & (df.subject == subject))]
        
    if (len(relevant_df)==0):
        return
    
    relevant_df['sender_date'] = pd.to_datetime(relevant_df['sender_date'])
    relevant_df['replier_date'] = pd.to_datetime(relevant_df['replier_date'])
    
    messages = pd.concat([relevant_df['message'], relevant_df['reply_message']]).reset_index(drop=True)
    dates = pd.concat([relevant_df['sender_date'], relevant_df['replier_date']]).reset_index(drop=True)
    name = pd.concat([relevant_df['sender'], relevant_df['replier']]).reset_index(drop=True)
    
    thread_df = pd.DataFrame({'message': messages,'date': dates,'name':name})
    thread_df = thread_df.sort_values(by='date',ascending=False)
    
    ordered_names = list(thread_df.name)
    ordered_messages = list(thread_df.message)
    
    thread_string = ''
    for i in range(num_emails_past):
        thread_string = thread_string + f"{ordered_names[i]} Email {math.ceil((i+1)/2)}: {ordered_messages[i]} \n \n"
        
    # print(thread_string)
    return thread_string

def get_replier_sender_past_emails(sender,
                                   replier,
                                   df,
                                   num_past_emails):
    
    relevant_df = df[(((df.sender==sender) & (df.replier==replier)) | ((df.sender==replier) & (df.replier==sender)))]
    
    relevant_df['sender_date'] = pd.to_datetime(relevant_df['sender_date'])
    relevant_df['replier_date'] = pd.to_datetime(relevant_df['replier_date'])
    
    messages = pd.concat([relevant_df['message'], relevant_df['reply_message']]).reset_index(drop=True)
    dates = pd.concat([relevant_df['sender_date'], relevant_df['replier_date']]).reset_index(drop=True)
    name = pd.concat([relevant_df['sender'], relevant_df['replier']]).reset_index(drop=True)
    
    relationship_df = pd.DataFrame({'message': messages,'date': dates,'name':name})
    relationship_df = relationship_df.sort_values(by='date',ascending=False)
    
    relationship_df = relationship_df[relationship_df.name==replier]
    
    ordered_names = list(relationship_df.name)
    ordered_messages = list(relationship_df.message)
    
    past_emails_string = ''
    for i in range(num_past_emails):
        past_emails_string = past_emails_string + f"Replier Email {i+1}: {ordered_messages[i]} \n \n"
        
    return past_emails_string
    
async def zeroth_LLM_endpoint(vector_store, sender_email, num_emails):
    found_rel_emails = await vector_store.amax_marginal_relevance_search(sender_email, k=num_emails, fetch_k=num_emails)
    list_rel_emails=[]
    for i, doc in enumerate(found_rel_emails):
        list_rel_emails.append(doc.page_content)
    return list_rel_emails

def first_LLM_endpoint(llm_endpoint):
    template_string_globalcontext="""You are the person recieving this email {sender_email},
    Write a reply to the email as the person who recieved it, 
    deriving context and writing style and email length from previous relevant emails from the person given: {relevant_emails}, 
    Make sure to use salutation and signature style similar to the revelant emails above.
    You are replying to {sender_name} on behalf of {replier_name}."""

    # Setting up LangChain
    prompt_template_globalcontext = ChatPromptTemplate.from_template(template=template_string_globalcontext)    
    llm_chain_globalcontext=LLMChain(llm=llm_endpoint, prompt=prompt_template_globalcontext, output_key='Global_Context_Email')
    return llm_chain_globalcontext

def remove_punctuation(text):
    # Define a translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    # Use the translate method of the string class to remove any punctuation
    return text.translate(translator)

def get_tokens(replier_id,
               df):
    user_df = df[(df.replier==replier_id)]
    # Tokenize each entry
    # user_df['cleaned_message'] = user_df['reply_message'].apply(lambda x: text.translate(str.maketrans('', '', string.punctuation)))
    user_df['tokens'] = user_df['reply_message'].apply(lambda x: word_tokenize(remove_punctuation(x)))
    user_df['token_count'] = user_df['tokens'].apply(lambda x: len(x))
    # Calculate the average number of tokens
    average_tokens = user_df['token_count'].median()
    return average_tokens

    
# https://medium.com/@faizififita1/how-to-deploy-your-streamlit-web-app-to-google-cloud-run-ba776487c5fe
# gcloud builds submit --tag gcr.io/msca310019-capstone-49b3/streamlit-app

st.set_page_config(layout="wide")

df_messages=pd.read_csv('human_validation_with_relevent_date.csv', parse_dates=['sender_date','replier_date'])
df_messages.dropna(subset=['sender'], axis=0, inplace=True)
df_messages.rename(columns={'Sender_Receiver_Emails':'Replier_Emails_Sender', 'Sender_Emails_All':'Replier_Emails_All'}, inplace=True)

# df_messages=pd.read_csv('Hedwig/07_HumanValidation/20231104_human_validation_dataset.csv')
df_messages['Replier_Emails_Sender'] = df_messages['Replier_Emails_Sender'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
df_messages['num_emails_toSender'] = df_messages['Replier_Emails_Sender'].apply(lambda x: len(x) if isinstance(x, list) else np.nan) + 1
df_messages['Replier_Emails_All'] = df_messages['Replier_Emails_All'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
df_messages['num_emails_all'] = df_messages['Replier_Emails_All'].apply(lambda x: len(x) if isinstance(x, list) else np.nan) + 1
df_messages['sender_replier_thread'] = df_messages['sender'].str.cat(df_messages['replier'], sep='-')

# sender_id='Kshitij'
# replier_id='Scott'

# Define a function to generate a random email and its sender ID
def generate_random_email():
    st.session_state.random_email = df_messages.sample(n=1)

# # Define a function to type out a string letter by letter
def type_string(text):
    t = st.empty()
    
    for i in range(len(text) + 1):
        t.markdown("## %s" % text[0:i])
        time.sleep(0.005)

# # Set the title image path
title_image_path = 'Hedwig Logo.jpeg'  # Replace with the actual path

# # Display the title and image side by side
# st.title("Hedwig.AI")

col1, mid, col2 = st.columns([1,1,20])

with col1:
    st.image('Hedwig Logo.jpeg', width=60)
with col2:
    st.title("Hedwig.AI")

# Use Streamlit's 'columns' layout to display buttons side by side
col1, mid,col2 = st.columns(3)

with col1:
    st.subheader("Incoming Email")

with col2:
    st.subheader("Generated Email")
    replier_id = st.text_area("Replier:")


with mid: 
    st.subheader("Retrieval and Ranking")
    

# Button to get a random email
# if col1.button("Get Random Email Reply Pair"):
#     generate_random_email()

if 'random_email' not in st.session_state:
    generate_random_email()
   
random_email = st.session_state.random_email

with col1:
    # st.write(f"Sender Name: {sender_id}")
    # st.write(f"Subject: {sender_id}")
    
    sender_id = st.text_area("Sender:", height=10, max_chars=50)
    subject_email = st.text_area("Subject:")

    # Create a larger text area for user input (e.g., 10 rows)
    user_input = st.text_area("Enter Email:", height=500)
    
    # st.write(f"Email: {user_input}")

# Input field for email response
# replier_id = st.text_input("Enter your user :")

# Button to generate the response
if col2.button("Generate Response") and st.session_state.random_email is not None:
    # st.write("Response:")
    chroma_client = chromadb.PersistentClient(path="vectorstores")

    sender_name=sender_id
    replier_name=replier_id

    sender_email = user_input

    num_emails = 10 #FOR RETRIEVEL + RANKING
    vector_db_client = chroma_client 

    openai_api_key = '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'
    os.environ['OPENAI_API_KEY'] = openai_api_key

    base_dataset=df_messages
    vector_db_client=chroma_client # FOR RANKING VECTOR DATABASE
    num_emails = 5
    
    api_key=openai_api_key
    llm_model='gpt-3.5-turbo-0301' # CAN CHANGE
    llm_endpoint=ChatOpenAI(temperature=0.1, model=llm_model, openai_api_key=openai_api_key) # CAN CHANGE

    ### ZEROTH LLM ENDPOINT
    user_vector_store = Chroma(client=vector_db_client, collection_name='user'+str(replier_id),embedding_function=OpenAIEmbeddings())
    # list_rel_emails = zeroth_LLM_endpoint(user_vector_store, sender_email, num_emails)

    list_rel_emails = asyncio.run(zeroth_LLM_endpoint(user_vector_store, sender_email, num_emails))

    ### FIRST LLM ENDPOINT
    template_string_globalcontext="""You are {replier_name}, receiving an email {sender_name},
    The email is: {sender_email},
    Write a reply to the email as the person who received it, 
    deriving context and writing style and email length from previous relevant emails from the person given: {relevant_emails}, 
    Make sure to use salutation and signature style similar to the revelant emails above.
"""

    prompt_template_globalcontext = ChatPromptTemplate.from_template(template=template_string_globalcontext)    
    llm_chain_globalcontext=LLMChain(llm=llm_endpoint, prompt=prompt_template_globalcontext, output_key='Global_Context_Email')

    ### SECOND LLM ENDPOINT
    past_threads=get_threads(sender=sender_id,
            replier=replier_id,
            subject=subject_email,
            df=base_dataset,
            num_emails_past=2*num_emails)

    template_string_thread="""Take this LLM generated email: {Global_Context_Email}. 
    This email might have some trailing emails, stored in the email thread here: {past_threads}.
    Rewrite the LLM Generated Email, by deprioritizing topics which are not present in the past email thread.
    Otherwise don't make major changes to the LLM generated email"""
    
    prompt_template_thread=ChatPromptTemplate.from_template(template=template_string_thread)
    llm_chain_thread=LLMChain(llm=llm_endpoint, prompt=prompt_template_thread, output_key='Local_Context_Email')

    ### THIRD LLM ENDPOINT
    past_emails=get_replier_sender_past_emails(sender=sender_id,
                               replier=replier_id,
                               df=base_dataset,
                               num_past_emails=num_emails)

    template_string_pairstyle="""Extract Email Writing Style in 3 words that best decribe the replier by analyzing these past emails between the sender and replier: {past_emails}"""
    
    prompt_template_pairstyle = ChatPromptTemplate.from_template(template=template_string_pairstyle)    
    llm_chain_pairstyle=LLMChain(llm=llm_endpoint, prompt=prompt_template_pairstyle, output_key='pair_style')

    ### FOURTH LLM ENDPOINT
    template_string_personalization="""Take this email :<{Local_Context_Email}>, update the email and create one single email which is {pair_style}. 
    Remember that these adjectives collectively describe your writing style,
    DO NOT add any more information, just tweak the style a little.
    Don't be dramatic, and the output should have approximately {avg_tokens} number of tokens"""
    
    prompt_template_personalization=ChatPromptTemplate.from_template(template=template_string_personalization)
    llm_chain_personalization=LLMChain(llm=llm_endpoint, prompt=prompt_template_personalization, output_key='Personalized_Email')

    ### Sequential LLM CHAIN
    super_chain = SequentialChain(memory=SimpleMemory(memories={"sender_name":sender_name,
                                                           "replier_name":replier_name,
                                                           "avg_tokens":get_tokens(replier_id=replier_name, df=df_messages)}),
                              chains=[llm_chain_globalcontext, llm_chain_thread,llm_chain_pairstyle, llm_chain_personalization],
                              input_variables=['relevant_emails','sender_email','past_threads','past_emails'],
                              output_variables=['Global_Context_Email','Local_Context_Email','pair_style','Personalized_Email']
                             )

    results = super_chain({"relevant_emails": list_rel_emails, 
                             "sender_email": sender_email,
                             "past_threads": past_threads,
                             "past_emails":past_emails})

    with mid: 
        # st.write("Previous Emails: ")
        # for email in previous_emails:
        #     st.write(email)
        st.write("Ranked Emails: ")
        for i in range(0,5):
            st.write((f"{i+1}. {list_rel_emails[i]}"))
        
    with col2: 
        st.write(f"Replier Name: {replier_id}")
        st.text_area("Generated Response: ", value=results['Personalized_Email'], key='response_area', height=500)
        # type_text_in_textarea(personalized_response)
        


    
    