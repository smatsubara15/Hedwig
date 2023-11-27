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
        t.markdown("%s" % text[0:i])
        time.sleep(0.01)

# # Set the title image path
title_image_path = 'new_logo.svg'  # Replace with the actual path

# Create a space column on the left, then the title, and then the image on the right
col_image,space, col_title = st.columns([1, 0.2, 0.2])

st.markdown(
    """
    <style>
    .grey_subheaders {
        font-size:20px !important;
        color:#808080 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .grey_text {
        font-size:15px !important;
        color:#808080 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .white_header {
        font-size:40px !important;
        color:white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .blue_header {
        font-size:40px !important;
        color:#0086f1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .white_text {
        font-size:20px !important;
        color:white !important;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Display the image in the image column
with col_image:
    st.image(title_image_path, width=250)  # Adjust width as needed
#     st.title("Hedwig.AI")

# Use Streamlit's 'columns' layout to display buttons side by side
col1,col2,vertical_line,outputs = st.columns([2.5,2.5,0.1,1.5])

if 'sender_id' not in st.session_state:
    st.session_state['sender_id'] = 'Kshitij'
    st.session_state['replier_id'] = 'Scott'
    st.session_state['subject'] = 'New Member Onboarding'

if 'button_pressed' not in st.session_state:
    st.session_state['button_pressed'] = False

if 'counter' not in st.session_state:
    st.session_state['counter'] = 0
    
with col1:
    st.markdown('<p class="blue_header">Incoming Email</p>', unsafe_allow_html=True)
    

# Button to get a random email
if col1.button("Get New Email Reply Pair"):
    st.session_state['counter'] += 1
    if(st.session_state['counter'] % 2 == 1):
        st.session_state['replier_id'] = 'Radhika'
    else:
        st.session_state['replier_id'] = 'Scott'

sender_id = st.session_state['sender_id']
replier_id = st.session_state['replier_id']
subject_email = st.session_state['subject']

with col2:
    st.markdown('<p class="blue_header">Generated Email</p>', unsafe_allow_html=True)
    

with vertical_line:
    st.markdown('<div style="border-left: 2px solid #808080; height: 1000px"></div>', unsafe_allow_html=True)


with outputs: 
    st.markdown(
        """
        <style>
        .big-font {
            font-size:25px !important;
            color:#808080 !important;
        }
        </style>
        <p class='big-font'>Email Generation Process</p>
        """, 
        unsafe_allow_html=True
    )

    
with col1:
    st.markdown(f'<p class="white_text">Sender Name: {sender_id}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="white_text">Subject: {subject_email}</p>', unsafe_allow_html=True)
        
    # Create a larger text area for user input (e.g., 10 rows)
    st.markdown(f'<p class="white_text">Enter Email:</p>', unsafe_allow_html=True)
    user_input = st.text_area("", height=500)
    

# Button to generate the response
if col2.button("Generate Response") and st.session_state.sender_id is not None:
    st.session_state['button_pressed'] = True
    
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
    with outputs: 
        st.markdown('<p class="grey_subheaders">1. Global Context</p>', unsafe_allow_html=True)

        st.markdown(
        """
        <ul class="grey_text">
            <li>Ranking Relevant Emails Written by the Replier</li>
            <li>Generating Initial Email Based on Relevant Emails</li>
        </ul>
        """,
        unsafe_allow_html=True
        )
        
        time.sleep(2)
        
        
    user_vector_store = Chroma(client=vector_db_client, collection_name='user'+str(replier_id),embedding_function=OpenAIEmbeddings())

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
    with outputs: 
        st.markdown('<p class="grey_subheaders">2. Local Context</p>', unsafe_allow_html=True)

        st.markdown(
        """
        <ul class="grey_text">
            <li> Revising Email using Relevant Context Found within Threads</li>
        </ul>
        """,
        unsafe_allow_html=True
        )
        
        time.sleep(3)
        
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
    with outputs: 
        st.markdown('<p class="grey_subheaders">3. Pair Style Personalization</p>', unsafe_allow_html=True)
        
        st.markdown(
        """
        <ul class="grey_text">
            <li>Extracting Writing Style from Emails Written Between Replier and Sender</li>
            <li>Personalizing Email</li>
        </ul>
        """,
        unsafe_allow_html=True
        )
        
        time.sleep(3)
    
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

    with outputs: 
        st.markdown('<p class="grey_subheaders">4. Hyper-Personalization</p>', unsafe_allow_html=True)

        st.markdown(
        """
        <ul class="grey_text">
            <li>Generating Final Email by Combing all Previous Steps</li>
        </ul>
        """,
        unsafe_allow_html=True
        )
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

    with outputs:         
        st.markdown('<p class="grey_subheaders">Ranked Emails:</p>', unsafe_allow_html=True)
        
        # Start the unordered list
        bullet_points = "<ol class='grey_text'>"

        # Loop through the items and add them as list items
        for i in range(0,5):
            bullet_points += f"<li>{list_rel_emails[i]}</li>"

        # Close the unordered list
        bullet_points += "</ol>"

        
        # Display the bullet points
        st.markdown(bullet_points, unsafe_allow_html=True)
    
        
with col2: 
    st.markdown(f'<p class="white_text">Replier Name: {replier_id}</p>', unsafe_allow_html=True)
    
    st.write('\n')
    st.write('\n')
    st.write('\n')
    
    st.markdown(f'<p class="white_text">Generated Response:</p>', unsafe_allow_html=True)
 
    if (st.session_state['button_pressed'] == True):
        type_string(results['Personalized_Email'])
        st.session_state['button_pressed'] = False

    

    
    

    
    