import streamlit as st
import pandas as pd
import time
import os
import gcsfs

# https://medium.com/@faizififita1/how-to-deploy-your-streamlit-web-app-to-google-cloud-run-ba776487c5fe
# gcloud builds submit --tag gcr.io/msca310019-capstone-49b3/streamlit-app

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials/msca310019-capstone-49b3-9b57ab2e649d.json"

# Create a GCS filesystem instance
fs = gcsfs.GCSFileSystem(project='msca310019-capstone-49b3')

# Define the path to your CSV file in the GCS bucket
file_path = 'user-scripts-msca310019-capstone-49b3/data/data_message_reply_pairs_cleaned.csv'


# Read the CSV file from Google Cloud Storage
with fs.open(file_path, 'rb') as file:
    # Use Pandas to read the CSV content into a DataFrame
    data = pd.read_csv(file)
    
# data = pd.read_csv('gs://user-scripts-msca310019-capstone-49b3/data/20231019_Dataset_Users_Greater_Than_50.csv') 

# data = pd.read_csv('/Users/scottsmacbook/Hedwig/Hedwig/00_Data/data_data_message_reply_pairs_cleaned.csv')

# Define a function to generate a random email and its sender ID
def generate_random_email():
    st.session_state.random_email = data.sample(n=1)

# Define a function to type out a string letter by letter
def type_string(text):
    t = st.empty()
    
    for i in range(len(text) + 1):
        t.markdown("## %s" % text[0:i])
        time.sleep(0.005)

# Set the title image path
title_image_path = 'Hedwig Logo.jpeg'  # Replace with the actual path

# Display the title and image side by side
st.title("Hedwig.AI")

st.image(title_image_path, use_column_width=False, width=100) 
# Streamlit app layout
st.title("Hedwig.AI")

# Display user information
st.subheader("Incoming Email")
user_button = st.button("Get Random Email")

if user_button:
    generate_random_email()

if 'random_email' not in st.session_state:
    generate_random_email()
   
random_user = st.session_state.random_email
st.write(f"Sender ID: {str(list(random_user['sender'])[0])}")
st.write(f"Email: {str(list(random_user['message'])[0])}")

# Input field for email response
# replier_id = st.text_input("Enter your user :")

# Button to generate the response
if st.button("Generate Response") and st.session_state.random_email is not None:
    st.write(f"Replier ID: {str(list(random_user['reply_sender'])[0])}")
    st.write("Response:")
    random_user = st.session_state.random_email
    type_string(str(list(random_user['reply_message'])[0]))
    