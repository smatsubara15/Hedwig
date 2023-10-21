import streamlit as st

data = pd.read_csv('gs://user-scripts-msca310019-capstone-49b3/data/20231019_Dataset_Users_Greater_Than_50.csv') 

# Define a function to generate a random email and its sender ID
def generate_random_email():
    return data.sample(n=1)

# Define a function to type out a string letter by letter
def type_string(text):
    for char in text:
        st.text(char)
        time.sleep(0.1)  # Adjust the sleep duration for the typing speed

# Streamlit app layout
st.title("Hedwig App")

# Display user information
st.subheader("Incoming Email")
user_button = st.button("Get Random Email")
if user_button:
    random_user = generate_random_email()
    st.write(f"Sender ID: {str(list(random_user['sender'])[0])}")
    st.write(f"Email: {str(list(random_user['message'])[0])}")

                        
# Input field for email response
# replier_id = st.text_input("Enter your user :")

# Button to generate the response
if st.button("Generate Response"):
    st.write("Response:")
    type_string(str(list(random_user['reply_message'])[0]))