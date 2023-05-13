import os
import requests
from requests_oauthlib import OAuth2Session
import adal

# Link to Azure:
# https://portal.azure.com/?Microsoft_Azure_Education_correlationId=fbc78e78e9ee40b7a0164746ed8a6ecd#view/Microsoft_AAD_RegisteredApps/ApplicationMenuBlade/~/Overview/appId/0e4550e7-185c-4564-a181-35ffcb44d603/isMSAApp~/true

# Azure AD application details
tenant_id = "f8cdef31-a31e-4b4a-93e4-5f571e91255a"
client_id = "0e4550e7-185c-4564-a181-35ffcb44d603"
client_secret = 'fd15b289-9b8e-40f1-ac83-1577666d3c21'
scopes = 'https://graph.microsoft.com/.default'

token_endpoint = f'https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token'

client = OAuth2Session(client_id, client_secret, scope=scopes)
# client = OAuth2Session(client_id, redirect_uri='http://localhost/auth-response', scope=scopes)
token = client.fetch_token(token_endpoint, authorization_response='http://localhost/auth-response')

# Retrieve the user's messages
endpoint = 'https://graph.microsoft.com/v1.0/me/messages'
response = client.get(endpoint)

# Process the messages
if response.status_code == requests.codes.ok:
    messages = response.json()['value']
    for message in messages:
        print(f"Subject: {message['subject']}, From: {message['from']['emailAddress']['name']}, To: {message['toRecipients'][0]['emailAddress']['name']}")
else:
    print(f"Failed to retrieve messages. Error code: {response.status_code}, Message: {response.text}")
