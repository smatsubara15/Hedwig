import asyncio
import configparser
from graph import Graph
import pandas as pd

# Azure Application: https://portal.azure.com/?Microsoft_Azure_Education_correlationId=fbc78e78e9ee40b7a0164746ed8a6ecd#view/Microsoft_AAD_RegisteredApps/ApplicationMenuBlade/~/Overview/appId/0e4550e7-185c-4564-a181-35ffcb44d603/isMSAApp~/true
# code from: https://learn.microsoft.com/en-us/graph/tutorials/python?context=outlook%2Fcontext&tabs=aad&tutorial-step=5
async def main():
    print('Python Graph Tutorial\n')

    # Load settings
    config = configparser.ConfigParser()
    config.read(['config.cfg', 'config.dev.cfg'])
    azure_settings = config['azure']

    graph: Graph = Graph(azure_settings)

    await greet_user(graph)

    choice = -1

    while choice != 0:
        print('Please choose one of the following options:')
        print('0. Exit')
        print('1. Display access token')
        print('2. List my inbox')
        print('3. Send mail')
        print('4. Create Draft')

        try:
            choice = int(input())
        except ValueError:
            choice = -1

        if choice == 0:
            print('Goodbye...')
        elif choice == 1:
            await display_access_token(graph)
        elif choice == 2:
            await list_inbox(graph)
        elif choice == 3:
            await send_mail(graph)
        elif choice == 4:
            await create_draft(graph)
        else:
            print('Invalid choice!\n')

async def greet_user(graph: Graph):
    user = await graph.get_user()
    if user is not None:
        print('Hello,', user.display_name)
        # For Work/school accounts, email is in mail property
        # Personal accounts, email is in userPrincipalName
        print('Email:', user.mail or user.user_principal_name, '\n')

async def display_access_token(graph: Graph):
    token = await graph.get_user_token()
    print('User token:', token, '\n')

async def list_inbox(graph: Graph):
    messages = []
    message_page = await graph.get_inbox()
    if message_page is not None and message_page.value is not None:
        # Output each message's details
        for message in message_page.value:

            # print('Message:', message.__dict__.keys())
            #print('Message:', message.body.__dict__.keys())
            print('Message:', message.subject)
            messages.append("NEW EMAIL")
            messages.append(message.body.content)

            if (
                message.from_ is not None and
                message.from_.email_address is not None
            ):
                print('  From:', message.from_.email_address.name or 'NONE')
            else:
                print('  From: NONE')
            print('  Status:', 'Read' if message.is_read else 'Unread')
            print('  Received:', message.received_date_time)

        
        message_df = pd.DataFrame(messages)
        message_df.to_csv("test.csv",mode = 'w+')

        # If @odata.nextLink is present
        more_available = message_page.odata_next_link is not None
        print('\nMore messages available?', more_available, '\n')

async def send_mail(graph: Graph):
    # TODO
    return

async def create_draft(graph: Graph):
    # Send mail to the signed-in user
    # Get the user for their email address
    user = await graph.get_user()
    if user:
        user_email = user.mail or user.user_principal_name

        await graph.create_draft('Low','Testing Microsoft Graph', 'Hello world!', user_email or '')
        print('Mail sent.\n')

# Run main
asyncio.run(main())