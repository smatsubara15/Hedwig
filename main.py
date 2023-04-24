import asyncio
import configparser
from graph import Graph

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
        print('4. Make a Graph call')

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
            await make_graph_call(graph)
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
    message_page = await graph.get_inbox()
    if message_page is not None and message_page.value is not None:
        # Output each message's details
        for message in message_page.value:

            # print('Message:', message.__dict__.keys())
            #print('Message:', message.body.__dict__.keys())
            print('Message:', message.body.content)
            if (
                message.from_ is not None and
                message.from_.email_address is not None
            ):
                print('  From:', message.from_.email_address.name or 'NONE')
            else:
                print('  From: NONE')
            print('  Status:', 'Read' if message.is_read else 'Unread')
            print('  Received:', message.received_date_time)

        # If @odata.nextLink is present
        more_available = message_page.odata_next_link is not None
        print('\nMore messages available?', more_available, '\n')

async def send_mail(graph: Graph):
    # TODO
    return

async def make_graph_call(graph: Graph):
    # TODO
    return

# Run main
asyncio.run(main())