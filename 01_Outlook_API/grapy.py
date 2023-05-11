from configparser import SectionProxy
from azure.identity import DeviceCodeCredential
from kiota_authentication_azure.azure_identity_authentication_provider import (
    AzureIdentityAuthenticationProvider)
from msgraph import GraphRequestAdapter, GraphServiceClient
from msgraph.generated.me.me_request_builder import MeRequestBuilder
from msgraph.generated.me.mail_folders.item.messages.messages_request_builder import (
    MessagesRequestBuilder)
from msgraph.generated.me.send_mail.send_mail_post_request_body import SendMailPostRequestBody
from msgraph.generated.models.message import Message
from msgraph.generated.models.item_body import ItemBody
from msgraph.generated.models.body_type import BodyType
from msgraph.generated.models.recipient import Recipient
from msgraph.generated.models.email_address import EmailAddress

class Graph:
    settings: SectionProxy
    device_code_credential: DeviceCodeCredential
    adapter: GraphRequestAdapter
    user_client: GraphServiceClient

    def __init__(self, config: SectionProxy):
        self.settings = config
        client_id = self.settings['clientId']
        tenant_id = self.settings['tenantId']
        graph_scopes = self.settings['graphUserScopes'].split(' ')

        self.device_code_credential = DeviceCodeCredential(client_id, tenant_id = tenant_id)
        auth_provider = AzureIdentityAuthenticationProvider(
            self.device_code_credential,
            scopes=graph_scopes)
        self.adapter = GraphRequestAdapter(auth_provider)
        self.user_client = GraphServiceClient(self.adapter)

    async def get_user_token(self):
        graph_scopes = self.settings['graphUserScopes']
        access_token = self.device_code_credential.get_token(graph_scopes)
        return access_token.token

    async def get_user(self):
        # Only request specific properties using $select
        query_params = MeRequestBuilder.MeRequestBuilderGetQueryParameters(
            select=['displayName', 'mail', 'userPrincipalName']
        )
        request_config = MeRequestBuilder.MeRequestBuilderGetRequestConfiguration(
            query_parameters=query_params
        )

        user = await self.user_client.me.get(request_configuration=request_config)
        return user

    async def get_inbox(self):

        # change 'top' to change the amount of emails scraped from a page
        query_params = MessagesRequestBuilder.MessagesRequestBuilderGetQueryParameters(
            # Only request specific properties
            select=['from', 'isRead', 'receivedDateTime', 'subject','body'],
            # Get at most 25 results
            top=25,
            # Sort by received time, newest first
            orderby=['receivedDateTime DESC']
        )

        request_config = MessagesRequestBuilder.MessagesRequestBuilderGetRequestConfiguration(
            query_parameters= query_params
        )

        # can change inbox to other mailboxes: https://learn.microsoft.com/en-us/graph/api/resources/mailfolder?view=graph-rest-1.0
        messages = await self.user_client.me.messages.get(request_configuration=request_config)
        return messages

    async def create_draft(self, importance: str, subject: str, body: str, recipient: str):
        message = Message()
        message.subject = subject

        message.importance = importance

        message.body = ItemBody()
        message.body.content_type = BodyType.Text
        message.body.content = body

        to_recipient = Recipient()
        to_recipient.email_address = EmailAddress()
        to_recipient.email_address.address = recipient
        message.to_recipients = []
        message.to_recipients.append(to_recipient)









        message.is_draft = True

        # request_body = SendMailPostRequestBody()
        # request_body.message = message

        await self.user_client.me.messages.post(message)