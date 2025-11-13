import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import imapclient
import email
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from mcp.types import SamplingMessage, TextContent

async def _send_email(to: str, subject: str, body: str):
    email_user = os.environ['EMAIL_USER']
    email_password = os.environ['EMAIL_APP_PASSWORD']
    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = to
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(email_user, email_password)
        server.send_message(msg)


mcp = FastMCP(name="email-server")

@mcp.tool()
async def create_draft_reply(email_id: str, email_content: str, ctx: Context[ServerSession, None]) -> str:
    """Generate a reply to an email using LLM sampling.
    
    Args:
        email_id: The ID of the email to reply to.
        email_content: The content of the email to reply to.
    """
    prompt = f"Draft a reply to the following email: {email_id} with the following content: {email_content}. The reply should be in the same language as the email."

    result = await ctx.session.create_message(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=prompt),
            )
        ],
        max_tokens=1000,
    )

    if result.content.type == "text":
        return result.content.text
    return str(result.content)

@mcp.tool()
async def get_unread_emails(ctx: Context[ServerSession, None]) -> str:
    """Get unread emails"""
    email_user = os.environ['EMAIL_USER']
    email_password = os.environ['EMAIL_APP_PASSWORD']
    
    with imapclient.IMAPClient('imap.gmail.com', 993) as server:
        server.login(email_user, email_password)
        server.select_folder("INBOX", readonly=True)
        emails = []
        messages = server.search("UNSEEN")
        for uid, message_data in server.fetch(messages, "RFC822").items():
            email_message = email.message_from_bytes(message_data[b"RFC822"])
            emails.append({"id": uid, "subject": email_message.get("Subject"), "from": email_message.get("From")})
        
    return f"Unread emails: {'\n'.join([f'{email['id']}: {email['subject']} from {email['from']}' for email in emails])}"

@mcp.tool()
async def send_email(to: str, subject: str, body: str, ctx: Context[ServerSession, None]) -> str:
    """Send an email"""
    await _send_email(to, subject, body)
    return f"✓ Email sent to {to}"


def main():
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()