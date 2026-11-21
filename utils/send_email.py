import smtplib, ssl, os
from email.message import EmailMessage

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
ALERT_TO = os.getenv("ALERT_RECIPIENT", EMAIL_USER)

def send_email(subject, body, attachment_path=None):
    if not EMAIL_USER or not EMAIL_PASS or not ALERT_TO:
        print("Email not configured. Set EMAIL_USER, EMAIL_PASS, ALERT_RECIPIENT.")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = ALERT_TO
    msg.set_content(body)

    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:
            data = f.read()
        msg.add_attachment(data, maintype="image", subtype="jpeg", filename=os.path.basename(attachment_path))

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls(context=context)
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        return True
    except Exception as e:
        print("Failed to send email:", e)
        return False
