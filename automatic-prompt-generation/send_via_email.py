import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate

# Configure this with your email details
send_from = "ilatyarshia@gmail.com"
send_to = "arshia.ilaty99@gmail.com"
subject = "JupyterHub Files"

msg = MIMEMultipart()
msg["From"] = send_from
msg["To"] = send_to
msg["Date"] = formatdate(localtime=True)
msg["Subject"] = subject

# Attach the file
part = MIMEBase("application", "octet-stream")
with open("/home/jovyan/prompts.tar.gz", "rb") as file:
    part.set_payload(file.read())
encoders.encode_base64(part)
part.add_header("Content-Disposition", "attachment", filename="prompts.tar.gz")
msg.attach(part)

# Setup SMTP server and send
smtp = smtplib.SMTP("smtp-server.example.com", 587)
smtp.starttls()
smtp.login("username", "password")
smtp.sendmail(send_from, send_to, msg.as_string())
smtp.quit()
