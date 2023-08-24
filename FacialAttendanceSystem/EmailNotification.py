from email.message import EmailMessage
import ssl
import smtplib

email_sender = "aineuroface@gmail.com"
email_password = "pcrglrujnrconpyh"
email_receiver = "doshijainam69@gmail.com"

subject = "Test Email - Verified"

# Construct the HTML-formatted body with company stamps and details
with open("/Users/jainamdoshi/Desktop/Machine Learning Models/Projects/FacialAttendanceSystem/EmailBody.html","r") as file:
    body = file.read()

em = EmailMessage()
em['From'] = email_sender
em['To'] = email_receiver
em['Subject'] = subject
em.add_alternative(body, subtype='html')  # Set the body as HTML

certfile = "/Users/jainamdoshi/Desktop/Machine Learning Models/Projects/FacialAttendanceSystem/Credentials/cacert.pem"
context = ssl.create_default_context(cafile=certfile)

with smtplib.SMTP_SSL('smtp.gmail.com',465,context=context) as smtp:
    smtp.login(email_sender,email_password)
    smtp.sendmail(email_sender,email_receiver,em.as_string())
