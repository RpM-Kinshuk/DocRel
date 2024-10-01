import smtplib
from utils.pwr import givepw, givesender
from utils.genmail import generate_mail
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# from flask import Flask
# from flask_mail import Mail, Message

def rnf_mail(email, rainfall):
    message = EmailMessage()
    message['from'] = givesender()
    message['to'] = email
    message['subject'] = "Rainfall Prediction Result"
    message.set_content(
        "Hello, this is an automated message from the Rainfall Prediction Page!\n\n\t\tThe Predicted Rainfall is: {0:.2f} mm.\n\nRegards,\nKin and Bells :3".format(rainfall))
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(givesender(), givepw())
    s.send_message(message)
    s.quit()


def doc_mail(email, query, keywords, st_year, en_year, pubs, top_n, model, sim_measure, results):
    message = EmailMessage()
    message['from'] = givesender()
    message['to'] = email
    message['subject'] = "DocRel Results"
    # text_part = "\n\n\
    #     Hello, this is an automated message from the Rainfall Prediction Page!\n\n\t\tThe Predicted Rainfall is: {0:.2f} mm.\n\nRegards,\nKin and Bells :3\
    #         ".format(rainfall)
    html_part = generate_mail(query, keywords, st_year, en_year, pubs, top_n, model, sim_measure, results)
    message.set_content(f'''{html_part}''',subtype='html')
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(givesender(), givepw())
    s.sendmail(givesender(), email, message.as_string())
    s.quit()


# def flask_send(email, rainfall):
#     app = Flask(__name__)
#     app.run(debug=True)
#     mail = Mail(app)
#     app.config["MAIL_DEFAULT_SENDER"] = givesender()
#     app.config["MAIL_PASSWORD"] = givepw()
#     app.config["MAIL_PORT"] = 587
#     app.config["MAIL_SERVER"] = "smtp.gmail.com"
#     app.config["MAIL_USE_TLS"] = True
#     app.config["MAIL_USERNAME"] = "Kinshuk"
#     mail = Mail(app)
#     msg = Message(
#         sender=givesender(),
#         recipients=email,
#         subject='Rainfall Prediction Result'
#     )
#     msg.body = 'Hello, this is an automated message from the Rainfall Prediction Page!\n\n\t\tThe Predicted Rainfall is: {0:.2f} mm.\n\nRegards,\nKin and Bells :3'.format(rainfall)
#     mail.send(msg)
