import yaml
from yaml.loader import SafeLoader

conf = yaml.load(open('utils/cred.yml'),Loader=SafeLoader)
sender_email = conf['us']['email']
sender_pw = conf['us']['pw']

def givesender():
    return sender_email

def givepw():
    return sender_pw
