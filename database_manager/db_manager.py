import json
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db

# database access setup
LOCAL_CRT_PATH = "D:/Kody/semestr_8/stochastyczne/stochastic_project/ac-stochastic-firebase-adminsdk-pm0us-9806178558.json"

env_crt = os.environ.get('CRT', '')
crt = None
if env_crt:
    crt = json.loads(env_crt, strict=False)
else:
    f = open(LOCAL_CRT_PATH)
    crt = json.load(f)

cred = credentials.Certificate(crt)
db_app = firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://ac-stochastic-default-rtdb.europe-west1.firebasedatabase.app/'
})


# db test
# ref = db.reference("/Books/")
# data_str = '{"Book1":{ "Title": "The Fellowship of the Ring", "Author": "J.R.R. Tolkien", "Genre": "Epic fantasy", "Price": 100}}'
# data = json.loads(data_str)
# ref.push().set(data)


def append_new_answers(answers):
    today = datetime.today().strftime('%Y-%m-%d')
    ref = db.reference('/turing_tests/{}'.format(today))
    ref.push().set(answers)
