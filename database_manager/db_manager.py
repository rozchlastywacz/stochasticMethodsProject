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
def calculate_answer_percent_score(answers):
    # answer = {'user': user_ans, 'correct': correct_ans}
    good = 0
    for ans in answers:
        if ans['user'] == ans['correct']:
            good += 1

    return 100.0 * good / len(answers)


def append_new_answers(answers, id):
    today = datetime.today().strftime('%Y-%m-%d')
    ref = db.reference('/turing_tests/')
    ref.push().set({'user_id': id, 'answers': answers, 'date': today})


def get_percent_distribution(user_id):
    ref = db.reference('/turing_tests/')
    data = ref.get()
    ans_percent = [calculate_answer_percent_score(row['answers']) for row in data.values()]
    user_ans_percent = [
        calculate_answer_percent_score(row['answers'])
        for row in data.values() if row['user_id'] == user_id
    ]
    return ans_percent, user_ans_percent

