import uuid
import os
import dash
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
import numpy as np  # pip install numpy
import plotly.express as px
from dash import dcc, html, Input, Output, callback, State

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1
from dash.exceptions import PreventUpdate

from database_manager.db_manager import append_new_answers
from models.image_provider import get_image_from_dbn, rescale_grayscale_image, get_real_image, get_starter_image

dash.register_page(__name__)
MAX_QUESTIONS = int(os.environ.get('MAX_QUE', 10))


def create_figure(img):
    fig = px.imshow(img)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                      paper_bgcolor="Gainsboro")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def load_initial_image():
    img = get_starter_image()

    fig = create_figure(img)
    return fig


true_button = dbc.Button('Real image', disabled=True, color='secondary', id='true-button',
                         style={'marginRight': '10px'})
false_button = dbc.Button('False image', disabled=True, color='secondary', id='false-button',
                          style={'marginLeft': '10px'})
layout = html.Div(
    [
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H1('Turing test'),
                        html.P('On this page You can try to test Your skills '
                               'of recognizing what is true and what is false!'),
                        html.P('The test consist of {} questions. '.format(MAX_QUESTIONS) +
                               'In each question one single image is shown and it could be: '
                               'real image taken from the dataset '
                               'or generated image by one of our trained models.'),
                        html.P(
                            children=[
                                'In each question You - a human being - have to decide if ',
                                html.A(html.I('it is a real image or just fantasy.'),
                                       href='https://www.youtube.com/watch?v=fJ9rUzIMcZQ&ab_channel=QueenOfficial')
                            ]
                        ),
                        html.Hr(),

                    ],
                    id='test-info-div',
                    style={'textAlign': 'center'}
                ),
                # width={"size": 6, "offset": 3}
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div(
                    children=[
                        html.H2('How to do the test'),
                        html.P('Initially, some dummy image will be displayed, answers button disabled.'),
                        html.P('After pressing the button below app will prepare questions for You (few seconds).'),
                        html.P('When preparation is done, '
                               'the first image will be displayed and answers buttons will be unlocked.'),
                        html.P('Look at the image, choose answer by clicking button '
                               '- after that next image will be displayed.'),
                        html.P('After You finish the test, '
                               'the last image will be displayed, answers buttons will be disabled.'),
                        html.Hr(),
                        html.P('Press the button below to run/re-run Your test'),
                        dbc.Button('Run test',
                                   color='primary',
                                   id='reset-test-button',
                                   style={'marginBottom': '10px'}
                                   ),
                    ],
                    id='instruction-div',
                    style={'textAlign': 'center'}
                ),
                # width={"size": 6, "offset": 3}
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div(
                    children=[
                        dbc.Spinner(
                            dcc.Graph(figure=load_initial_image(), id='image-graph', config={'staticPlot': True}),
                            color='secondary',
                            spinner_style={"width": "6rem", "height": "6rem"}
                        ),
                    ],
                    id='image-div',
                    style={'textAlign': 'center'}
                ),
                # width={"size": 6, "offset": 3}
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div(
                    children=[
                        true_button,
                        false_button
                    ],
                    id='button-div',
                    style={'textAlign': 'center', 'marginTop': '20px'}
                ),
                # width={"size": 6, "offset": 3}
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div(
                    children=[
                        dbc.Progress(
                            [
                                dbc.Progress(value=0, color='success', bar=True, id='success-bar', max=MAX_QUESTIONS),
                                dbc.Progress(value=0, color='danger', bar=True, id='fail-bar', max=MAX_QUESTIONS)
                            ],
                            max=MAX_QUESTIONS,
                            style={"height": "2rem"}
                        )
                    ],
                    id='output-div',
                    style={'textAlign': 'center', 'marginTop': '20px'}
                ),
                width={"size": 6, "offset": 3}
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div(
                    children=[
                        html.P('Progress bar has the same length as number of questions, '
                               'the green bar represents correct answers '
                               '- times when You properly labeled real images as real and generated as false, '
                               'and the red bar represents incorrect answers.'),
                        html.P('If you answer correctly the green one gets longer, otherwise the red gets longer.'),
                        html.Hr(),
                    ],
                    id='bar-explanation-div',
                    style={'textAlign': 'center', 'marginTop': '10px'}
                ),
                # width={"size": 6, "offset": 3}
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Footer(
                    "Cwikla et al, AGH UST",
                    style={'textAlign': 'center', 'marginTop': '100px'}
                ),
                width={"size": 6, "offset": 3}
            ),
        ),
        dcc.Store(
            id='browser-storage',
            data={'current_que': 0, 'questions': [], 'answers': []}
        ),
        dcc.Store(id='local-storage', storage_type='local'),
        dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle("You finished the test!")
                ),
                dbc.ModalBody(
                    'Your score: {}% If you wish to try again, close this pop-up and press re-run button.',
                    id='results-modal-body'
                ),
                dbc.ModalFooter(dbc.Button("Close", id='close-modal'))
            ],
            id='results-modal',
            is_open=False
        ),
        html.Div(
            children=[
                html.Div(
                    children=[],
                    id='garbage-output-0',
                    hidden=True
                ),
                html.Div(
                    children=[],
                    id='garbage-output-1',
                    hidden=True
                )
            ],
            id='garbage-outputs',
            hidden=True
        ),


    ]
)
# move after clicking reset
dash.clientside_callback(
    """
    function(clicks, elemid) {
        document.getElementById(elemid).scrollIntoView({
          behavior: 'smooth'
        });
    }
    """,
    Output('garbage-output-0', 'children'),
    [Input('reset-test-button', 'n_clicks')],
    [State('image-graph', 'id')],
    prevent_initial_call=True

)

# move after clicking close in modal
dash.clientside_callback(
    """
    function(clicks, elemid) {
        document.getElementById(elemid).scrollIntoView({
          behavior: 'smooth'
        });
    }
    """,
    Output('garbage-output-1', 'children'),
    [Input('close-modal', 'n_clicks')],
    [State('instruction-div', 'id')],
    prevent_initial_call=True

)

image_type = ['false', 'real']


def create_image():
    t = np.random.randint(0, 2)
    if t == 0:
        img = get_image_from_dbn()
    else:
        img = get_real_image()

    img = rescale_grayscale_image(img)
    fig = create_figure(img)
    return fig, image_type[t]


def reset_test_clicked(storage_data):
    val_s = 0
    val_f = 0

    questions = []
    for i in range(MAX_QUESTIONS):
        img, label = create_image()
        question = {'img': img, 'label': label}
        questions.append(question)
    storage_data['questions'] = questions

    storage_data['answers'] = []
    storage_data['current_que'] = 0
    n_img = questions[0]['img']

    return val_s, str(val_s), val_f, str(val_f), n_img, storage_data


def answer_is_correct(button_id, img_t):
    return button_id == 'true-button' and img_t == image_type[1] \
           or button_id == 'false-button' and img_t == image_type[0]


def calculate_answer_percent_score(answers):
    # answer = {'user': user_ans, 'correct': correct_ans}
    good = 0
    for ans in answers:
        if ans['user'] == ans['correct']:
            good += 1

    return 100.0 * good / len(answers)


@callback(
    Output("local-storage", "data"),
    Input('garbage-output-0', 'children'),
    State("local-storage", "data")
)
def save_unique_id(_, data):
    if data is not None:
        raise PreventUpdate
    data = data or {'uniq_browser_id': str(uuid.uuid1())}
    return data


@callback(
    Output("results-modal", "is_open"),
    Output("results-modal-body", "children"),
    Input("close-modal", "n_clicks"),
    Input("browser-storage", "data"),
    prevent_initial_call=True
)
def close_modal_callback(n, storage_data):
    modal_msg = 'Your score: {}% If you wish to try again, close this pop-up and press re-run button.'
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'close-modal':
        return False, modal_msg
    else:
        if storage_data['current_que'] < MAX_QUESTIONS:
            return False, modal_msg
        answers = storage_data['answers']
        score = calculate_answer_percent_score(answers)
        return True, modal_msg.format(score)


@callback(
    Output("reset-test-button", "children"),
    Input("reset-test-button", "n_clicks"),
    prevent_initial_call=True
)
def start_button_clicked(_n):
    return 'Re-run test'


@callback(
    Output("true-button", "disabled"),
    Output("false-button", "disabled"),
    Input("browser-storage", "data"),
    prevent_initial_call=True
)
def toggle_buttons(storage_data):
    if storage_data['current_que'] < MAX_QUESTIONS:
        return False, False
    else:
        return True, True


@callback(
    Output("success-bar", "value"),
    Output("success-bar", "label"),
    Output("fail-bar", "value"),
    Output("fail-bar", "label"),
    Output("image-graph", "figure"),
    Output("browser-storage", "data"),
    Input("true-button", "n_clicks"),
    Input("false-button", "n_clicks"),
    Input("reset-test-button", "n_clicks"),
    Input("success-bar", "value"),
    Input("fail-bar", "value"),
    Input("browser-storage", "data"),
    Input("local-storage", "data"),
    prevent_initial_call=True
)
def some_button_clicked(_n_s, _n_f, _n_r, val_s, val_f, storage_data, local_storage_data):
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'local-storage':
        raise PreventUpdate

    answer_buttons = ['true-button', 'false-button']
    if button_id in answer_buttons:
        img_i = storage_data['current_que']
        img_t = storage_data['questions'][img_i]['label']

        if answer_is_correct(button_id, img_t):
            val_s += 1
        else:
            val_f += 1

        user_ans = True if button_id == answer_buttons[0] else False
        correct_ans = True if img_t == image_type[1] else False
        answer = {'user': user_ans, 'correct': correct_ans}
        storage_data['answers'].append(answer)

        storage_data['current_que'] += 1

        img_i = storage_data['current_que']
        if img_i < MAX_QUESTIONS:
            n_img = storage_data['questions'][img_i]['img']
        else:
            n_img = storage_data['questions'][-1]['img']
            # TODO save answers
            # print(storage_data['answers'])
            append_new_answers(storage_data['answers'], local_storage_data['uniq_browser_id'])
            return val_s, str(val_s), val_f, str(val_f), n_img, storage_data

        return val_s, str(val_s), val_f, str(val_f), n_img, storage_data

    if button_id == 'reset-test-button':
        return reset_test_clicked(storage_data)
