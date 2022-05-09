import dash
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
import numpy as np  # pip install numpy
import plotly.express as px
from dash import dcc, html, Input, Output, callback

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1
from database_manager.db_manager import append_new_answers
from models.image_provider import get_image_from_dbn, rescale_grayscale_image, get_real_image, get_starter_image

dash.register_page(__name__)
MAX_QUESTIONS = 10
np.random.seed(2020)


def load_initial_image():
    img = get_starter_image()

    fig = px.imshow(img)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                      paper_bgcolor="Gainsboro")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


true_button = dbc.Button('Real image', disabled=True, color='success', id='true-button', style={'marginRight': '10px'})
false_button = dbc.Button('False image', disabled=True, color='danger', id='false-button', style={'marginLeft': '10px'})
layout = html.Div(
    [
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H1('Turing test'),
                        html.Hr(),
                        html.P('On this page You can try to test Your skills '
                               'of recognizing what is true and what is false!'),
                        html.P('The test consist of 10 questions. '
                               'In each question one single image is shown and it could be: '
                               'real image taken from the dataset '
                               'or generated image by one of our trained models'),
                        html.P(
                            children=[
                                'In each question You - a human being - have to decide if ',
                                html.I('it is a real image or just fantasy')
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
                        html.P('Press the button below to start/reset Your test'),
                        dbc.Button('Restart test',
                                   color='primary',
                                   id='reset-test-button',
                                   style={'marginBottom': '10px'}
                                   ),
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
                        html.Hr(),
                        html.P('Your test has ended, please click reset button to start again'),
                    ],
                    id='dummy-div',
                    hidden=True,
                    style={'textAlign': 'center', 'marginTop': '10px'}
                ),
                width={"size": 6, "offset": 3}
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
        )

    ]
)

image_type = ['false', 'real']


def create_image():
    t = np.random.randint(0, 2)
    if t == 0:
        img = get_image_from_dbn()
    else:
        img = get_real_image()

    img = rescale_grayscale_image(img)
    fig = px.imshow(img)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                      paper_bgcolor="Gainsboro")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
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


@callback(
    Output("true-button", "disabled"),
    Output("false-button", "disabled"),
    Output("dummy-div", "hidden"),
    Input("browser-storage", "data"),
    prevent_initial_call=True
)
def toggle_buttons(storage_data):
    if storage_data['current_que'] < MAX_QUESTIONS:
        return False, False, True
    else:
        return True, True, False


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
    prevent_initial_call=True
)
def some_button_clicked(_n_s, _n_f, _n_r, val_s, val_f, storage_data):
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

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
            append_new_answers(storage_data['answers'])
            return val_s, str(val_s), val_f, str(val_f), n_img, storage_data

        return val_s, str(val_s), val_f, str(val_f), n_img, storage_data

    if button_id == 'reset-test-button':
        return reset_test_clicked(storage_data)
