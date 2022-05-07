import dash
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
import numpy as np  # pip install numpy
import plotly.express as px
from dash import dcc, html, Input, Output, callback

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1
from models.image_provider import get_image_from_dbn, rescale_grayscale_image, get_real_image

dash.register_page(__name__)
MAX_QUESTIONS = 10.0
np.random.seed(2020)

image_type = ['false', 'real']
dupa = 0


def create_image():
    global dupa
    dupa += 1
    print("dupa = {}".format(dupa))
    t = np.random.randint(0, 2)
    if t == 0:
        img = get_image_from_dbn()
    else:
        img = get_real_image()

    img = rescale_grayscale_image(img)
    fig = px.imshow(img)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                      paper_bgcolor="LightSteelBlue")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig, image_type[t]


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
                        dbc.Spinner(
                            dcc.Graph(figure=create_image()[0], id='image-graph', config={'staticPlot': True}),
                            color='secondary',
                            spinner_style={"width": "6rem", "height": "6rem"}
                        ),
                        html.Div(id='image-type', hidden=True)
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
                        dbc.Button('Real image',
                                   color='success',
                                   id='true-button',
                                   style={'marginRight': '10px'}
                                   ),
                        dbc.Button('False image',
                                   color='danger',
                                   id='false-button',
                                   active=False,
                                   style={'marginLeft': '10px'}
                                   )
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
        )
    ]
)


@callback(
    Output("success-bar", "value"),
    Output("success-bar", "label"),
    Output("fail-bar", "value"),
    Output("fail-bar", "label"),
    Output("image-graph", "figure"),
    Output("image-type", "children"),
    Input("true-button", "n_clicks"),
    Input("false-button", "n_clicks"),
    Input("success-bar", "value"),
    Input("fail-bar", "value"),
    Input("image-type", "children"),
    prevent_initial_call=True
)
def true_button_clicked(_n_s, _n_f, val_s, val_f, img_t):
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'true-button' and img_t == image_type[1] or button_id == 'false-button' and img_t == image_type[0]:
        val_s += 1
    else:
        val_f += 1
    return val_s, str(val_s), val_f, str(val_f), *create_image()

# @callback(
#     Output("fail-bar", "value"),
#     Input("false-button", "n_clicks"),
#     Input("fail-bar", "value"),
# )
# def false_button_clicked(n, val):
#     return val + 1
