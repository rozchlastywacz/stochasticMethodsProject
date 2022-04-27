import dash
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1
from models.image_provider import get_image_from_dbn, rescale_grayscale_image, get_real_image

dash.register_page(__name__)

from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import numpy as np  # pip install numpy

np.random.seed(2020)

layout = html.Div(
    [
        dbc.Row(
            dbc.Col(
                html.Div(
                    children=[
                        dcc.Graph(id='image-graph'),
                        html.Div(id='image-type', hidden=False)
                    ],
                    id='image-div',
                    style={'textAlign': 'center'}
                ),
                width={"size": 6, "offset": 3}
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div(
                    children=[
                        dbc.Button('Real image', color='success', id='true-button'),
                        dbc.Button('False image', color='danger', id='false-button')
                    ],
                    id='button-div',
                    style={'textAlign': 'center'}
                ),
                width={"size": 6, "offset": 3}
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div(
                    children=[
                        dbc.Progress(
                            [
                                dbc.Progress(value=0, color='success', bar=True, id='success-bar'),
                                dbc.Progress(value=0, color='danger', bar=True, id='fail-bar')
                            ]
                        )
                    ],
                    id='output-div',
                    style={'textAlign': 'center'}
                ),
                width={"size": 6, "offset": 3}
            )
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
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig, image_type[t]


@callback(
    Output("success-bar", "value"),
    Output("fail-bar", "value"),
    Output("image-graph", "figure"),
    Output("image-type", "children"),
    Input("true-button", "n_clicks"),
    Input("false-button", "n_clicks"),
    Input("success-bar", "value"),
    Input("fail-bar", "value"),
    Input("image-type", "children")
)
def true_button_clicked(n_s, n_f, val_s, val_f, img_t):
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'true-button' and img_t == image_type[1] or button_id == 'false-button' and img_t == image_type[0]:
        val_s += 1
    else:
        val_f += 1
    return val_s, val_f, *create_image()

# @callback(
#     Output("fail-bar", "value"),
#     Input("false-button", "n_clicks"),
#     Input("fail-bar", "value"),
# )
# def false_button_clicked(n, val):
#     return val + 1
