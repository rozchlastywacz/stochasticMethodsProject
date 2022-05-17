import dash
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
import numpy as np
import plotly.express as px
from dash import dcc, html, Input, Output, callback, State

from database_manager.db_manager import get_percent_distribution

dash.register_page(__name__)

layout = html.Div(
    [
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H1('Histogram of percentages of correct answers in turing tests'),
                        html.P('On this page You can see how well other people dealt with our model'),
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
                            [
                                dcc.Graph(id="histograms-graph"),
                            ],
                            color='secondary',
                            spinner_style={"width": "6rem", "height": "6rem"}
                        ),
                        html.Hr(),
                        html.P("Number of bins:"),
                        dcc.Slider(
                            min=5,
                            max=25,
                            step=None,
                            marks={
                                5: '5',
                                10: '10',
                                25: '25',
                            },
                            value=10,
                            id='histograms-mean')
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
                    children=[],
                    id='user-info-div',
                    style={'textAlign': 'center'}
                ),
                # width={"size": 6, "offset": 3}
            )
        ),
        dcc.Store(id='local-storage', storage_type='local')

    ]
)
data, user_data = None, None


def get_simple_stats(d):
    mean = np.mean(d)
    std = np.std(d)
    min_v = np.min(d)
    max_v = np.max(d)
    length = len(d)

    return min_v, max_v, mean, std, length


def get_user_summary():
    g_min, g_max, g_mean, g_std, g_len = get_simple_stats(data)
    g_info = 'For all answers: number of tests: {}, minimum {}%, maximum {}%, mean {:.1f}%, std {:.1f}%'.format(g_len,
                                                                                                                g_min,
                                                                                                                g_max,
                                                                                                                g_mean,
                                                                                                                g_std,
                                                                                                                )
    u_info = ''
    if user_data:
        u_min, u_max, u_mean, u_std, u_len = get_simple_stats(user_data)
        u_info = 'For Your answers: number of tests: {}, minimum {}%, maximum {}%, mean {:.1f}%, std {:.1f}%'.format(
            u_len, u_min,
            u_max,
            u_mean,
            u_std,
            )
    return [
        html.Hr(),
        html.H2('Summary'),
        html.P('Lets look on some basic statistics of tests results'),
        html.P(g_info),
        html.P(u_info),
        html.Hr(),
    ]


@callback(
    Output("histograms-mean", "value"),
    Output("user-info-div", "children"),
    Input("test-info-div", "children"),
    Input("local-storage", "data")
)
def fetch_data(_, store):
    global data, user_data
    user_id = store['uniq_browser_id']
    data, user_data = get_percent_distribution(user_id)
    return 10, get_user_summary()


@callback(
    Output("histograms-graph", "figure"),
    Input("histograms-mean", "value"),
    prevent_initial_call=True

)
def update_figure(bins):

    fig = px.histogram(
        data,
        nbins=bins,
        range_x=[0, 100],
        text_auto=True,
        color_discrete_sequence=['dimgrey'],
        opacity=0.8,
    )
    fig.update_layout(
        bargap=0.2,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="Gainsboro",
        xaxis_title='percentages',
        yaxis_title='answers count'
    )
    return fig
