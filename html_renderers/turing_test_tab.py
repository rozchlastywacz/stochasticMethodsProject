from dash import dcc, html


def render():
    return html.Div([
        html.H3('Turing test tab'),
        dcc.Graph(
            id='graph-2-tabs-dcc',
            figure={
                'data': [{
                    'x': [1, 2, 3],
                    'y': [5, 10, 6],
                    'type': 'bar'
                }]
            }
        )
    ])
