from dash import dcc, html


def render():
    return html.Div([
        html.H3('Info tab'),
        dcc.Graph(
            figure={
                'data': [{
                    'x': [1, 2, 3],
                    'y': [3, 1, 2],
                    'type': 'bar'
                }]
            }
        )
    ])

def render_0():
    return html.Div([
        dcc.Tabs(id="tabs-example-inner", value='tab-1-example-graph-inner', children=[
            dcc.Tab(label='Tab One-inner', value='tab-1-example-graph-inner', children=[
                render()
            ]),
            dcc.Tab(label='Tab Two-inner', value='tab-2-example-graph-inner', children=[
                render()
            ]),
        ]),
    ])
