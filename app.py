from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from html_renderers import info_tab, turing_test_tab

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets, title='stochasticMethodsProject')

app.layout = html.Div([
    html.H1('Generative models comparison', style={'textAlign': 'center'}),
    dcc.Tabs(id="tabs-example-graph", value='tab-1-example-graph', children=[
        dcc.Tab(label='About models', value='tab-1-example-graph'),
        dcc.Tab(label='Turing test', value='tab-2-example-graph'),
    ]),
    html.Div(id='tabs-content-example-graph')
])


@app.callback(Output('tabs-content-example-graph', 'children'),
              Input('tabs-example-graph', 'value'))
def render_content(tab):
    if tab == 'tab-1-example-graph':
        return info_tab.render()
    elif tab == 'tab-2-example-graph':
        return turing_test_tab.render()


if __name__ == '__main__':
    app.run_server(debug=True)
