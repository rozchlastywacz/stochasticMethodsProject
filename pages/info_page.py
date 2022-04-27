
import dash
# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1
dash.register_page(__name__, path="/")

from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px

df = px.data.medals_wide(indexed=True)

layout = html.Div(
    [
        html.P("This page will contain information about models used", style={'textAlign': 'center'}),
        # dcc.Checklist(
        #     id="heatmaps-medals",
        #     options=[{"label": x, "value": x} for x in df.columns],
        #     value=df.columns.tolist(),
        #     style={'textAlign': 'center'}
        # ),
        # dcc.Graph(id="heatmaps-graph"),
    ],
    # style={
    #     'width': '50%',
    #     # 'margin': '0 auto'
    # }
)


# @callback(Output("heatmaps-graph", "figure"), Input("heatmaps-medals", "value"))
# def filter_heatmap(cols):
#     fig = px.imshow(df[cols])
#     return fig