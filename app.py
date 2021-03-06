import dash  # pip install dash
import dash_labs as dl  # pip install dash-labs
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1

app = dash.Dash(
    __name__, plugins=[dl.plugins.pages],
    external_stylesheets=[dbc.themes.SLATE],
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=1",
        }
    ]
)

server = app.server

# for x in dash.page_registry.values():
#     print(x)

navbar = dbc.NavbarSimple(
    [
        dbc.NavItem(dbc.NavLink(page["name"], href=page["path"]))
        for page in dash.page_registry.values()
        if page["module"] != "pages.not_found_404"
    ],
    brand="Generating synthetic images",
    color="primary",
    dark=True,
    className="mb-2",
)

app.layout = dbc.Container(
    [navbar, dl.plugins.page_container],
    fluid=True,
)

if __name__ == "__main__":
    app.run_server(debug=True)
