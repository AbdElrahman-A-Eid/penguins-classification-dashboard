from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import dash_daq as daq
import pandas as pd
import plotly.express as px

app = Dash(
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com",
        "https://fonts.gstatic.com",
        "https://fonts.googleapis.com/css2?family=Quattrocento+Sans:wght@700&display=swap"
    ]
)

df = pd.read_csv('/home/yinshe/Documents/PythonZone/DataViz/ExplanationProject/penguins_size.csv')
scatter_fig = px.scatter(df, x='culmen_length_mm', y='flipper_length_mm', color='species', template='simple_white', height=750)
scatter_fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'legend_bgcolor': 'rgba(0, 0, 0, 0)',
})
app.layout = html.Div([
    dbc.Row([
            dbc.Col(dbc.Row([
                dbc.Col([
                    html.H5('F1 Score'),
                    # dbc.CardBody('95.7%', className='bans')
                    dbc.Row([
                        dbc.Col([
                            dbc.CardBody('95.7%', className='bans Adelie')
                        ], width=3),
                        dbc.Col([
                            dbc.CardBody('95.7%', className='bans Chinstrap')
                        ], width=3),
                        dbc.Col([
                            dbc.CardBody('95.7%', className='bans Gentoo')
                        ], width=3),
                    ], className='justify-content-evenly'),
                ], width=4),
                dbc.Col([
                    html.H5('Recall'),
                    # dbc.CardBody('95.7%', className='bans')
                    dbc.Row([
                        dbc.Col([
                            dbc.CardBody('95.7%', className='bans Adelie')
                        ], width=3),
                        dbc.Col([
                            dbc.CardBody('95.7%', className='bans Chinstrap')
                        ], width=3),
                        dbc.Col([
                            dbc.CardBody('95.7%', className='bans Gentoo')
                        ], width=3),
                    ], className='justify-content-evenly'),
                ], width=4),
                dbc.Col([
                    html.H5('Precision'),
                    # dbc.CardBody('95.7%', className='bans')
                    dbc.Row([
                        dbc.Col([
                            dbc.CardBody('95.7%', className='bans Adelie')
                        ], width=3),
                        dbc.Col([
                            dbc.CardBody('95.7%', className='bans Chinstrap')
                        ], width=3),
                        dbc.Col([
                            dbc.CardBody('95.7%', className='bans Gentoo')
                        ], width=3),
                    ], className='justify-content-evenly'),
                ], width=4),
            ], className='justify-content-evenly overlay'), width=12)
        ], style={'color': 'white'}),
    dbc.Row(
        [
            dbc.Col(
                dcc.Graph(figure=scatter_fig)
            , width=6),
            dbc.Col([
                dbc.Row([
                    dbc.Col([

                    ], width=6),
                    dbc.Col([

                    ], width=6)
                ]),
                dbc.Row([

                ]),
            ], width=6)
        ]
    , className='overlay figs')
    
], className='container-fluid')

if __name__ == "__main__":
    app.run_server(debug=True)
