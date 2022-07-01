from turtle import width
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import dash_daq as daq

app = Dash(
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com",
        "https://fonts.gstatic.com",
        "https://fonts.googleapis.com/css2?family=Quattrocento+Sans:wght@700&display=swap"
    ]
)

app.layout = html.Div([
    dbc.Row([
            dbc.Col(dbc.Row([
                dbc.Col([
                    html.H5('F1 Score'),
                    dbc.CardBody('95.7%', className='bans')
                ], width=2),
                dbc.Col([
                    html.H5('Recall'),
                    dbc.CardBody('85.4%', className='bans')
                ], width=2),
                dbc.Col([
                    html.H5('Precision'),
                    dbc.CardBody('90.5%', className='bans')
                ], width=2),
                dbc.Col([
                    html.H5('Accuracy'),
                    dbc.CardBody('98.3%', className='bans')
                ], width=2)
            ], className='justify-content-evenly'), width=12)
        ], style={'color': 'white'}),
    dbc.Row(
        [
            dbc.Col(
                dbc.Row([
                    daq.ToggleSwitch(
                        label='My toggle switch',
                        labelPosition='bottom'
                    )
                ], className='overlay')
            , width=6, id='overlay_col'),
            dbc.Col(html.Div("Two of three columns"), width=6),
        ]
    ),
    
    daq.BooleanSwitch(id='my-boolean-switch', on=False),
    

], className='container-fluid')

if __name__ == "__main__":
    app.run_server(debug=True)
