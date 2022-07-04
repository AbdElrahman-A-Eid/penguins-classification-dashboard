from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
# import dash_daq as daq
import pandas as pd
import plotly.express as px
from pyrsistent import v

app = Dash(
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com",
        "https://fonts.gstatic.com",
        "https://fonts.googleapis.com/css2?family=Quattrocento+Sans:wght@700&display=swap"
    ]
)

df = pd.read_csv(
    '/home/yinshe/Documents/PythonZone/DataViz/ExplanationProject/data/penguins_cleaned.csv')
scatter_fig = px.scatter(df, x='culmen_length_mm', y='flipper_length_mm', color='species', template='simple_white', size=[2 for _ in range(len(df))], size_max=8, height=778)
hist_fig = px.histogram(df, x="body_mass_g", template='simple_white', color='species', barmode="overlay", nbins=25, height=400)
count_fig = px.histogram(df, x=['species', 'island'], template='simple_white', color='species', barmode='group', height=400)
violin_fig = px.violin(df, y="culmen_length_mm", x="species", color='species', box=True, template='simple_white', facet_col='sex', height=400)
no_color = 'rgba(0, 0, 0, 0)'
for fig in [scatter_fig, count_fig, hist_fig, violin_fig]:
    legend_bool = fig == scatter_fig
    margin_dict = dict(l=10, r=10, t=15, b=0) if fig==scatter_fig else dict(l=10, r=10, t=20, b=70)
    fig.update_layout({
        'plot_bgcolor': no_color,
        'paper_bgcolor': no_color,
        'legend_bgcolor': no_color,
        'showlegend': legend_bool,
        'margin': margin_dict
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
                        dcc.Graph(figure=hist_fig)
                    ], width=7),
                    dbc.Col([
                        dcc.Graph(figure=count_fig)
                    ], width=5)
                ]),
                dbc.Row([
                    dcc.Graph(figure=violin_fig)
                ]),
            ], width=6)
        ]
    , className='overlay figs')
    
], className='container-fluid')

if __name__ == "__main__":
    app.run_server(debug=True)
