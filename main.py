##### ##### ##### ##### ##### ##### Imports  ##### ##### ##### ##### ##### #####

from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
# import dash_daq as daq
import pandas as pd
import numpy as np
import plotly.express as px
from pyrsistent import v


##### ##### ##### ##### ##### ##### App Instantiation  ##### ##### ##### ##### ##### #####

app = Dash(
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"
    ]
)


##### ##### ##### ##### ##### ##### Figures  ##### ##### ##### ##### ##### #####

df = pd.read_csv(
    '/home/yinshe/Documents/PythonZone/DataViz/ExplanationProject/data/penguins_cleaned.csv')
column_options = {value: ' '.join(value.upper().split('_')) for value in set(df.columns) - set(['species'])}
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


##### ##### ##### ##### ##### ##### App Layout  ##### ##### ##### ##### ##### #####

app.layout = html.Div([


    # *********** BANs *********** #

    dbc.Row([

            dbc.Col(dbc.Row([

                dbc.Col([
                    html.H5('F1 Score'),
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
    

    # *********** Graphs *********** #

    dbc.Row(
        [
            dbc.Col(
                dcc.Graph(figure=scatter_fig, id='scatter_fig')
            , width=6),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=hist_fig, id='hist_fig')
                    ], width=7),
                    dbc.Col([
                        dcc.Graph(figure=count_fig, id='count_fig')
                    ], width=5)
                ]),
                dbc.Row([
                    dcc.Graph(figure=violin_fig, id='violin_fig')
                ]),
            ], width=6)
        ]
    , className='overlay figs'),


    # *********** Filters Offcanvas *********** #

    dbc.Button([
        html.I(className='fas fa-filter fa-lg fa-beat')
    ], id='open_filters', class_name='btn-floating btn-lg float'),

    dbc.Offcanvas([
            dbc.Row([
                html.H6('Graphs Configurations and Filters'),
                dbc.Label('Scatter Type', html_for='scatter-type'),
                dcc.RadioItems(options={'2': '2D', '3': '3D'}, value='2', inline=True, id='scatter-type', inputStyle={'margin-right': '10px', 'margin-left': '25px'}),
                dbc.Label('Scatter Axes', html_for='scatter-axes'),
                dcc.Dropdown(options=column_options, multi=True, id='scatter-axes'),
                dbc.Label('Histogram Feature', html_for='hist_feat'),
                dcc.Dropdown(options=column_options, multi=True, id='hist_feat'),
                dbc.Label('Count Plot Feature', html_for='count_feat'),
                dcc.Dropdown(options=column_options, multi=True, id='count_feat'),
                dbc.Label('Violin Plot Feature', html_for='violin_feat'),
                dcc.Dropdown(options=column_options, multi=True, id='violin_feat'),
            ]),
            html.Hr(),
            dbc.Row([
                html.H6('Model Parameters'),
                dbc.Form([
                    dbc.Label('', html_for='')
                ])
            ])
        ],
            id='filters_canvas',
            title='Dashboard Settings',
            placement='start',
            is_open=False)
    
], className='container-fluid')


##### ##### ##### ##### ##### ##### Callback Definition ##### ##### ##### ##### ##### #####


# *********** Offcanvas *********** #

@app.callback(
    Output("filters_canvas", "is_open"),
    Input("open_filters", "n_clicks"),
    [State("filters_canvas", "is_open")],
)
def toggle_filter(n1, is_open):
    if n1:
        return not is_open
    return is_open


# *********** CrossFiltering *********** #




##### ##### ##### ##### ##### ##### App Runner ##### ##### ##### ##### ##### #####

if __name__ == "__main__":
    app.run_server(debug=True)
