# -*- coding: utf-8 -*-

##### ##### ##### ##### ##### ##### Imports  ##### ##### ##### ##### ##### #####

from datetime import datetime
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import re


##### ##### ##### ##### ##### ##### Variables  ##### ##### ##### ##### ##### #####

df = pd.read_csv('/'.join(re.split('[\\/]', __file__)[:-1]) + '/assets/data/penguins_cleaned.csv')

columns_dict = {'species':'Species', 'culmen_length_mm':'Culmen Length (mm)', 'culmen_depth_mm':'Culmen Depth (mm)',
 'flipper_length_mm':'Flipper Length (mm)', 'body_mass_g':'Body Mass (g)', 'sex':'Sex', 'island':'Island'}
column_options = {k: v for k, v in columns_dict.items() if k not in ['species', 'sex', 'island']}
cat_cols = {'sex':'Sex', 'island':'Island'}

c_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 20]
iter_values = [1, 5, 10, 15, 20, 50]
k_values = [1, 2, 3, 4, 5, 6, 7, 8]
weight_values = ['uniform', 'distance']


##### ##### ##### ##### ##### ##### App Instantiation  ##### ##### ##### ##### ##### #####

app = Dash(
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"
    ]
)


##### ##### ##### ##### ##### ##### Helper Functions  ##### ##### ##### ##### ##### #####

def update_fig_layout(fig_list):
    no_color = 'rgba(0, 0, 0, 0)'
    
    for i, fig in enumerate(fig_list):
        legend_bool = i == 0
        margin_dict = dict(l=10, r=10, t=15, b=15) if i==0 else dict(l=10, r=10, t=20, b=70)
        fig.update_layout({
            'plot_bgcolor': no_color,
            'paper_bgcolor': no_color,
            'legend_bgcolor': no_color,
            'showlegend': legend_bool,
            'margin': margin_dict
        })
    

def generate_figures(type='2', scatter_axes=list(column_options.keys())[:2], hist_feat=list(column_options.keys())[0], count_feat=list(cat_cols.keys())[0], violin_feat=list(column_options.keys())[0]):
    if int(type) == 2:
        scatter = px.scatter(df, x=scatter_axes[0], y=scatter_axes[1], color='species', template='simple_white', size=[2 for _ in range(len(df))], size_max=8, height=778)
    else:
        scatter = px.scatter_3d(df, x=scatter_axes[0], y=scatter_axes[1], z=scatter_axes[2], color='species',\
             size=[2 for _ in range(len(df))], size_max=20, height=776, color_discrete_map={'Adelie': '#2077B4', 'Chinstrap':'#FB8139', 'Gentoo': '#49A02D'})
    
    hist = px.histogram(df, x=hist_feat, template='simple_white', color='species', barmode="overlay", nbins=25, height=400)
    count = px.histogram(df, x=['species', count_feat], template='simple_white', color='species', barmode='group', height=400)
    violin = px.violin(df, y=violin_feat, x="species", color='species', box=True, template='simple_white', facet_col='sex', height=400)
    
    update_fig_layout([scatter, hist, count, violin])
    return scatter, hist, count, violin


def prepare_data(norm, imb):
    pass


def generate_model(type, p1, p2):
    if type=='logistic':
        pass
    else:
        pass


    

##### ##### ##### ##### ##### ##### Load Default Figures  ##### ##### ##### ##### ##### #####

scatter_fig, hist_fig, count_fig, violin_fig = generate_figures()


##### ##### ##### ##### ##### ##### App Layout  ##### ##### ##### ##### ##### #####

app.layout = html.Div([

    html.Div([
        # *********** BANs *********** #

        dbc.Row([

                dbc.Col(dbc.Row([

                    dbc.Col([
                        html.H5('F1 Score'),
                        dbc.Row([
                            dbc.Col([
                                dbc.CardBody('95.7%', id='f-1', className='bans Adelie')
                            ], width=3),
                            dbc.Col([
                                dbc.CardBody('95.7%', id='f-2', className='bans Chinstrap')
                            ], width=3),
                            dbc.Col([
                                dbc.CardBody('95.7%', id='f-3', className='bans Gentoo')
                            ], width=3),
                        ], className='justify-content-evenly'),
                    ], width=4),

                    dbc.Col([
                        html.H5('Recall'),
                        dbc.Row([
                            dbc.Col([
                                dbc.CardBody('95.7%', id='r-1', className='bans Adelie')
                            ], width=3),
                            dbc.Col([
                                dbc.CardBody('95.7%', id='r-2', className='bans Chinstrap')
                            ], width=3),
                            dbc.Col([
                                dbc.CardBody('95.7%', id='r-3', className='bans Gentoo')
                            ], width=3),
                        ], className='justify-content-evenly'),
                    ], width=4),

                    dbc.Col([
                        html.H5('Precision'),
                        dbc.Row([
                            dbc.Col([
                                dbc.CardBody('95.7%', id='p-1', className='bans Adelie')
                            ], width=3),
                            dbc.Col([
                                dbc.CardBody('95.7%', id='p-2', className='bans Chinstrap')
                            ], width=3),
                            dbc.Col([
                                dbc.CardBody('95.7%', id='p-3', className='bans Gentoo')
                            ], width=3),
                        ], className='justify-content-evenly'),
                    ], width=4),

                ], className='justify-content-evenly overlay'), width=12)
                ], style={'color': 'white'}),
        

        # *********** Graphs *********** #

        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(figure=scatter_fig, id='scatter-fig')
                , width=6),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(figure=hist_fig, id='hist-fig')
                        ], width=7),
                        dbc.Col([
                            dcc.Graph(figure=count_fig, id='count-fig')
                        ], width=5)
                    ]),
                    dbc.Row([
                        dcc.Graph(figure=violin_fig, id='violin-fig')
                    ]),
                ], width=6)
            ]
        , className='overlay figs'),


        # *********** Filters Offcanvas *********** #

        dbc.Button([
            html.I(className='fas fa-filter fa-lg fa-beat')
        ], id='open-filters', class_name='btn-floating btn-lg float'),

        dbc.Offcanvas([
                dbc.Row([
                    html.H6('Graphs Configurations and Filters'),
                    dbc.Form([
                        dbc.Label('Scatter Type', html_for='scatter-type'),
                        dcc.RadioItems(options={'2': '2D', '3': '3D'}, value='2', inline=True, id='scatter-type', inputStyle={'margin-right': '10px', 'margin-left': '25px'}),
                        dbc.Label('Scatter Axes', html_for='scatter-axes'),
                        dcc.Dropdown(options=column_options, multi=True, id='scatter-axes', value=list(column_options.keys())[:2]),
                        dbc.Input(id='feedback', style={'display':'none'}),
                        dbc.FormFeedback(['You must select the proper number of features!'], type='invalid'),
                        dbc.Label('Histogram Feature', html_for='hist-feat'),
                        dcc.Dropdown(options=column_options, id='hist-feat', value=list(column_options.keys())[0]),
                        dbc.Label('Count Plot Feature', html_for='count-feat'),
                        dcc.Dropdown(options=cat_cols, id='count-feat', value=list(cat_cols.keys())[0]),
                        dbc.Label('Violin Plot Feature', html_for='violin-feat'),
                        dcc.Dropdown(options=column_options, id='violin-feat', value=list(column_options.keys())[0]),
                        html.Div([
                                dbc.Button('Update Figures', id='apply-fig', class_name='me-1')
                            ], className='d-grid gap-2 col-6 mx-auto')
                    ])
                ]),
                html.Hr(),
                dbc.Row([
                    html.H6('Model Parameters'),
                    dbc.Form([
                        dbc.Label('Model Type', html_for='model-type'),
                        dcc.RadioItems(options={'logistic': 'Logistic Regression', 'knn': 'KNearest Neighbors'}, value='logistic', inline=True, id='model-type', inputStyle={'margin-right': '7px', 'margin-left': '10px'}),
                        dbc.Label('Parameter #1', id='param-1-lbl', html_for='param-1'),
                        html.Div([
                        ], id='param-1-container'),
                        dbc.Label('Parameter #2', id='param-2-lbl', html_for='param-2'),
                        html.Div([
                        ], id='param-2-container'),
                        dbc.Label('Data Normalization', html_for='normalize'),
                        dcc.Dropdown(options={'none':'None', 'minmax':'Min Max Scaling', 'std':'Standardization', 'robust':'Robust Scaling'}, value='none', id='normalize'),
                        dbc.Label('Dataset Imbalance', html_for='imbalance'),
                        dcc.Dropdown(options={'none':'None', 'over':'SMOTE Oversampling', 'under':'Undersampling'}, value='none', id='imbalance'),
                        html.Div([
                            dbc.Button('Apply Parameters', id='apply', class_name='me-1')
                        ], className='d-grid gap-2 col-6 mx-auto')
                    ])
                ])
            ],
                id='filters-canvas',
                title='Dashboard Settings',
                placement='start',
                is_open=False),
        
        ], className='container-fluid'),

        # *********** Footer *********** #

        html.Div([
            html.Small(f'© {datetime.now().year}. Developed by: '), 
            html.A('Abdelrahman Eid', href='https://www.github.com/AbdElrahman-A-Eid', target='_blank'),
            ' and ', html.A('Ahmed Azzam', href='https://www.github.com/AhmedAzzam99', target='_blank')],
            id='footer'
        )
])


##### ##### ##### ##### ##### ##### Callback Definition ##### ##### ##### ##### ##### #####


# *********** Offcanvas *********** #

@app.callback(
    Output("filters-canvas", "is_open"),
    Input("open-filters", "n_clicks"),
    Input("apply", "n_clicks"),
    Input("apply-fig", "n_clicks"),
    [State("filters-canvas", "is_open")],
)
def toggle_filter(n1, n2, n3, is_open):
    if n1 or n2 or n3:
        return not is_open
    return is_open


# *********** Model Type *********** #

@app.callback(
    Output('param-1-lbl', 'children'),
    Output('param-2-lbl', 'children'),
    Output('param-1-container', 'children'),
    Output('param-2-container', 'children'),
    Input('model-type', 'value'),
)
def change_param(model_type):
    if model_type == 'logistic':
        lbl_1, lbl_2 = 'C Parameter', 'Maximum Iterations'
        param_1 = dcc.Slider(id='param-2', min=0, max=len(c_values)-1, marks={i: str(v) for i, v in enumerate(c_values)}, step=None)
        param_2 = dcc.Slider(id='param-2', min=0, max=len(iter_values)-1, marks={i: str(v) for i, v in enumerate(iter_values)}, step=None)
    else:
        lbl_1, lbl_2 = 'No. of Neighbors', 'Weight Function'
        param_1 = dcc.Slider(id='param-2', min=0, max=len(k_values)-1, marks={i: str(v) for i, v in enumerate(k_values)}, step=None)
        param_2 = dcc.Dropdown(options={v: v.title() for v in weight_values}, value=weight_values[0], id='weight_func')
    
    return lbl_1, lbl_2, param_1, param_2


# *********** Plot Config Form Feedback *********** #

@app.callback(
    Output('feedback', 'invalid'),
    Output('apply-fig', 'disabled'),
    Input('scatter-type', 'value'),
    Input('scatter-axes', 'value')
)
def check_scatter_axes(type, axes):
    invalid = True if int(type) != len(axes) else False
    return invalid, invalid
    

# *********** Plots Configs *********** #

@app.callback(
    Output('scatter-fig', 'figure'),
    Output('hist-fig', 'figure'),
    Output('count-fig', 'figure'),
    Output('violin-fig', 'figure'),
    State('scatter-type', 'value'),
    State('scatter-axes', 'value'),
    State('hist-feat', 'value'),
    State('count-feat', 'value'),
    State('violin-feat', 'value'),
    Input('apply-fig', 'n_clicks')
)
def update_figs(type, scatter_axes, hist_feat, count_feat, violin_feat, n):
    return generate_figures(type, scatter_axes, hist_feat, count_feat, violin_feat)


# *********** Plots Configs *********** #

@app.callback(
    Output('scatter-fig', 'figure'),
    Output('f-1', 'children'),
    Output('f-2', 'children'),
    Output('f-3', 'children'),
    Output('r-1', 'children'),
    Output('r-2', 'children'),
    Output('r-3', 'children'),
    Output('p-1', 'children'),
    Output('p-2', 'children'),
    Output('p-3', 'children'),
    State('scatter-type', 'value'),
    State('scatter-axes', 'value'),
    State('model-type', 'value'),
    State('param-1', 'value'),
    State('param-2', 'value'),
    State('normalize', 'value'),
    State('imbalance', 'value'),
    Input('apply', 'n_clicks')
)
def create_model(dim, axes, model, p1, p2, norm, imb, n):
    pass

##### ##### ##### ##### ##### ##### App Runner ##### ##### ##### ##### ##### #####

if __name__ == "__main__":
    app.run_server(debug=True)