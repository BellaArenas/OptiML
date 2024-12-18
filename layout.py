from dash import dcc, html
import dash_bootstrap_components as dbc

external_stylesheets = [
    'https://fonts.googleapis.com/css?family=Open+Sans&display=swap',
    dbc.themes.LITERA
]

layout = dbc.Container(fluid=True, style={'font-family': 'Sans-serif'}, children=[
    dcc.Tabs(id="tabs-example", value='tab-1', children=[    
        dcc.Tab(label='Upload & Select Data', value='tab-1', children=[
            dbc.Row([
                dbc.Col(html.Label("Step 1: Upload  CSV file.", style={'fontWeight': 'bold'}), width=12),
                dbc.Col(dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False
                ), width=12),
                dbc.Col(html.Label("Step 2: Select the columns for analysis.", style={'fontWeight': 'bold'}), width=12),
                dbc.Col(dcc.Dropdown(id='column-dropdown', multi=True, style={'marginBottom': '20px'}), width=12),
                dbc.Col(html.Label("Step 3: Select the Identifier Column.", style={'fontWeight': 'bold'}), width=12),
                dbc.Col(dcc.Dropdown(id='identifier-dropdown', style={'marginBottom': '20px'}), width=12),
                dbc.Col(html.Div(id='conversion-div', style={'marginBottom': '20px'}), width=12),
                dbc.Col(html.Div(id='converted-data'), width=12),
                dbc.Col(dcc.Store(id='updated-data-store')), 
                dbc.Col(dcc.Store(id='log-transformed-store')),
                dbc.Col(dcc.Store(id='conversion-selections-store')),  # Store to hold conversion selections
            ]),
        ]),
        dcc.Tab(label='Scatter & Heatmap Plot', value='tab-2', children=[
            dbc.Row([
                dbc.Col(html.Label("Create a scatter plot by selecting variables for the X and Y axes.", style={'fontWeight': 'bold'}), width=12),
                dbc.Col([
                    html.Label('Select X-axis:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='x-axis-dropdown', options=[], style={'marginBottom': '20px'})
                ], width=3),
                dbc.Col([
                    html.Label('Select Y-axis:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='y-axis-dropdown', options=[], style={'marginBottom': '20px'})
                ], width=3),
                dbc.Col([
                    html.Label('Select Correlation Type for Heatmap:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='correlation-type-dropdown',
                        options=[
                            {'label': 'Pearson', 'value': 'pearson'},
                            {'label': 'Spearman', 'value': 'spearman'}
                        ],
                        value='pearson',  
                        style={'marginBottom': '20px'}
                    )
                ], width=6),
                dbc.Row([
                    dbc.Col(html.Div(id='scatter-plot-content'), width=6),
                    dbc.Col(html.Div(id='heatmap-content'), width=6),
                ]),
            ])
        ]),
        dcc.Tab(label='Machine Learning', value='tab-3', children=[
            dbc.Row([
                # Row for selecting algorithm type and specific algorithm
                dbc.Col([
                    html.Label('Select Algorithm Type:', style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='algorithm-type-dropdown',
                        options=[
                            {'label': 'Regression', 'value': 'Regression'},
                            # {'label': 'Classification', 'value': 'Classification'},
                            # {'label': 'Neural Network', 'value': 'Neural Network'}
                        ],
                        value='Regression'
                    )
                ], width=6),
                dbc.Col([
                    html.Label('Select Specific Algorithm:', style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                    dcc.Dropdown(id='algorithm-dropdown', options=[])
                ], width=6),
            ]),
            dbc.Row([
                # Column for selecting target variable
                dbc.Col([
                    html.Label('Select Target Variable:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='target-variable-dropdown', multi=False, options=[])
                ], width=6, className="mb-3"),

                # Column for selecting metric
                dbc.Col([
                    html.Label('Select Metric:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='metric-dropdown', options=[], value=None)
                ], width=6, className="mb-3"),
            ]), 
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        'Run Simulation',
                        id='run-simulation-button',
                        n_clicks=0,
                        color="primary",
                        className="me-1"
                    ),
                ], width=12, className="mb-3"),
            ]), 
            dbc.Row([
                # Area for displaying best model information
                dbc.Col(html.Div(id='best-model-info', style={'margin-top': '20px', 'fontWeight': 'bold', 'fontSize': '16px'})),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='cross-val-predict-plot'), width=6),
                dbc.Col(dcc.Graph(id='cross-val-residuals-plot'), width=6),
            ])
        ]),
        dcc.Tab(label='SHAP', value='tab-shap', children=[
            dbc.Row([
                    dbc.Col(html.Label('SHAP Values help to explain the black box models. The larger the magnitude, the more imporant that feature is. They all show impact on target prediction',
                                className="text-center"), width=12)
                ]),
            dbc.Row([
                dbc.Col([html.Label('Select Global SHAP Plot:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='global-shap-plot-dropdown',
                        options=[
                            {'label': 'Beeswarm', 'value': 'beeswarm'},
                            {'label': 'Heatmap', 'value': 'heatmap'},
                            {'label': 'Bar Plot', 'value': 'bar'}
                        ],
                        value='beeswarm'
                    ),
                    dcc.Graph(id='global-shap-plot')
                ], width=6),
                dbc.Col([
                    html.Label('Select LSN for Waterfall Plot:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='lsn-dropdown',
                        options=[], 
                        value=None 
                    ),
                    dcc.Graph(id='waterfall-plot')
                ], width=6),
            ])
        ]),
        dcc.Tab(label='Partial Dependence', value='tab-partial-dependence', children=[
            dbc.Row([
                    dbc.Col(html.Label('Partial Dependence Plots show interactions between features and then the effects of the target prediction. For linear models, when you see diagonal lines that shows interaction. Straight lines show dominance.',
                                className="text-center"), width=12)
                ]),
                dbc.Row([
                dbc.Col(
                    html.Label('Select Features for PDP Interact:', style={'fontWeight': 'bold'}), width=12),
                dbc.Col([
                    html.Label('Select X-axis Feature:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='pdp-interact-x-dropdown', options=[], style={'marginBottom': '20px'}),
                ], width=3),
                dbc.Col([
                    html.Label('Select Y-axis Feature:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='pdp-interact-y-dropdown', options=[], style={'marginBottom': '20px'}),
                ], width=3),
                dbc.Col([
                    html.Label('Select Feature for PDP:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='pdp-feature-dropdown', options=[], style={'marginBottom': '20px'}),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='pdp-interact-plot'), width=6),
                dbc.Col(dcc.Graph(id='single-feature-pdp-plot'), width=6),
            ])
        ]),
        dcc.Tab(label='Feature Optimization', value='tab-feature-optimization', children=[
            html.Div([
                html.Label('Desired Target Value:'),
                dcc.Input(id='desired-target-value', type='number'),

                # Dynamically generate input fields for each feature
                html.Div(id='feature-inputs'),

                html.Button('Calculate', id='calculate-optimization'),
                html.Div(id='optimization-result')
            ])
        ]),
    ]),
    dbc.Alert(id="alert-auto", is_open=False, duration=4000),
    html.Div(id='tab-content'),
    html.Div(id='placeholder-output', style={'display': 'none'}),
    html.Div(id='completion-signal', style={'display': 'none'}),


    # Place the Store component here
    dcc.Store(id='store-for-figures'),

])