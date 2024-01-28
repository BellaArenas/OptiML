import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import urllib.parse
import pandas as pd
import base64
import io
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, RepeatedKFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_predict, cross_val_score
import plotly.graph_objects as go
import io
import contextlib
import sys
from threading import Thread
import queue
import shap
import pdpbox
from pdpbox import pdp, info_plots
from algorithms import regression_algorithms, classification_algorithms, neural_network_algorithms, algorithm_options, regression_dropdown_options, classification_dropdown_options , neural_network_dropdown_options, regression_metrics, classification_metrics
from rdkit import Chem
import plotly.graph_objs as go
import plotly.tools as tls



# Define  external stylesheets
external_stylesheets = [
    'https://fonts.googleapis.com/css?family=Open+Sans&display=swap',
    dbc.themes.LITERA
]

# Initialize  app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#app.suppress_callback_exceptions = True  # Suppress callback exceptions

# Define  constants
unit_prefixes = {
    'base': 1,
    'deci': 1e-1,
    'centi': 1e-2,
    'milli': 1e-3,
    'micro': 1e-6,
    'nano': 1e-9,
    'pico': 1e-12,
}

transformations = [
    'None',
    'Log10'
]

app.layout = dbc.Container(fluid=True, style={'font-family': 'Sans-serif'}, children=[
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

# Global variable to store column types
global_column_types = {}

def classify_column_type(column, column_name):
    # Check for SMILES based on column name
    if "smiles" in column_name.lower():
        return "SMILES"

    # Check dtype for numerical data
    if pd.api.types.is_numeric_dtype(column):
        return "Numerical"

    # Check dtype for string data which could be categorical or SMILES
    if pd.api.types.is_string_dtype(column):
        # Sample a few values for SMILES validation
        sampled_values = column.dropna().sample(n=min(10, len(column)), random_state=42)
        for value in sampled_values:
            if not Chem.MolFromSmiles(str(value)):
                return "Categorical"  # If any value is not a valid SMILES, classify as categorical
        return "SMILES"  # If all sampled values are valid SMILES
 
    return "Categorical"  # Default to categorical if none of the above conditions are met

@app.callback(
    [Output('column-dropdown', 'options'),
     Output('identifier-dropdown', 'options')],
    [Input('upload-data', 'contents')]
)
def update_dropdowns(contents):
    if contents is None:
        return [], []

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    df = df[[col for col in df.columns if ".Q" not in col]]

     # Classify and store column types
    global global_column_types
    global_column_types = {col: classify_column_type(df[col], col) for col in df.columns}

    options = [{'label': col, 'value': col} for col in df.columns]
    return options, options

@app.callback(
    Output('conversion-div', 'children'),
    [Input('column-dropdown', 'value')],
    [State('conversion-selections-store', 'data')]
)
def update_conversion_div(selected_columns, stored_selections):
    if not selected_columns:
        return None

    if stored_selections is None:
        stored_selections = {}

    conversion_ui_elements = []
    for col in selected_columns:
        column_type = global_column_types.get(col, "Numerical")
        ui_elements = generate_conversion_ui(col, column_type, stored_selections.get(col, {}))
        conversion_ui_elements.append(ui_elements)

    return html.Div(conversion_ui_elements)

def generate_conversion_ui(col, column_type, previous_selections):
    component_container_style = {'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '1%'}

    # Common UI elements (e.g., rename input)
    rename_ui = html.Div([
        html.Div([  # Wrapper for vertical centering
            html.Label(f"Rename Column:", style={'fontWeight': 'bold'}),
            dcc.Input(
                id={'type': 'rename-input', 'index': col},
                value=previous_selections.get('new_col_name', col),
                type='text',
                placeholder='Rename column',
                style={'width': '100%'}
            )
        ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center'}),  # Flexbox for vertical centering
    ], style=component_container_style)

    specific_ui_elements = [rename_ui]

    if column_type == "Numerical":
        # Add numerical UI elements (e.g., unit conversion, transform)
        specific_ui_elements.extend([
            # From Unit
            html.Div([
                html.Label("From Unit:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id={'type': 'from-dropdown', 'index': col},
                    options=[{'label': prefix, 'value': factor} for prefix, factor in unit_prefixes.items()],
                    value=previous_selections.get('from_factor', 1),
                )
            ], style=component_container_style),
            # To Unit
            html.Div([
                html.Label("To Unit:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id={'type': 'to-dropdown', 'index': col},
                    options=[{'label': prefix, 'value': factor} for prefix, factor in unit_prefixes.items()],
                    value=previous_selections.get('to_factor', 1),
                )
            ], style=component_container_style),
            # Transform
            html.Div([
                html.Label("Transform:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id={'type': 'transform-dropdown', 'index': col},
                    options=[{'label': transform, 'value': transform} for transform in transformations],
                    value=previous_selections.get('transform', 'none'),
                )
            ], style=component_container_style)
        ])

    # elif column_type == "SMILES":
    #     # Add SMILES specific UI element
    #     specific_ui_elements.append(html.Div([
    #         html.Label("Encoding Type:", style={'fontWeight': 'bold'}),
    #         dcc.Dropdown(
    #             id={'type': 'smiles-encoding-dropdown', 'index': col},
    #             options=[
    #                 {'label': 'Morgan Fingerprint 1024 bits', 'value': 'morgan_1024'},
    #                 {'label': 'Morgan Fingerprint 2048 bits', 'value': 'morgan_2048'},
    #                 {'label': 'Topological Fingerprint', 'value': 'topological'},
    #                 {'label': 'MACCS Keys', 'value': 'maccs'}
    #             ],
    #             value=previous_selections.get('smiles_encoding', 'morgan_1024'),
    #         )
    #     ], style=component_container_style))

    # elif column_type == "Categorical":
    #     # Add Categorical specific UI element
    #     specific_ui_elements.append(html.Div([
    #         html.Label("Encoding Method:", style={'fontWeight': 'bold'}),
    #         dcc.Dropdown(
    #             id={'type': 'categorical-encoding-dropdown', 'index': col},
    #             options=[
    #                 {'label': 'One-Hot Encoding', 'value': 'one_hot'},
    #                 {'label': 'Label Encoding', 'value': 'label'},
    #                 {'label': 'Frequency Encoding', 'value': 'frequency'},
    #                 {'label': 'Binary Encoding', 'value': 'binary'}
    #             ],
    #             value=previous_selections.get('encoding_method', 'one_hot'),
    #         )
    #     ], style=component_container_style))

    elif column_type in ["SMILES", "Categorical"]:
        # Add a message for non-supported column types
        message_ui = html.Div([
            html.Label("Encoding Type:", style={'fontWeight': 'bold'}),
            html.Div([
                "Currently, encoding for SMILES and categorical data is not supported. ",
                "Please limit your data to numerical values for optimal functionality."
            ], style={'color': 'red', 'marginTop': '10px'})
        ], style=component_container_style)
        specific_ui_elements.append(message_ui)

    # Combine all UI elements
    ui_elements = [html.H5(col, style={'marginBottom': '5px'})] + specific_ui_elements

    return html.Div(ui_elements, style={'marginBottom': '20px'})


@app.callback(
    Output('converted-data', 'children'),
    Input('updated-data-store', 'data')
)
def update_converted_data(data):
    if not data:
        return html.Div()
    df = pd.DataFrame(data=data['data'], index=data['index'], columns=data['columns'])
    data_length = len(df)
    csv_string = df.to_csv(index=True, encoding='utf-8')
    csv_data = f"data:text/csv;charset=utf-8,{csv_string}"

    return html.Div([
        html.H5(f'Converted Data (Number of Data Points: {data_length}):'),
        dcc.Graph(figure={
            'data': [{
                'type': 'table',
                'header': {
                    'values': df.columns.tolist()
                },
                'cells': {
                    'values': df.values.T.tolist()
                }
            }]
        }),
        html.A(
            dbc.Button("Download CSV", color="primary", className="me-1"),
            id='download-link',
            download="converted_data.csv",
            href=csv_data,
            target="_blank"
        )
    ])

@app.callback(
    [Output('updated-data-store', 'data'),
     Output('log-transformed-store', 'data')],   
    [Input({'type': 'from-dropdown', 'index': ALL}, 'value'),
     Input({'type': 'to-dropdown', 'index': ALL}, 'value'),
     Input({'type': 'rename-input', 'index': ALL}, 'value'),
     Input({'type': 'transform-dropdown', 'index': ALL}, 'value'),
     Input('column-dropdown', 'value'),
     Input('upload-data', 'contents'),
     Input('identifier-dropdown', 'value')]
)
def update_data_store(from_factors, to_factors, rename_columns, transform_values, selected_columns, contents, identifier_column):
    if contents is None or not selected_columns:
        return dash.no_update

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    if identifier_column and identifier_column in df.columns:
        df.set_index(identifier_column, inplace=True)

    df = df[selected_columns]
    log_transformed_columns = {}


    for col, from_factor, to_factor, new_col_name, transform_value in zip(selected_columns, from_factors, to_factors, rename_columns, transform_values):
        conversion_factor = from_factor / to_factor
        df[col] = df[col] * conversion_factor  # Update values in the original column
        if transform_value == 'Log10':
            df[col] = np.log10(df[col])
            log_transformed_columns[col] = True
        else:
            log_transformed_columns[col] = False
  # Apply transformation to the original column
        df[col] = df[col].round(4)  # Round values in the original column

        if col != new_col_name:
            df.rename(columns={col: new_col_name}, inplace=True)  # Rename the column if a new name is provided

    df.dropna(inplace=True)
    df_dict = df.to_dict(orient='split')
    return df_dict, log_transformed_columns

@app.callback(
    [Output('x-axis-dropdown', 'options'), Output('y-axis-dropdown', 'options')],
    Input('updated-data-store', 'data')
)
def update_axis_dropdowns(data):
    if not data:
        return [], []
    df = pd.DataFrame(data=data['data'], index=data['index'], columns=data['columns'])
    options = [{'label': col, 'value': col} for col in df.columns]
    return options, options

@app.callback(
    Output('scatter-plot-content', 'children'),
    [Input('x-axis-dropdown', 'value'), Input('y-axis-dropdown', 'value'),
     Input('updated-data-store', 'data'),
     Input('identifier-dropdown', 'value')]
)
def update_scatterplot(x_axis, y_axis, data, identifier_column):
    if not x_axis or not y_axis or not data:
        return dash.no_update

    df = pd.DataFrame(data=data['data'], index=data['index'], columns=data['columns'])

    # Decide what to use for hover data
    if identifier_column and identifier_column in df.columns:
        hover_data = [identifier_column]  # Ensure this is a column name
    else:
        df['index_column'] = df.index  # Add index as a column
        hover_data = ['index_column']  # Use 'index_column' for hover data

    fig = px.scatter(df, x=x_axis, y=y_axis, hover_data=hover_data)

    # Adding trendline, R2, MSE, and MAE
    trendline = np.polyfit(df[x_axis], df[y_axis], 1)
    trendline_eq = f'y = {trendline[0]:.2f}x + {trendline[1]:.2f}'
    r2 = r2_score(df[y_axis], trendline[0] * df[x_axis] + trendline[1])
    mse = mean_squared_error(df[y_axis], trendline[0] * df[x_axis] + trendline[1])
    mae = mean_absolute_error(df[y_axis], trendline[0] * df[x_axis] + trendline[1])
    metrics_text = f'R²: {r2:.2f}<br>MSE: {mse:.2f}<br>MAE: {mae:.2f}'

    fig.add_trace(go.Scatter(
        x=df[x_axis],
        y=trendline[0] * df[x_axis] + trendline[1],
        mode='lines',
        name='Trendline',
        line=dict(color='red')
    ))
    fig.add_annotation(
        text=metrics_text, xref='paper', yref='paper',x=0.05,y=0.95,showarrow=False,bordercolor='black',borderwidth=1,bgcolor='white'
    )
    # Adjust  layout and margins
    fig.update_layout(
        autosize=True,  # Ensure the figure adjusts to the size of its container
        legend=dict(orientation='h', yanchor='bottom', xanchor='right', y=1.02, x=1),
        hovermode='closest',
        xaxis=dict(
            constrain='domain',  # This keeps the x-axis within the plotting domain
            # If   want to set a specific range for x-axis,   can use xaxis=dict(range=[min_val, max_val])
        ),
        yaxis=dict(
            scaleanchor='x',  # This forces the y-axis to scale with the x-axis
            scaleratio=1,  # This sets a 1:1 aspect ratio
            # If   want to set a specific range for y-axis,   can use yaxis=dict(range=[min_val, max_val])
        ),
        # If   need the plot to span the entire width (like in a Dash grid), this might help:
        # width=None, height=None
    )
    fig.update_traces(hoverinfo='all')
    return dcc.Graph(figure=fig)

@app.callback(
    Output('heatmap-content', 'children'),
    [Input('correlation-type-dropdown', 'value'), Input('updated-data-store', 'data')]
)
def update_heatmap(correlation_type, data):
    if not data:
        return dash.no_update

    df = pd.DataFrame(data=data['data'], columns=data['columns'])

    # Calculate the correlation matrix
    if correlation_type == 'pearson':
        corr_matrix = df.corr(method='pearson')
    else:
        corr_matrix = df.corr(method='spearman')

    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", labels=dict(color="Correlation"))
    fig.update_layout(
        autosize=True,  # Ensure the figure adjusts to the size of its container
        legend=dict(orientation='h', yanchor='bottom', xanchor='right', y=1.02, x=1),
        hovermode='closest',
        xaxis=dict(
            constrain='domain',  # This keeps the x-axis within the plotting domain
            # If   want to set a specific range for x-axis,   can use xaxis=dict(range=[min_val, max_val])
        ),
        yaxis=dict(
            scaleanchor='x',  # This forces the y-axis to scale with the x-axis
            scaleratio=1,  # This sets a 1:1 aspect ratio
            # If   want to set a specific range for y-axis,   can use yaxis=dict(range=[min_val, max_val])
        ),
        # If   need the plot to span the entire width (like in a Dash grid), this might help:
        # width=None, height=None
    )

    return dcc.Graph(figure=fig)

@app.callback(
    Output('algorithm-dropdown', 'options'),
    [Input('algorithm-type-dropdown', 'value')]
)
def update_algorithm_options_by_type(algorithm_type):
    if algorithm_type == 'Regression':
        return regression_dropdown_options
    elif algorithm_type == 'Classification':
        return classification_dropdown_options  
    elif algorithm_type == 'Neural Network':
        return neural_network_dropdown_options  
    else:
        return []

@app.callback(
    Output('metric-dropdown', 'options'),
    [Input('algorithm-type-dropdown', 'value')]
)
def set_metric_options(algorithm_type):
    if algorithm_type == 'Regression':
        return regression_metrics
    elif algorithm_type == 'Classification':
        return classification_metrics
    # Add more conditions for other algorithm types if necessary
    else:
        return []
        
@app.callback(
    Output('target-variable-dropdown', 'options'),
    Input('updated-data-store', 'data')
)
def update_target_variable_dropdown(data):
    if not data:
        return []
    df = pd.DataFrame(data=data['data'], index=data['index'], columns=data['columns'])
    options = [{'label': col, 'value': col} for col in df.columns]
    return options

def preprocess_data(df):
    # Convert string values to lowercase
    df = df.applymap(lambda s: s.lower() if isinstance(s, str) else s)
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Encoding categorical variables
    cat_cols = df.select_dtypes(include=['object']).columns
    
    if not cat_cols.empty:  # Check if there are any categorical columns to encode
        encoder = OneHotEncoder(drop='first', sparse=False)
        df_encoded = pd.DataFrame(encoder.fit_transform(df[cat_cols]), columns=encoder.get_feature_names(cat_cols))
        df = df.drop(cat_cols, axis=1)
        df = pd.concat([df, df_encoded], axis=1)
    
    return df

# Split the data
def split_data(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def recreate_model_with_best_params(best_params, algorithm_type, selected_algorithm):
    if algorithm_type == 'Regression':
        model_class = regression_algorithms[selected_algorithm]['model'].__class__
    elif algorithm_type == 'Classification':
        model_class = classification_algorithms[selected_algorithm]['model'].__class__
    else:  # Assuming Neural Network or other types
        model_class = neural_network_algorithms[selected_algorithm]['model'].__class__
    
    # Create a new instance of the model class with best parameters
    model = model_class(**best_params)
    return model

def create_prediction_plot(y, y_pred_cv, index, identifier_column):
    df = pd.DataFrame({'Actual': y, 'Predicted': y_pred_cv, identifier_column: index})
    fig = px.scatter(df, x='Actual', y='Predicted', hover_data=[identifier_column])
    fig.update_layout(title="Predicted vs Actual Values", xaxis_title="Actual", yaxis_title="Predicted", autosize=True,
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
        )
    )
    return fig

def create_residuals_plot(y_pred_cv, residuals, index, identifier_column):
    df = pd.DataFrame({'Predicted': y_pred_cv, 'Residuals': residuals, identifier_column: index})
    fig = px.scatter(df, x='Predicted', y='Residuals', hover_data=[identifier_column])

    # Updating  layout to match the style of create_prediction_plot
    fig.update_layout(  title="Residuals Plot", xaxis_title="Predicted", yaxis_title="Residuals", autosize=True,
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
        )
    )

    return fig

# Set up K-Fold cross-validation
rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

global trained_model 
trained_model = None

@app.callback(
    [Output('store-for-figures', 'data'),
     Output('best-model-info', 'children'),
     Output('completion-signal', 'children')],    
     [Input('run-simulation-button', 'n_clicks')],
    [State('algorithm-type-dropdown', 'value'),
     State('algorithm-dropdown', 'value'),
     State('target-variable-dropdown', 'value'),
     State('metric-dropdown', 'value'),
     State('updated-data-store', 'data')]
)
def run_ml_simulation(n_clicks, algorithm_type, selected_algorithm, target_variable, selected_metric, data):
    global trained_model
    if n_clicks == 0 or not data:
        return dash.no_update, '', dash.no_update
       
    # Convert data from JSON format to DataFrame
    df = pd.DataFrame(data=data['data'], index=data['index'], columns=data['columns'])
    df = preprocess_data(df)
    X = df.drop(target_variable, axis=1)
    y = df[target_variable].squeeze()

    best_score = -float('inf')
    best_model_info = ''
    best_y_pred_cv = None
    n_iter = 20 # Number of iterations

    print("Before training, Data shape:", df.shape)
    print("Model being trained:", selected_algorithm)

    scoring_metrics = {
        'Regression': ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
        'Classification': ['accuracy', 'precision_macro', 'recall_macro']
    }

    if selected_algorithm == 'optiML_all_regression':

        for alg_name, alg_details in regression_algorithms.items():
            model = alg_details['model']
            param_grid = alg_details['params']

            # Use RandomizedSearchCV
            grid_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n_iter,
                                              scoring=selected_metric, cv=kf, verbose=1, n_jobs=-1)
            grid_search.fit(X, y)

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model_info = f"Best Model: {alg_name}, Best Params: {grid_search.best_params_}, Best Score: {best_score:.4f}"
                best_y_pred_cv = cross_val_predict(grid_search.best_estimator_, X, y, cv=kf)
                trained_model = grid_search.best_estimator_
                print(f"Trained model for {alg_name}: {trained_model}")
    else:
        # Handle individual model
        model = None
        param_grid = {}
        if algorithm_type == 'Regression':
            model = regression_algorithms[selected_algorithm]['model']
            param_grid = regression_algorithms[selected_algorithm]['params']
        elif algorithm_type == 'Classification':
            model = classification_algorithms[selected_algorithm]['model']
            param_grid = classification_algorithms[selected_algorithm]['params']
        # Add more elif blocks here for other types like Neural Network, etc.

        # Initialize and run RandomizedSearchCV
        grid_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n_iter,
                                              scoring=selected_metric, cv=kf, verbose=1, n_jobs=-1)
        grid_search.fit(X, y)

        # Update best model information
        best_score = grid_search.best_score_
        trained_model = grid_search.best_estimator_
        best_model_info = f"Current Model: {selected_algorithm}, Best Params: {grid_search.best_params_}, Best Score: {best_score:.4f}"
        best_y_pred_cv = cross_val_predict(grid_search.best_estimator_, X, y, cv=kf)

    print(grid_search.cv_results_.keys())

    if best_y_pred_cv.ndim > 1:
        best_y_pred_cv = best_y_pred_cv.flatten()

    cv_results = grid_search.cv_results_

    scores = {}

    for metric in scoring_metrics[algorithm_type]:
        cv_scores = cross_val_score(trained_model, X, y, cv=kf, scoring=metric)
        scores[metric] = np.mean(cv_scores)

    for metric, score in scores.items():
        best_model_info += f"{metric}: {score:.3f}\n"
    # Prepare data for store and return
    y_pred_cv_to_store = pd.Series(best_y_pred_cv)
    store_data = {
        'y': y.tolist(),
        'y_pred_cv': y_pred_cv_to_store.tolist(),
        'index': df.index.tolist()  # Storing the index separately
    }

    print("Trained model:", trained_model)

    print("Model trained. Best score:", best_score)
    
    print("Best model info:", best_model_info)

    print("Store data keys:", store_data.keys())
    print("Sample of stored predictions:", y_pred_cv_to_store[:5])


    return store_data, best_model_info, 'Completed'

@app.callback(
    [Output('cross-val-predict-plot', 'figure'), Output('cross-val-residuals-plot', 'figure')],
    [Input('store-for-figures', 'data')],
    [State('identifier-dropdown', 'value'),
     State('target-variable-dropdown', 'value')]
)
def create_ml_plots(stored_data, identifier_column, target_variable):
    if stored_data is None:
        return dash.no_update, dash.no_update

    y = np.array(stored_data['y'])
    y_pred_cv = np.array(stored_data['y_pred_cv'])
    index = np.array(stored_data['index']) 

    if len(y_pred_cv.shape) > 1:
        y_pred_cv = y_pred_cv.flatten()

    residuals = y - y_pred_cv

    # Create prediction plot
    fig1 = create_prediction_plot(y, y_pred_cv, index, identifier_column)
    # Create residuals plot
    fig2 = create_residuals_plot(y_pred_cv, residuals, index, identifier_column)

    return fig1, fig2

def map_value_to_color(value, min_val, max_val, colorscale):
    # Normalize the value to a range between 0 and 1
    normalized_value = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
    return px.colors.sample_colorscale(colorscale, normalized_value)[0]    

def create_shap_beeswarm_plot(shap_values, features, feature_values, index):
    # Prepare the SHAP and feature values dataframes
    shap_df = pd.DataFrame(shap_values.values, columns=features, index=index)
    feature_df = pd.DataFrame(feature_values, columns=features, index=index)

    # Melt the dataframes to long format and reset the index
    shap_df_long = shap_df.melt(ignore_index=False, var_name='Feature', value_name='SHAP').reset_index().rename(columns={'index': 'Identifier'})
    feature_df_long = feature_df.melt(ignore_index=False, var_name='Feature', value_name='FeatureValue').reset_index().rename(columns={'index': 'Identifier'})

    # Merge the two dataframes
    merged_df = pd.merge(shap_df_long, feature_df_long, on=['Identifier', 'Feature'])

    # Apply color mapping for each feature
    for feature in features:
        min_val = feature_df[feature].min()
        max_val = feature_df[feature].max()
        mask = merged_df['Feature'] == feature
        merged_df.loc[mask, 'Color'] = merged_df[mask]['FeatureValue'].apply(
            lambda x: map_value_to_color(x, min_val, max_val, 'bluered')
        )

    fig = go.Figure()

    # Add traces for each feature
    for feature in features:
        df_subset = merged_df[merged_df['Feature'] == feature]
        hover_text = df_subset['Identifier'].astype(str) + '<br>SHAP: ' + df_subset['SHAP'].round(4).astype(str)
        fig.add_trace(go.Scatter(
            x=df_subset['SHAP'],
            y=df_subset['Feature'],
            mode='markers',
            marker=dict(color=df_subset['Color'], size=5),
            name=feature,
            text=hover_text,
            hoverinfo='text'
        ))

    # Adding a dummy trace for the color bar
    color_min = feature_df.values.min()
    color_max = feature_df.values.max()
    color_range = np.linspace(color_min, color_max, 10)
    fig.add_trace(go.Scatter(
        x=[None] * len(color_range),
        y=[None] * len(color_range),
        mode='markers',
        marker=dict(
            size=10,
            color=color_range,
            colorbar=dict(
                title="Feature Value",
                titleside="right",
                tickvals=[]
            ),
            colorscale='bluered',
            showscale=True
        ),
        hoverinfo='none'
    ))

    fig.update_layout(
        xaxis=dict(title='SHAP value'),
        yaxis=dict(title='Feature'),
        showlegend=False  # Disable the legend
    )

    return fig

def create_shap_bar_plot(shap_values):
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = shap_values.feature_names
    print(mean_abs_shap)

    fig = go.Figure([go.Bar(
        x=mean_abs_shap,
        y=feature_names,
        orientation='h'
    )])

    fig.update_layout(
        title="Mean Absolute SHAP Values",
        xaxis_title="Average Impact on Model Output Magnitude",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed")
    )

    return fig

def create_shap_heatmap(shap_values, index):
    # Assuming shap_values is a SHAP values object with .values and .feature_names
    shap_df = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)

      # Normalize the SHAP values to the range [0, 1]
    shap_min = shap_df.values.min()
    shap_max = shap_df.values.max()
    normalized_shap = (shap_df.values - shap_min) / (shap_max - shap_min)

    # Determine where 0 falls in the normalized range
    zero_position = (0 - shap_min) / (shap_max - shap_min)

    # Define a custom colorscale
    colorscale = [
        [0, 'blue'],       # Blue for smaller SHAP values
        [zero_position, 'white'],              # White at the position of 0
        [1, 'red']         # Red for larger SHAP values
    ]

    # Create the heatmap using go.Heatmap
    heatmap_trace = go.Heatmap(
        z=shap_df.values,
        x=shap_df.columns,
        y=list(range(len(index))), 
        hoverinfo='x+y+z',  # Specify hoverinfo as desired (x=feature, y=sample, z=SHAP Value)
        colorscale=colorscale  # Use the custom colorscale
    )

    # Create a figure and add the heatmap trace
    fig = go.Figure(data=[heatmap_trace])

    # Customize the  layout if needed
    fig.update_layout(
        title="SHAP Heatmap",
        xaxis_title="Features",
        yaxis_title="Samples"
    )

    return fig

global baseline

@app.callback(
    Output('global-shap-plot', 'figure'),
    [Input('global-shap-plot-dropdown', 'value'),
     Input('store-for-figures', 'data')],  # Triggered by either dropdown change or data store update
    [State('updated-data-store', 'data'),
     State('target-variable-dropdown', 'value')]
)
def update_shap_plot(plot_type, stored_data, data, target_variable):
    global trained_model
    global shap_values, explainer 

    print('SHAP Callback - Data received:', data is not None)
    print('SHAP Callback - Stored Data received:', stored_data is not None)

    if trained_model is not None:
        print('SHAP Callback - Generating SHAP values...')
    else:
        print('SHAP Callback - Trained model not available.')

    # Check if the trained model is available
    if trained_model is None:
        return go.Figure()

    # Handle the case when data is not available
    if not data or not stored_data:
        return go.Figure()

    df = pd.DataFrame(data['data'], columns=data['columns'])
    df.set_index(pd.Index(data['index']), inplace=True)

    print('df:', df)

    # Drop the target variable from the DataFrame
    X = df.drop(target_variable, axis=1)

    
    # # # Now,   can use model_predict as the callable model
    # explainer = shap.Explainer(trained_model, X)
    # shap_values = explainer(X)

    # Create the SHAP explainer and calculate SHAP values
    explainer = shap.explainers.Exact(trained_model.predict, X)
    shap_values = explainer(X)

    # explainer = shap.Explainer(trained_model, X)
    # shap_values = explainer(X)

    print(np.abs(shap_values.values).mean(axis=0))

    # Assume features and index are already defined or obtained from df_for_shap
    features = X.columns
    index = np.array(stored_data['index']) 

    # Obtain feature values (assuming df_for_shap is used)
    feature_values = X.values

    if plot_type == 'beeswarm':
        # Create the beeswarm plot
        fig = create_shap_beeswarm_plot(shap_values, features, feature_values, df.index)
        return fig
    if plot_type == 'beeswarm':
        fig = create_shap_beeswarm_plot(shap_values, features, X.values, df.index)
    elif plot_type == 'heatmap':
        fig = create_shap_heatmap(shap_values, df.index)
    elif plot_type == 'bar':
        fig = create_shap_bar_plot(shap_values)
    else:
        print("Shap Selection Error")
        return go.Figure()
    return fig

@app.callback(
    Output('lsn-dropdown', 'options'),
    [Input('updated-data-store', 'data')],
    [State('identifier-dropdown', 'value')]
)
def set_lsn_dropdown_options(data, identifier_column):
    if not data:
        return []

    # Convert the data to a DataFrame
    df = pd.DataFrame(data['data'], columns=data['columns'])

    # Check if the identifier column is in the DataFrame
    if identifier_column and identifier_column in df.columns:
        # Use unique values from the identifier column
        options = [{'label': str(id_val), 'value': id_val} for id_val in df[identifier_column].unique()]
    else:
        # If no identifier column, use the DataFrame's index
        df.set_index(pd.Index(data['index']), inplace=True)
        options = [{'label': str(idx), 'value': idx} for idx in df.index]

    return options

def create_plotly_waterfall(shap_values, baseline, features, feature_names):
    shap_values = shap_values.values

    # Convert to numpy array if it's a pandas Series
    shap_values = np.array(shap_values)

    # Sort the features by their SHAP values, negative values first then positive
    order = np.argsort(shap_values)
    sorted_shap_values = shap_values[order]
    sorted_feature_names = [feature_names[i] for i in order]

    # All bars will start from 0
    base_values = np.zeros_like(sorted_shap_values)

    # Determine the color of the bars based on the SHAP value sign
    bar_colors = ['blue' if val < 0 else 'red' for val in sorted_shap_values]

    # Create the bar chart
    fig = go.Figure(data=[go.Bar(
        x=sorted_feature_names,
        y=sorted_shap_values,
        base=base_values,  # Set the base of the bars to 0
        marker_color=bar_colors  # Set the color of the bars
    )])

    # Update  layout for the figure
    fig.update_layout(
        title='SHAP Waterfall Plot',
        xaxis_title='Features',
        yaxis_title='SHAP Value',
        showlegend=False
    )

    return fig

@app.callback(
    Output('waterfall-plot', 'figure'),
    [Input('lsn-dropdown', 'value')],
    [State('updated-data-store', 'data'),
     State('target-variable-dropdown', 'value')]

)
def update_waterfall_plot(selected_id, data, target_variable):
    global shap_values, explainer
    if not data or selected_id is None:
        return go.Figure()

    df = pd.DataFrame(data=data['data'], columns=data['columns'])
    df.set_index(pd.Index(data['index']), inplace=True)

    # # Check if target_variable is a list and use the first element if so
    # if isinstance(target_variable, list) and target_variable:
    #     target_variable = target_variable[0]

    if target_variable in df.columns:
        df.drop(target_variable, axis=1)

    feature_names = df.columns

    if selected_id in df.index:
        selected_shap_values = shap_values[df.index.get_loc(selected_id)]

        if isinstance(selected_shap_values, list) or len(selected_shap_values.shape) > 1:
            selected_shap_values = selected_shap_values[0]

        feature_values = df.loc[selected_id]

        fig = create_plotly_waterfall(selected_shap_values, 0, feature_values, feature_names)
        return fig
    else:
        return go.Figure()

### PDP BOX ###
@app.callback(
    [Output('pdp-interact-x-dropdown', 'options'),
     Output('pdp-interact-y-dropdown', 'options'),
     Output('pdp-feature-dropdown', 'options')],
    [Input('store-for-figures', 'data')],
    [State('updated-data-store', 'data'),
     State('target-variable-dropdown', 'value')]
)
def update_pdp_dropdowns(stored_data, updated_data, target_variable):
    if stored_data is None or updated_data is None:
        return [], [], []

    # Convert updated_data to DataFrame
    df = pd.DataFrame(updated_data['data'], columns=updated_data['columns'])

    # Exclude target variable(s) from the DataFrame
    df = df.drop(target_variable, axis=1)

    options = [{'label': col, 'value': col} for col in df.columns]

    return options, options, options

@app.callback(
    Output('pdp-interact-plot', 'figure'),
    [Input('pdp-interact-x-dropdown', 'value'),
     Input('pdp-interact-y-dropdown', 'value')],
    [State('updated-data-store', 'data'),
     State('target-variable-dropdown', 'value')]
)
def update_pdp_interact_plot(x_feature, y_feature, updated_data, target_variable):
    try:
        if not x_feature or not y_feature or not updated_data or 'data' not in updated_data or 'columns' not in updated_data:
            return go.Figure()

        df = pd.DataFrame(updated_data['data'], columns=updated_data['columns'])

        print('PDP Callback - Data received:', updated_data is not None)
        print('PDP Callback - Selected features:', x_feature, y_feature)

        # Exclude the target variables if they are in the DataFrame
        df = df.drop(target_variable, axis=1)

        features = df.columns.tolist()
        n_classes = 0  # Update this based on  model (classification or regression)

        # Create PDPInteract object
        pdp_inter = pdp.PDPInteract(
            model=trained_model,
            df=df,
            model_features=features,
            n_classes=n_classes, 
            features=[x_feature, y_feature], 
            feature_names=[x_feature, y_feature]
        )

        # Generate PDP Plot
        fig = pdp_inter.plot(
            plot_type='contour',
            to_bins=True,
            plot_pdp=True,
            engine='plotly'
        )

        if fig and isinstance(fig, tuple) and len(fig) == 2:
            filtered_fig = fig[0]  # Extract the figure from the tuple

            # Reverse the colorscale by reversing the list
            reversed_colorscale = [
                [0.0, 'rgb(8,48,107)'],
                [0.125, 'rgb(8,81,156)'],
                [0.25, 'rgb(33,113,181)'],
                [0.375, 'rgb(66,146,198)'],
                [0.5, 'rgb(107,174,214)'],
                [0.625, 'rgb(158,202,225)'],
                [0.75, 'rgb(198,219,239)'],
                [0.875, 'rgb(222,235,247)'],
                [1.0, 'rgb(247,251,255)']
            ]

            # Set the reversed colorscale
            filtered_fig['data'][0]['colorscale'] = reversed_colorscale

            filtered_fig['layout']['height'] = 600
            filtered_fig['layout']['width'] = 550

            return filtered_fig
        else:
            print('Filtering Failed')
            return go.Figure()  # Return an empty figure if pdp_fig is not in the expected format


    except Exception as e:
        print(f"Error in generating PDP: {e}")
        return go.Figure()  # Return an empty figure in case of error

@app.callback(
    Output('single-feature-pdp-plot', 'figure'),
    [Input('pdp-feature-dropdown', 'value')],
    [State('updated-data-store', 'data'),
     State('target-variable-dropdown', 'value')]
)
def update_single_feature_pdp_plot(feature, stored_data, target_variable):
    if stored_data is None or feature is None:
        return go.Figure()  # Return an empty figure if there's missing data or feature

    df = pd.DataFrame(stored_data['data'], columns=stored_data['columns'])
    #target_variable = stored_data.get('target_variable')

    # Exclude the target variables if they are in the DataFrame
    df = df.drop(target_variable, axis=1)

    # Use the global trained model
    global trained_model

    if trained_model is None:
        return go.Figure()  # Return an empty figure if there's no trained model

    # Create the PDP or ICE plot
    pdp_vals = pdp.PDPIsolate(
        model=trained_model,
        df=df,
        model_features=df.columns,
        feature=feature,
        feature_name=feature,
        n_classes=0,
    )

    fig = pdp_vals.plot(
        center=True,
        plot_lines=True,
        frac_to_plot=100,
        cluster=False,
        n_cluster_centers=None,
        cluster_method='accurate',
        plot_pts_dist=True,
        to_bins=True,
        show_percentile=True,
        which_classes=None,
        figsize=None,
        ncols=2,
        plot_params={"pdp_hl": True},
        engine='plotly',
        template='plotly_white',
    )

    if fig and isinstance(fig, tuple) and len(fig) == 2:
            filtered_fig = fig[0]  # Extract the figure from the tuple
            
            filtered_fig['layout']['height'] = 600
            filtered_fig['layout']['width'] = 550

            return filtered_fig
    else:
        print('Filtering Failed')
        return go.Figure() 

@app.callback(
    Output('feature-inputs', 'children'),
    [Input('updated-data-store', 'data'),
     Input('target-variable-dropdown', 'value')]
)
def generate_feature_inputs(updated_data, target_variable):
    if updated_data is None:
        return []

    # Convert updated_data to DataFrame
    df = pd.DataFrame(updated_data['data'], columns=updated_data['columns'])

    # Exclude target variable(s) from the DataFrame
    if target_variable:
        if isinstance(target_variable, list):
            df = df.drop(columns=[col for col in target_variable if col in df.columns])
        else:
            df = df.drop(columns=target_variable, errors='ignore')

    # Create an input component for each feature
    return [html.Div([
        html.Label(f'Input for {feature}'),
        dcc.Input(id={'type': 'feature-input', 'index': feature}, type='text')
    ], style={'margin': '10px'}) for feature in df.columns]

@app.callback(
    Output('optimization-result', 'children'),
    Input('calculate-optimization', 'n_clicks'),
    [State('desired-target-value', 'value'),
     State({'type': 'feature-input', 'index': ALL}, 'value'),
     State('updated-data-store', 'data'),
     State('log-transformed-store', 'data'),
     State('target-variable-dropdown', 'value')]
)
def perform_feature_optimization(n_clicks, desired_target_value, feature_input_values, updated_data, log_transformed_info, selected_target_variable):
    if n_clicks is None:
        return 'No optimization performed yet.'

    if not updated_data:
        return "Data not available for optimization."

    # Convert the updated data to a DataFrame
    df = pd.DataFrame(updated_data['data'], columns=updated_data['columns'])

    # Prepare the feature set and check the target variable for log transformation
    X = df.drop(selected_target_variable, axis=1)
    target_value = np.log10(desired_target_value) if log_transformed_info.get(selected_target_variable, False) else desired_target_value

    # Initialize dictionaries for known and unknown feature values
    known_feature_values = {}
    unknown_feature_names = []

    # Map input values to their corresponding features
    for feature_name, input_value in zip(X.columns, feature_input_values):
        if input_value:
            # Apply log transformation if necessary
            value = np.log10(float(input_value)) if log_transformed_info.get(feature_name, False) else float(input_value)
            known_feature_values[feature_name] = value
        else:
            unknown_feature_names.append(feature_name)

    # Define an objective function for optimization
    def objective(unknown_features):
        all_features = X.iloc[0].copy()  # Base values to start with
        for feature, value in zip(unknown_feature_names, unknown_features):
            all_features[feature] = value
        all_features.update(known_feature_values)

        predicted_target = trained_model.predict([all_features.values])[0]

        return abs(predicted_target - target_value)

    # Perform optimization
    initial_guess = np.zeros(len(unknown_feature_names))
    result = minimize(objective, initial_guess, method='L-BFGS-B')

    # Convert the optimized feature values back to their original scales
    optimized_values_original_scale = []
    for feature, value in zip(unknown_feature_names, result.x):
        if log_transformed_info.get(feature, False):
            # If the feature was originally log-transformed, apply the inverse transformation
            optimized_values_original_scale.append(10**value)
        else:
            # If the feature was not log-transformed, use the value as is
            optimized_values_original_scale.append(value)

    # Prepare the result string for display
    optimization_result_str = "\n".join([f"{feature}: {value:.4f}" for feature, value in zip(unknown_feature_names, optimized_values_original_scale)])
    return f"Optimized Feature Values:\n{optimization_result_str}"

# Run  app
if __name__ == '__main__':
    #app.run_server(debug=True, host='0.0.0.0')
    app.run_server(debug=True, host='0.0.0.0', port=8050)
    #app.run_server(debug=True) #FOR HOSTING


