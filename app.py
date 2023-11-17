#from dotenv import load_dotenv
#load_dotenv()
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import pandas as pd
import base64
import io
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
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
from algorithms import regression_algorithms, classification_algorithms, neural_network_algorithms, algorithm_options, regression_dropdown_options, classification_dropdown_options , neural_network_dropdown_options
from rdkit import Chem
import plotly.graph_objs as go


# Define your external stylesheets
external_stylesheets = [
    'https://fonts.googleapis.com/css?family=Open+Sans&display=swap',
    dbc.themes.LITERA
]

# Initialize your app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#app.suppress_callback_exceptions = True  # Suppress callback exceptions

# Define your constants
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
                dbc.Col(html.Label("Step 1: Upload your CSV file.", style={'fontWeight': 'bold'}), width=12),
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
                dbc.Col(html.Div(id='conversion-div', style={'marginBottom': '20px'}), width=12),
                dbc.Col(html.Div(id='converted-data'), width=12),
                dbc.Col(dcc.Store(id='updated-data-store')),  # Store component to hold updated data
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
                            {'label': 'Classification', 'value': 'Classification'},
                            {'label': 'Neural Network', 'value': 'Neural Network'}
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
                # Row for selecting target variable
                dbc.Col([
                    html.Label('Select Target Variable(s):', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='target-variable-dropdown', multi=True, options=[])
                ], width=10),

                # Column for Alert and Button
                dbc.Col([
                    dbc.Alert(
                        "Running models. This may take some time...",
                        id='loading-alert',
                        is_open=False,
                        color="primary"
                    ),
                    dbc.Button('Run Simulation', id='run-simulation-button', n_clicks=0, color="primary", className="mr-1", disabled=False),
                ], width=2),
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
                dbc.Col([
                    html.Label('Select Global SHAP Plot:', style={'fontWeight': 'bold'}),
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
                        # Options will be populated based on the dataset
                    ),
                    dcc.Graph(id='waterfall-plot')
                ], width=6),
            ])
        ]),
        dcc.Tab(label='Partial Dependence', value='tab-partial-dependence', children=[
            dbc.Row([
                dbc.Col(html.Label('Select Features for PDP Interact:', style={'fontWeight': 'bold'}), width=12),
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
    ]),
    dbc.Alert(id="alert-auto", is_open=False, duration=4000),
    html.Div(id='tab-content'),
    html.Div(id='placeholder-output', style={'display': 'none'}),
    html.Div(id='completion-signal', style={'display': 'none'}),


    # Place the Store component here
    dcc.Store(id='store-for-figures'),

    # Interval component for real-time updates
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # in milliseconds
        n_intervals=0
    ),

    # Div for displaying real-time output messages
    html.Div(id='real-time-output'),
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
    Output('column-dropdown', 'options'),
    [Input('upload-data', 'contents')],
    [State('conversion-selections-store', 'data')]
)
def update_dropdown(contents, stored_selections):
    if contents is None:
        return []

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    df = df[[col for col in df.columns if ".Q" not in col]]

    # Classify and store column types
    global global_column_types
    global_column_types = {col: classify_column_type(df[col], col) for col in df.columns}

    return [{'label': col, 'value': col} for col in df.columns]

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

    elif column_type == "SMILES":
        # Add SMILES specific UI element
        specific_ui_elements.append(html.Div([
            html.Label("Encoding Type:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id={'type': 'smiles-encoding-dropdown', 'index': col},
                options=[
                    {'label': 'Morgan Fingerprint 1024 bits', 'value': 'morgan_1024'},
                    {'label': 'Morgan Fingerprint 2048 bits', 'value': 'morgan_2048'},
                    {'label': 'Topological Fingerprint', 'value': 'topological'},
                    {'label': 'MACCS Keys', 'value': 'maccs'}
                ],
                value=previous_selections.get('smiles_encoding', 'morgan_1024'),
            )
        ], style=component_container_style))

    elif column_type == "Categorical":
        # Add Categorical specific UI element
        specific_ui_elements.append(html.Div([
            html.Label("Encoding Method:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id={'type': 'categorical-encoding-dropdown', 'index': col},
                options=[
                    {'label': 'One-Hot Encoding', 'value': 'one_hot'},
                    {'label': 'Label Encoding', 'value': 'label'},
                    {'label': 'Frequency Encoding', 'value': 'frequency'},
                    {'label': 'Binary Encoding', 'value': 'binary'}
                ],
                value=previous_selections.get('encoding_method', 'one_hot'),
            )
        ], style=component_container_style))

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
            'Download CSV',
            id='download-link',
            download="converted_data.csv",
            href=csv_data,
            target="_blank",
            style={'margin-top': '10px', 'display': 'block'}
        )
    ])

@app.callback(
    Output('updated-data-store', 'data'),
    Input({'type': 'from-dropdown', 'index': ALL}, 'value'),
    Input({'type': 'to-dropdown', 'index': ALL}, 'value'),
    Input({'type': 'rename-input', 'index': ALL}, 'value'),
    Input({'type': 'transform-dropdown', 'index': ALL}, 'value'),
    Input('column-dropdown', 'value'),
    Input('upload-data', 'contents')
)
def update_data_store(from_factors, to_factors, rename_columns, transform_values, selected_columns, contents):
    if contents is None or not selected_columns:
        return dash.no_update

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    df.set_index('Serial No.', inplace=True)
    df = df[selected_columns]

    for col, from_factor, to_factor, new_col_name, transform_value in zip(selected_columns, from_factors, to_factors, rename_columns, transform_values):
        conversion_factor = from_factor / to_factor
        df[col] = df[col] * conversion_factor  # Update values in the original column
        if transform_value == 'Log10':
            df[col] = np.log10(df[col])  # Apply transformation to the original column
        df[col] = df[col].round(4)  # Round values in the original column

        if col != new_col_name:
            df.rename(columns={col: new_col_name}, inplace=True)  # Rename the column if a new name is provided

    df.dropna(inplace=True)
    df_dict = df.to_dict(orient='split')
    return df_dict

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
     Input('updated-data-store', 'data')]
)
def update_scatterplot(x_axis, y_axis, data):
    if not x_axis or not y_axis or not data:
        return dash.no_update
    
    df = pd.DataFrame(data=data['data'], index=data['index'], columns=data['columns'])
    df.index.name = 'LSN'
    fig = px.scatter(df, x=x_axis, y=y_axis, hover_data=[df.index])

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
    # Adjust layout and margins
    fig.update_layout(
        autosize=True,  # Ensure the figure adjusts to the size of its container
        legend=dict(orientation='h', yanchor='bottom', xanchor='right', y=1.02, x=1),
        hovermode='closest',
        xaxis=dict(
            constrain='domain',  # This keeps the x-axis within the plotting domain
            # If you want to set a specific range for x-axis, you can use xaxis=dict(range=[min_val, max_val])
        ),
        yaxis=dict(
            scaleanchor='x',  # This forces the y-axis to scale with the x-axis
            scaleratio=1,  # This sets a 1:1 aspect ratio
            # If you want to set a specific range for y-axis, you can use yaxis=dict(range=[min_val, max_val])
        ),
        # If you need the plot to span the entire width (like in a Dash grid), this might help:
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
            # If you want to set a specific range for x-axis, you can use xaxis=dict(range=[min_val, max_val])
        ),
        yaxis=dict(
            scaleanchor='x',  # This forces the y-axis to scale with the x-axis
            scaleratio=1,  # This sets a 1:1 aspect ratio
            # If you want to set a specific range for y-axis, you can use yaxis=dict(range=[min_val, max_val])
        ),
        # If you need the plot to span the entire width (like in a Dash grid), this might help:
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

# def create_prediction_plot(y_actual, y_predicted, index):
#     fig = px.scatter(x=y_actual, y=y_predicted, labels={'x': 'Actual', 'y': 'Predicted'},
#                      hover_data=[index])
#     fig.update_layout(title="Predicted vs Actual Values", xaxis_title="Actual", yaxis_title="Predicted")
#     return fig

# def create_residuals_plot(y_predicted, residuals, index):
#     fig = px.scatter(x=y_predicted, y=residuals, labels={'x': 'Predicted', 'y': 'Residual'},
#                      hover_data=[index])
#     fig.update_layout(title="Residuals Plot", xaxis_title="Predicted", yaxis_title="Residual")
#     return fig

def create_prediction_plot(y, y_pred_cv, index):
    df = pd.DataFrame({'Actual': y, 'Predicted': y_pred_cv, 'LSN': index})
    fig = px.scatter(df, x='Actual',y='Predicted',hover_data=['LSN'])
    fig.update_layout(title="Predicted vs Actual Values", xaxis_title="Actual", yaxis_title="Predicted", autosize=True,
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
        )
    )
    return fig

def create_residuals_plot(y_pred_cv, residuals, index):
    # Creating a DataFrame from the provided data
    df = pd.DataFrame({'Predicted': y_pred_cv, 'Residuals': residuals, 'LSN': index})
    fig = px.scatter(df, x='Predicted', y='Residuals', hover_data=['LSN'])

    # Updating layout to match the style of create_prediction_plot
    fig.update_layout(  title="Residuals Plot", xaxis_title="Predicted", yaxis_title="Residuals", autosize=True,
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
        )
    )

    return fig

# Set up K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

global trained_model 
trained_model = None

@app.callback(
    [Output('store-for-figures', 'data'),
     Output('best-model-info', 'children'), 
     Output('completion-signal', 'children')],
    Input('run-simulation-button', 'n_clicks'),
    [State('algorithm-type-dropdown', 'value'),
     State('algorithm-dropdown', 'value'),
     State('target-variable-dropdown', 'value'),
     State('updated-data-store', 'data')]
)
def run_simulation(n_clicks, algorithm_type, selected_algorithm, target_variable, data):
    global trained_model
    if n_clicks == 0:
        return dash.no_update, '', dash.no_update
    
    # Convert data from JSON format to DataFrame
    df = pd.DataFrame(data=data['data'], index=data['index'], columns=data['columns'])
    df = preprocess_data(df)
    X = df.drop(target_variable, axis=1)
    y = df[target_variable].squeeze()

    best_score = -float('inf')
    best_model_info = ''
    best_y_pred_cv = None

    if selected_algorithm == 'optiML_all_regression':
        # Handle all regression models
        for alg_name, alg_details in regression_algorithms.items():
            model = alg_details['model']
            param_grid = alg_details['params']

            grid_search = GridSearchCV(model, param_grid, scoring=make_scorer(r2_score), cv=kf, verbose=1, n_jobs=-1)
            grid_search.fit(X, y)

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model_info = f"Best Model: {alg_name}, Best Params: {grid_search.best_params_}, Best Score: {best_score:.4f}"
                best_y_pred_cv = cross_val_predict(grid_search.best_estimator_, X, y, cv=kf)
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

        # Initialize and run GridSearchCV
        grid_search = GridSearchCV(model, param_grid, scoring=make_scorer(r2_score), cv=kf, verbose=1, n_jobs=-1)
        grid_search.fit(X, y)

        # Update best model information
        best_score = grid_search.best_score_
        best_model_info = f"Best Model: {selected_algorithm}, Best Params: {grid_search.best_params_}, Best Score: {best_score:.4f}"
        best_y_pred_cv = cross_val_predict(grid_search.best_estimator_, X, y, cv=kf)

    if best_y_pred_cv.ndim > 1:
        best_y_pred_cv = best_y_pred_cv.flatten()

    # Prepare data for store and return
    trained_model = grid_search.best_estimator_
    y_pred_cv_to_store = pd.Series(best_y_pred_cv)
    store_data = {
        'y': y.tolist(),
        'y_pred_cv': y_pred_cv_to_store.tolist(),
        'index': df.index.tolist()  # Storing the index separately
    }

    print("Trained model:", trained_model)

    return store_data, best_model_info, 'Completed'

@app.callback(
    [Output('run-simulation-button', 'disabled'),
     Output('loading-alert', 'is_open')],
    Input('completion-signal', 'children')
)
def reset_button_and_alert(completion_signal):
    if completion_signal == 'Completed':
        return False, False  # Re-enable button and hide alert
    return dash.no_update, dash.no_update

@app.callback(
    [Output('cross-val-predict-plot', 'figure'), Output('cross-val-residuals-plot', 'figure')],
    Input('store-for-figures', 'data'),
    [State('algorithm-type-dropdown', 'value'), State('algorithm-dropdown', 'value')]
)
def create_ml_plots(stored_data, algorithm_type, selected_algorithm):
    if stored_data is None:
        return dash.no_update, dash.no_update

    y = np.array(stored_data['y'])
    y_pred_cv = np.array(stored_data['y_pred_cv'])
    index = np.array(stored_data['index']) 

    if len(y_pred_cv.shape) > 1:
        y_pred_cv = y_pred_cv.flatten()

    residuals = y - y_pred_cv

    # Create plots
    fig1 = create_prediction_plot(y, y_pred_cv, index)
    fig2 = create_residuals_plot(y_pred_cv, residuals, index)

    return fig1, fig2

def map_value_to_color(value, min_val, max_val, colorscale):
    # Normalize the value to a range between 0 and 1
    normalized_value = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
    return px.colors.sample_colorscale(colorscale, normalized_value)[0]

def create_shap_beeswarm_plot(shap_values, features, feature_values, index):
    # Prepare the SHAP and feature values dataframes
    shap_df = pd.DataFrame(shap_values.values, columns=features, index=index)
    feature_df = pd.DataFrame(feature_values, columns=features, index=index)

    # Melt the dataframes to long format
    shap_df_long = shap_df.melt(ignore_index=False, var_name='Feature', value_name='SHAP').reset_index()
    feature_df_long = feature_df.melt(ignore_index=False, var_name='Feature', value_name='FeatureValue').reset_index()

    # Merge the two dataframes
    merged_df = pd.merge(shap_df_long, feature_df_long, on=['index', 'Feature'])

    # Map feature values to colors
    for feature in features:
        min_val = feature_df[feature].min()
        max_val = feature_df[feature].max()
        mask = merged_df['Feature'] == feature
        merged_df.loc[mask, 'Color'] = merged_df[mask]['FeatureValue'].apply(
            lambda x: map_value_to_color(x, min_val, max_val, 'bluered')
        )

    # Create the beeswarm plot
    fig = go.Figure()
    for feature in features:
        df_subset = merged_df[merged_df['Feature'] == feature]
        hover_text = ["LSN: " + str(idx) + "<br>SHAP: " + str(shap) for idx, shap in zip(df_subset['index'], df_subset['SHAP'])]  
        # Prepare hover text with LSN and SHAP value
        fig.add_trace(go.Scatter(
            x=df_subset['SHAP'],
            y=df_subset['Feature'],
            mode='markers',
            marker=dict(color=df_subset['Color'], size=5),
            text=hover_text,  # Set hover text
            hoverinfo='text',  # Display only the text on hover
            name=feature
        ))
        fig.update_layout(
            xaxis=dict(title='SHAP value'),
            yaxis=dict(title='Feature'),
            showlegend=False  # Disable the legend
        )

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
                tickvals=[],  # No tick values
                ticktext=[]   # No tick text
            ),
            colorscale='bluered',
            showscale=True
        ),
        hoverinfo='none'
    ))

    #   Update layout
    fig.update_layout(xaxis=dict(title='SHAP value'), yaxis=dict(title='Feature'))
    return fig


def create_shap_bar_plot(shap_values, features):
    # Summarize the SHAP values for each feature
    shap_summary = pd.DataFrame(shap_values.values, columns=features).abs().mean().sort_values(ascending=False)

    fig = go.Figure([go.Bar(x=shap_summary.values, y=shap_summary.index, orientation='h')])
    fig.update_layout(title='Mean SHAP Value (Impact on Model Output)',
                      xaxis_title='Mean(|SHAP value|)',
                      yaxis_title='Feature')
    return fig

@app.callback(
    Output('global-shap-plot', 'figure'),
    [Input('global-shap-plot-dropdown', 'value'),
     Input('store-for-figures', 'data')],  # Triggered by either dropdown change or data store update
    [State('updated-data-store', 'data'),
     State('target-variable-dropdown', 'value')]
)
def update_shap_plot(plot_type, stored_data, data, target_variable):
    global trained_model

    # Check if the trained model is available
    if trained_model is None:
        return go.Figure()

    # Handle the case when data is not available
    if not data or not stored_data:
        return go.Figure()

    # Convert data from JSON format to DataFrame
    df = pd.DataFrame(data['data'], columns=data['columns'])

    # Drop the target variable from the DataFrame
    df_for_shap = df.drop(target_variable, axis=1)

    # Create the SHAP explainer and calculate SHAP values
    explainer = shap.Explainer(trained_model.predict, df_for_shap)
    shap_values = explainer(df_for_shap)

    # Assume features and index are already defined or obtained from df_for_shap
    features = df_for_shap.columns
    index = np.array(stored_data['index']) 

    # Obtain feature values (assuming df_for_shap is used)
    feature_values = df_for_shap.values

    if plot_type == 'beeswarm':
        # Create the beeswarm plot
        fig = create_shap_beeswarm_plot(shap_values, features, feature_values, df.index)
        return fig

    # elif plot_type == 'heatmap':
    #     # Logic to create a heatmap plot
    #     # ...

    # elif plot_type == 'bar':
        fig = create_shap_bar_plot(shap_values, features)
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
    if target_variable:
        if isinstance(target_variable, list):
            df = df.drop(columns=[col for col in target_variable if col in df.columns])
        else:
            df = df.drop(columns=target_variable, errors='ignore')

    options = [{'label': col, 'value': col} for col in df.columns]

    return options, options, options

@app.callback(
    Output('pdp-interact-plot', 'figure'),
    [Input('pdp-interact-x-dropdown', 'value'),
     Input('pdp-interact-y-dropdown', 'value')],
    [State('updated-data-store', 'data'),
     State('target-variable-dropdown', 'value')]
)
def update_pdp_interact_plot(x_feature, y_feature, updated_data, target_variables):
    if not x_feature or not y_feature or not updated_data or 'data' not in updated_data or 'columns' not in updated_data:
        return go.Figure()

    df = pd.DataFrame(updated_data['data'], columns=updated_data['columns'])

    # Exclude the target variables if they are in the DataFrame
    if target_variables:
        for target_variable in target_variables:
            if target_variable in df.columns:
                df = df.drop(columns=target_variable)

    features = df.columns.tolist()

    n_classes = 0

    # Create PDPInteract object
    pdp_inter = pdp.PDPInteract(
        model=trained_model,
        df=df,
        model_features=features,
        n_classes=n_classes, 
        features=[x_feature, y_feature], 
        feature_names=[x_feature, y_feature]
)
    # Plotting with Plotly
    fig = pdp_inter.plot(
        plot_type='contour',
        to_bins=True,
        plot_pdp=True,
        show_percentile=False,
        ncols=1,
        plot_params={
            'line_kw': {'color': 'black'},
            'pdp_line_kw': {'linewidth': 2},
            'pdp_fill_between_kw': {'alpha': 0.2, 'color': 'black'}
        },
        engine='plotly'
    )

    return fig

@app.callback(
    Output('single-feature-pdp-plot', 'figure'),
    [Input('pdp-feature-dropdown', 'value'),
     Input('store-for-figures', 'data')]
)
def update_single_feature_pdp_plot(feature, stored_data):
    if stored_data is None or feature is None:
        return go.Figure()  # Return an empty figure if there's no data or feature selected

    df = pd.DataFrame(stored_data['data'], columns=stored_data['columns'])
    target_variable = stored_data.get('target_variable')

    # Check if the target variable is in the DataFrame and remove it
    if target_variable in df.columns:
        df = df.drop(columns=target_variable)

    # Use the global trained model
    global trained_model

    if trained_model is None:
        return go.Figure()  # Return an empty figure if there's no trained model

    # Create the PDP or ICE plot
    pdp_vals = pdp.pdp_isolate(
        model=trained_model,
        dataset=df,
        model_features=df.columns,
        feature=feature
    )

    fig = pdp.pdp_plot(pdp_vals, feature, plot_lines=True, frac_to_plot=0.5, plot_pts_dist=True)

    return fig


# Run your app
if __name__ == '__main__':
    app.run_server(debug=True)


