from dash import dcc, html
import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import plotly.graph_objects as go
from config import baseline
from algorithms import regression_algorithms, classification_algorithms, neural_network_algorithms

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

