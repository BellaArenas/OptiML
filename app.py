import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from layout import layout, external_stylesheets
#ToDo: fix divide by 0 error with Log10, add discete data, add classification, fix feature optimization

# Initialize  app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = layout
#app.suppress_callback_exceptions = True  # Suppress callback exceptions
import callbacks

# Run  app
if __name__ == '__main__':
    #app.run_server(debug=True, host='0.0.0.0')
    app.run_server(debug=True, host='0.0.0.0', port=8050)
    #app.run_server(debug=True) #FOR HOSTING


