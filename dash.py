import dash
from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
#pd.options.plotting.backend = "plotly"
import plotly.graph_objects as go
import statsmodels.api as sm
import pandas as pd
from datetime import datetime

#India China Research 2000-2023 Data ---------------
from pandas_datareader import wb
data = wb.download(indicator='SP.POP.TOTL', country=['CN', 'IN'], start=1960, end=2022).reset_index()
data = data.pivot(index='year', columns='country', values='SP.POP.TOTL')
data.index = pd.to_numeric(data.index)

app = JupyterDash()

app.layout = html.Div([
    html.H2('China vs India population forecast 2030'),
    html.Br(),
    html.I('Model training range', style={'text-align':'right', 'color': 'skyblue'}), 
    dcc.RangeSlider(min=int(data.index.min()), max=int(data.index.max()), step=1, value=[1980, 2000], id='range-slider', 
                   marks=None, tooltip={"placement": "bottom", "always_visible": True}), 
    html.Br(),
    html.Hr(),
    dcc.Graph(id='plot', config=dict(displayModeBar=False, autosizable=True)),
    html.Hr()
], style={'width': '100%'})

@app.callback(Output('plot', 'figure'), [Input('range-slider', 'value')])
def update_figure(value):
    df = data[data.index.isin([*range(value[0], value[1]+1)])]
    #Model
    X = sm.add_constant(df.index)
    y = df.China
    result = sm.OLS(y, X)
    p = [*range(value[1], 2030)]
    pred_ols_c = result.fit().predict(sm.add_constant(p))
    y = df.India
    result = sm.OLS(y, X)
    pred_ols_i = result.fit().predict(sm.add_constant(p))
    #fig = df.plot(template="simple_white", labels=dict(value='population'))
    fig = go.Figure(layout=go.Layout(template='simple_white', xaxis_title='year', yaxis_title='population'))
    fig.add_trace(go.Scatter(x=df.index, y=df.China, mode='markers', name='China', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=p, y=pred_ols_c, mode='lines', name='ChinaTrend', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df.India, mode='markers', name='India', marker=dict(color='blue'))) 
    fig.add_trace(go.Scatter(x=p, y=pred_ols_i, mode='lines', name='IndiaTrend', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[2023,2023], y=[0,1700000000], mode='lines', name='=2023=', line=dict(color='black',dash='dot')))
    return fig
    
app.run_server(mode='inline', port=8050, debug=True, use_reloader=False)
