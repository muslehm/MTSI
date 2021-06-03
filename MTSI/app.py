from __future__ import print_function

# import Dash libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from sklearn import preprocessing


# Loading DataFrame libraries
import pandas as pd
import numpy as np

# Loading Analysis Libraries
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# from sklearn.cluster import SpectralClustering, DBSCAN
import plotly.graph_objs as go
import cyclesJ as Cycle

# Methods:
# 1: calculate distance to previous data point
# 2: calculate distance to calculate average of all data points
# 3: calculate distance to calculate average of previous 3 data points
# 4: calculate distance to settings (except: exPressure, exSpeed, exTorque to average)
# 5: calculate product average of three cycles
# 6: calculate product

# FOR SERVER
# from flask import Flask
# server = Flask(__name__)
# app= dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'],server=server)
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# KK: updated config to make sure this runs behind a reverse proxy!
# See https://community.plotly.com/t/deploy-dash-on-apache-server-solved/4855/18
# app.config.update({
# as the proxy server will remove the prefix
#    'routes_pathname_prefix': '/',

# the front-end will prefix this string to the requests
# that are made to the proxy server
#    'requests_pathname_prefix': '/ilir-students/mtsiproject/'
# })

# The extruder parameters
initialSel = ['exSpeed', 'tempFZ', 'tempZA1', 'tempZA10', 'tempZA11', 'tempZA2', 'tempZA3',
              'exTorque', 'meltTemp', 'exPressure', 'totalTime']

# The pushout cylinder Parameters
# initialSel = ['RWTC1', 'RWTC2', 'suction', 'WTC', 'dieTempZ1', 'dieTempZ2',
#               'dieTempZ3', 'headTempZ1', 'headTempZ2', 'headTempZ3', 'ejectTime', 'totalTime']


switchValue = False  # to detect if the Parallel Coordinates is filtered by the scatter plots
df_cycle_main, df_full_main = Cycle.read_cycles("data_set-2020-06-16", 2)


# Processing the data
def process_df(featurelist, df_cycle_main_):
    df = drop_columns(featurelist, df_cycle_main_)
    df = normalize(df)
    return df


# Dropping features
def drop_columns(featurelist, df_cycle_main_):
    df = df_cycle_main_[featurelist]
    return df


# Normalize the data
def normalize(df):
    x = df.values
    columns = df.columns
    index = df.index
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(x_scaled, columns=columns, index=index)
    return df_normalized


# Normalize the Data
normalizedDF = process_df(initialSel, df_cycle_main)


# Performing a k-means clustering of the PCA features
def kmean(df_):
    kmeans_model = KMeans(n_clusters=2)
    cluster = kmeans_model.fit_predict(df_)
    return cluster

# Performing DBSCAN clustering of the PCA features
# def dbscan(df_):
#     clustering = DBSCAN(eps=0.1, min_samples=2).fit(df_)
#     cluster = clustering.labels_
#     return cluster

# Performing Spectral clustering of the PCA features
# def spectral(df_):
#     clustering = SpectralClustering(n_clusters=2, assign_labels = "discretize",random_state = 0).fit(df_)
#     cluster = clustering.labels_
#     return cluster


# Performing the PCA dimensionality reduction
def pca(df_):
    pca_method = PCA(n_components=2)
    # df_centered = df_ - df_.mean(axis=0)
    pca_method.fit(df_)
    df_pca = pca_method.transform(df_)
    df_.loc[:, 'pca-one'] = df_pca[:, 0]
    df_.loc[:, 'pca-two'] = df_pca[:, 1]
    # df_.loc[:, 'clusters'] = spectral(df_.loc[:, ['pca-one', 'pca-two']])
    # df_.loc[:, 'clusters'] = dbscan(df_.loc[:, ['pca-one', 'pca-two']])
    df_.loc[:, 'clusters'] = kmean(df_.loc[:, ['pca-one', 'pca-two']])
    cluster_map = df_['clusters'] + 1
    df_ = df_[['pca-one', 'pca-two']]
    return df_, cluster_map


# Plotting the PCA scatter plot
def pca_graph(clust_color_df, selectedpoints, df_pca):
    df_pca.loc[:, 'clusters'] = clust_color_df['clusters']
    colors_scale = [[0, 'rgb(241,163,64)'], [1, 'rgb(153,142,195)']]
    points = []
    for point in selectedpoints:
        pos_index = df_pca.index.get_loc(point)
        points.append(pos_index)
    fig = go.Figure(data=go.Scattergl(
        x=df_pca['pca-one'],
        y=df_pca['pca-two'],
        mode='markers',
        hovertemplate='Cycle: ' + df_pca.index,
        hoverlabel=dict(font_size=24),
        hovertext=df_pca.index,
        name='',
        marker=dict(color=df_pca['clusters'], colorscale=colors_scale, line_width=1)
    ))
    fig.update_traces(selectedpoints=points)
    fig.update_layout(dragmode='lasso', title="Principal Component Analysis (PCA) Plot",
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', clickmode='event+select',
                      width=750, height=750)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title_font_size=24, margin_t=50)
    return fig


# T rain the data using the t-sne model
def tsne_train(normalized_df, perplexity, n_iter):
    tsne_method = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne_method.fit_transform(normalized_df)
    return tsne_results


def tsne_fit(df_, tsne_results):
    df_.loc[:, 'tsne-2d-one'] = tsne_results[:, 0]
    df_.loc[:, 'tsne-2d-two'] = tsne_results[:, 1]
    df_ = df_[['tsne-2d-one', 'tsne-2d-two']]
    return df_


# Plot the t-sne scatter plot
def tsne_graph(clust_color_df, selectedpoints, df_tsne):
    df_tsne.loc[:, 'clusters'] = clust_color_df['clusters']
    colors_scale = [[0, 'rgb(241,163,64)'], [1, 'rgb(153,142,195)']]

    points = []
    for point in selectedpoints:
        pos_index = df_tsne.index.get_loc(point)
        points.append(pos_index)

    fig = go.Figure(data=go.Scattergl(
        x=df_tsne['tsne-2d-one'],
        y=df_tsne['tsne-2d-two'],
        mode='markers',
        hovertemplate='Cycle: ' + df_tsne.index,
        hoverlabel=dict(font_size=24),
        hovertext=df_tsne.index,
        name='',
        marker=dict(color=df_tsne['clusters'], colorscale=colors_scale, line_width=1)
    ))
    fig.update_traces(selectedpoints=points)
    fig.update_layout(dragmode='lasso', title='T-distributed Stochastic Neighbor Embedding (t-SNE) Plot',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', clickmode='event+select',
                      width=750, height=750)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title_font_size=24, margin_t=50,)
    return fig


# Plot the heatmap
# noinspection PyTypeChecker
def hm_graph(clust_color_df, df_):
    try:
        colors_scale = [[0, 'rgb(241,163,64)'], [0, 'rgb(153,142,195)']]
        the_colors = clust_color_df['clusters'].unique()
        for color in the_colors:
            print(colors_scale[color][1])
    except Exception as e:
        print(e)
        colors_scale = [[0, 'rgba(200,200,200,0.2)'], [1, 'rgb(241,163,64)'], [2, 'rgb(153,142,195)']]
    y = []
    for the_index in df_.index:
        label_ = "<span style='color:"+colors_scale[clust_color_df.loc[the_index, 'clusters']][1]+"';>" \
                 + the_index + "</span>"
        y.append(label_)
    fig = go.Figure(data=go.Heatmap(
        z=df_,
        x=df_.columns,
        y=y,
        colorscale=[[0, 'rgb(199, 223, 255)'], [1, 'rgb(47, 85, 151)']], hoverlabel=dict(font=dict(size=24)),
        hovertemplate='Cycle: %{y}<br>Feature: %{x}<br>Value: ≈%{z:.3f}',
        hoverlabel_bgcolor='rgb(85, 85, 85)', name=''
    ))
    fig.update_layout(title='Heatmap (Values are normalized to a 0-1 range)', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(tickfont=dict(size=16)),
                      height=1400, clickmode='event+select', margin_t=50)
    fig.update_layout(title_font_size=24)

    return fig


# Plot the parallel coordinates plot
def pg_graph(df_cycle, selected_pts, df_, clust_color_df):
    coords = []
    clust_color = pd.Series(data=clust_color_df['clusters'], index=clust_color_df.index)

    # Set the color_Scale, two colors if all are selected, tri if selection is partial
    if len(df_cycle) > len(df_):
        dfr_ = df_cycle.drop(selected_pts)
        dfr = df_.append(dfr_)
        for x in dfr_.index:
            clust_color.loc[x] = 0
        unique_colors = clust_color.unique()

        if len(unique_colors) == 3:
            colors_scale = [[0, 'rgba(200,200,200,0.2)'], [0.5, 'rgb(241,163,64)'], [1, 'rgb(153,142,195)']]
        elif unique_colors[0] == 2:
            colors_scale = [[0, 'rgba(200,200,200,0.2)'], [1, 'rgb(153,142,195)']]
        else:
            colors_scale = [[0, 'rgba(200,200,200,0.2)'], [1, 'rgb(241,163,64)']]
    else:
        dfr = df_
        colors_scale = [[0, 'rgb(241,163,64)'], [1, 'rgb(153,142,195)']]

    for col in dfr.columns:
        if col != 'clusters':
            coord = {}

            col_ = df_cycle[col].sort_values(ascending=True)
            min_range = float(col_.iloc[0])
            max_range = float(col_.iloc[-1])

            coord['range'] = [min_range, max_range]
            coord['label'] = col
            coord['values'] = dfr[col]
            coords.append(coord)

    fig = go.Figure(data=go.Parcoords(line=dict(color=clust_color, colorscale=colors_scale), labelfont=dict(size=14),
                                      tickfont=dict(size=14), rangefont=dict(size=14),
                                      customdata=dfr.index, dimensions=coords))
    fig.update_layout(title='Parallel Coordinates Plot',
                      paper_bgcolor='rgba(0,0,0,0)', margin_t=85,
                      plot_bgcolor='rgba(0,0,0,0)', clickmode='event+select')
    fig.update_layout(title_font_size=24, title_y=0.99)
    return fig


# Line plot of timeseries features
def lp_graph(df, default, var_name):
    all_y_title = {'RWTC1': 'RWTC1 Voltage', 'RWTC2': 'RWTC2 Voltage', 'suction': 'Suction Voltage',
                   'WTC': 'WTC Voltage', 'dieTempZ1': 'Celsius', 'dieTempZ2': 'Celsius', 'dieTempZ3': 'Celsius',
                   'headTempZ1': 'Celsius', 'headTempZ2': 'Celsius', 'headTempZ3': 'Celsius', 'meltTemp': 'Celsius',
                   'tempFZ': 'Celsius', 'tempZA1': 'Celsius', 'tempZA10': 'Celsius', 'tempZA11': 'Celsius',
                   'tempZA2': 'Celsius', 'tempZA3': 'Celsius', 'exPressure': 'Bar', 'exSpeed': 'RPM', 'exTorque': 'N⋅m'}
    y_title = all_y_title[var_name]
    fig = go.Figure()
    c = 0
    for i in df:
        df_length = [*range(0, len(i))]
        fig.add_trace(go.Scattergl(x=df_length, y=i, mode='lines',
                                   hovertemplate=df.index[c] + ': %{y}', name='', hovertext=df.index[c],
                                   hoverlabel=dict(font_size=24),
                                   marker=dict(color='rgba(46, 46, 46,0.1)', line_width=0.5)
                                   ))
        c += 1

    if type(default) is int:
        test_sum = []
        ss = 0
        for col in df:
            test_sum.append(len(col))
            ss += 1
        ff = int(sum(test_sum)/ss)
        average_list = []
        for x in [*range(0, ff)]:
            val_av = []
            for col in df:
                try:
                    val_av.append(col[x])
                except Exception as e:
                    print(e)
                    pass
            average_list.append(sum(val_av)/len(val_av))

        df_length = [*range(0, len(average_list))]
        fig.add_trace(go.Scattergl(x=df_length, y=average_list, mode='lines', name='',
                                   hovertext='Default Value', hoverlabel=dict(font_size=24),
                                   marker=dict(color='rgba(0, 255, 0,1)', line_width=4)))
    else:
        if type(default[0]) is int or type(default[0]) is float \
                or type(default[0]) is np.int64:
            average_length = 0
            for z in df:
                average_length += len(z)
            average_length = average_length / len(df)
            the_default = [default[0]] * int(average_length)
            df_length = [*range(0, len(the_default))]
            fig.add_trace(go.Scattergl(x=df_length, y=the_default, mode='lines', name='',
                                       hovertext='Default Value', hoverlabel=dict(font_size=24),
                                       marker=dict(color='rgba(0, 255, 0,1)', line_width=4)))
        else:
            df_length = [*range(0, len(default[0]))]
            fig.add_trace(go.Scattergl(x=df_length, y=default[0], mode='lines', name='',
                                       hovertext='Default Value', hoverlabel=dict(font_size=24),
                                       marker=dict(color='rgba(0, 255, 0,1)', line_width=4)))

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', clickmode='event+select',
                      width=2000, height=250, showlegend=False, margin_t=0, margin_b=0)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(title_text=y_title, title_font_size=18)
    return fig


# Set PCA components and cluster colors, and Plot PCA Graph
dfPCA, clusters = pca(normalizedDF)
clust = normalizedDF[['clusters']]
normalizedDF = normalizedDF.drop(['pca-one', 'pca-two', 'clusters'], axis=1)
graphPCA = pca_graph(clust, normalizedDF.index, dfPCA)

# Train t-SNE, and Plot Scatter
tsne_ = tsne_train(normalizedDF, 15, 1000)
dfTSNE = tsne_fit(normalizedDF, tsne_)
normalizedDF = normalizedDF.drop(['tsne-2d-one', 'tsne-2d-two'], axis=1)
graphTSNE = tsne_graph(clust, normalizedDF.index, dfTSNE)

graphHM = hm_graph(clust, normalizedDF)

graphPG = pg_graph(df_cycle_main[initialSel], normalizedDF.index, df_cycle_main[initialSel], clust)

dimensionsDF = dfPCA.merge(dfTSNE, left_index=True, right_index=True)
dimensionsDF.loc[:, 'clusters'] = clusters

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#2F5597',
    'color': 'white',
}

all_ts_features = {'RWTC1': 'setRWTC1', 'RWTC2': 'setRWTC2', 'suction': 'setSuction', 'WTC': 'setWTC',
                   'dieTempZ1': 'setDieTempZ1', 'dieTempZ2': 'setDieTempZ2', 'dieTempZ3': 'setDieTempZ3',
                   'headTempZ1': 'setHeadTempZ1', 'headTempZ2':  'setHeadTempZ2', 'headTempZ3': 'setHeadTempZ3',
                   'meltTemp': 'setMeltTemp', 'tempFZ': 'setTempFZ', 'tempZA1': 'setTempZA1',
                   'tempZA10': 'setTempZA10', 'tempZA11': 'setTempZA11', 'tempZA2': 'setTempZA2',
                   'tempZA3': 'setTempZA3', 'exPressure': '0', 'exSpeed': '0', 'exTorque': '0'}
the_feature = next(iter(all_ts_features))
default_value_feature = all_ts_features[the_feature]
if default_value_feature != '0':
    graph_lp = lp_graph(df_full_main[the_feature], df_full_main[default_value_feature], the_feature)
else:
    graph_lp = lp_graph(df_full_main[the_feature], 0, the_feature)
# Dash-HTML Layout
app.layout = html.Div([
    html.Div([
        html.Div([html.H2("Visualization of Blow-molding Machine Cycles")], className="banner"),
        html.Div([
            html.Div([
                      html.Div(["Compare original value of features for all data points to the setting "
                                "or average (green line). Choose a feature:",
                                dcc.Dropdown(id="timeseries-feat", options=[{"label": x, "value": x}
                                                                            for x, y in all_ts_features.items()],
                                             optionHeight=35, searchable=False, value=list(all_ts_features.keys())[0],
                                             clearable=False, className='ticker dropdown')],
                               className="two columns theoriginal"),
                      html.Div([
                          dcc.Loading(id="loading-gadget_2", type="cube", color='#C7DFFF', fullscreen=True,
                                      children=html.Div([dcc.Graph(
                                          id="time-series-chart", figure=graph_lp,
                                          config={
                                              'modeBarButtonsToRemove': ['zoom2d', 'select2d', 'autoScale2d',
                                                                         'toggleSpikelines', 'hoverClosestCartesian',
                                                                         'hoverCompareCartesian']}
                                      )], className='time series'))], className='eight columns')
            ], className='twelve columns'),
            html.Div([
                html.Div([dcc.Tabs(id='data-tabs', value='extruder', children=[
                    dcc.Tab(label='Extruder', value='extruder', selected_style=tab_selected_style),
                    dcc.Tab(label='Eject Cylinder', value='accumulator', selected_style=tab_selected_style),
                    dcc.Tab(label='All', value='all', selected_style=tab_selected_style)])], className='six columns'),
                html.Div([
                    html.Div(["Method:", dcc.Dropdown(id='calc-method',
                                                      options=[{'label': 'Distance to Previous', 'value': '1'},
                                                               {'label': 'Distance to Average', 'value': '2'},
                                                               {'label': 'Distance to Previous Three', 'value': '3'},
                                                               {'label': 'Distance to Setting', 'value': '4'},
                                                               {'label': 'Product Average Three', 'value': '5'},
                                                               {'label': 'Product Selective', 'value': '6'}],
                                                      value='2')], className="method select"),
                    dcc.Upload(id='upload-data',
                               children=html.Div(['Drag and Drop or ', html.A('Select Files')]), className='upload_box',
                               multiple=True, accept='.json')], className='five columns more')]),
            html.Div([
                dcc.Loading(
                    id="loading-gadget",
                    type="cube",
                    color='#C7DFFF', className='loader_two',
                    children=html.Div(id='selected-data', className='twelve columns'))
            ]),
            html.Div([
                html.Div(dcc.Graph(id="heatmap", figure=graphHM, config={
                    'modeBarButtonsToRemove': ['zoom2d', 'select2d', 'autoScale2d',
                                               'hoverClosestCartesian', 'hoverCompareCartesian']}),
                         id='plot-hm', className='five columns'),
                html.Div([
                    html.Div([html.Div([html.Button(id="plot-button", n_clicks=0, children="Plot")],
                                       className='divFeat'),
                              html.Div([dcc.Checklist(
                                  id="feature_list", inputStyle={}, inputClassName='checkBoxInput',
                                  options=[{'label': x, 'value': x}
                                           for x in normalizedDF.columns],
                                  value=normalizedDF.columns, labelClassName='checkBoxLabel')],
                                  id='divFeat', className='divFeat')],
                             className='seven columns'),
                    html.Div([
                        html.Div([html.Label('Perplexity (min=5, max=100):', className='labelTSNE'),
                                  dcc.Input(id='perplexity-state', type='number', value=15, max=100, min=5, step=1)],
                                 className='divTSNE'),
                        html.Div([html.Label('Iterations (min=200, max=1000):', className='labelTSNE'),
                                  dcc.Input(id="n-iter-state", type='number', value=1000, max=1000, min=200, step=1)],
                                 className='divTSNE'),
                        html.Button(id="tsne-train-button", n_clicks=0, children="Train t-SNE")],
                        className='seven columns'),
                    html.Div(dcc.Graph(id="tsne", figure=graphTSNE, config={'modeBarButtonsToRemove': [
                        'zoom2d', 'select2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian',
                        'hoverCompareCartesian']}),
                             id='plot-tsne', className='three columns'),
                    html.Div(dcc.Graph(id="pca", figure=graphPCA, config={'modeBarButtonsToRemove': [
                        'zoom2d', 'select2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian',
                        'hoverCompareCartesian']}),
                             id='plot-pca', className='three columns'),
                    html.Div(dcc.Graph(id="graphPG", figure=graphPG, config={'modeBarButtonsToRemove': [
                        'zoom2d', 'select2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian',
                        'hoverCompareCartesian']}),
                             id='plot-pg', className='seven columns')
                ])
            ]),
        ])]),

    # passing dataframe values
    html.Div(id='feature-value', children=dimensionsDF.to_json(date_format='iso', orient='split'),
             style={'display': 'none'}),
    html.Div(id='normalized-value', children=normalizedDF.to_json(date_format='iso', orient='split'),
             style={'display': 'none'}),
    html.Div(id='processed-df', children=df_cycle_main.to_json(date_format='iso', orient='split'),
             style={'display': 'none'}),
    html.Div(id='switchSel-value', children=switchValue, style={'display': 'none'}),
    html.Div(dcc.Markdown(tsne_, id='tsne-value'), style={'display': 'none'}),
    html.Div(id='raw-df', children=df_full_main.to_json(date_format='iso', orient='split'), style={'display': 'none'}),
    html.Div(id='test-df', children='', style={'display': 'none'}),
])

del dimensionsDF
del df_cycle_main
del df_full_main
del normalizedDF


@app.callback(
    [Output('time-series-chart', 'figure'),
     Output('processed-df', 'children'),
     Output('test-df', 'children'),
     Output('raw-df', 'children')],
    [Input('calc-method', 'value'),
     Input('timeseries-feat', 'value'),
     Input('upload-data', 'last_modified')],
    [State('calc-method', 'value'),
     State('upload-data', 'contents'),
     State('upload-data', 'filename'),
     State('raw-df', 'children')], prevent_initial_call=True)
def upload_data(calc_method, ts_feature, list_of_dates, sel_method, list_of_contents, list_of_names, raw_df):
    the_input = [p['prop_id'] for p in dash.callback_context.triggered][0]

    all_ts_features_ = {'RWTC1': 'setRWTC1', 'RWTC2': 'setRWTC2', 'suction': 'setSuction', 'WTC': 'setWTC',
                        'dieTempZ1': 'setDieTempZ1', 'dieTempZ2': 'setDieTempZ2', 'dieTempZ3': 'setDieTempZ3',
                        'headTempZ1': 'setHeadTempZ1', 'headTempZ2': 'setHeadTempZ2', 'headTempZ3': 'setHeadTempZ3',
                        'meltTemp': 'setMeltTemp', 'tempFZ': 'setTempFZ', 'tempZA1': 'setTempZA1',
                        'tempZA10': 'setTempZA10', 'tempZA11': 'setTempZA11', 'tempZA2': 'setTempZA2',
                        'tempZA3': 'setTempZA3', 'exPressure': '0', 'exSpeed': '0', 'exTorque': '0'}
    the_feature_ = ts_feature
    the_default_value_feature = all_ts_features_[the_feature_]

    if the_input == 'timeseries-feat.value':
        df_full = pd.read_json(raw_df, orient='split')
        if the_default_value_feature != '0':
            graph_lp_ = lp_graph(df_full[the_feature_], df_full[the_default_value_feature], the_feature_)
        else:
            graph_lp_ = lp_graph(df_full[the_feature_], 0, the_feature_)

        del df_full
        return graph_lp_, dash.no_update, dash.no_update, dash.no_update
    elif the_input == 'calc-method.value':
        df_full = pd.read_json(raw_df, orient='split')
        df_cycle_main_ = Cycle.update_method(df_full, int(calc_method))
        del df_full
        return dash.no_update, df_cycle_main_.to_json(date_format='iso', orient='split'), '1', dash.no_update
    elif the_input == 'upload-data.last_modified':
        if list_of_contents is not None:
            df_cycle_main_, df_full_main_ = Cycle.read_cycles_(list_of_contents, list_of_names, list_of_dates,
                                                               int(sel_method))
            if the_default_value_feature != '0':
                graph_lp_ = lp_graph(df_full_main_[the_feature_],
                                     df_full_main_[the_default_value_feature], the_feature_)
            else:
                graph_lp_ = lp_graph(df_full_main_[the_feature_], 0, the_feature_)

            return graph_lp_, df_cycle_main_.to_json(date_format='iso', orient='split'), '1',\
                df_full_main_.to_json(date_format='iso', orient='split')
    else:
        raise PreventUpdate


# Interactivity and cross-filtering of plots + new feature selection, training t-sne and re-plotting
@app.callback(
    [Output('selected-data', 'children'),
     Output('plot-hm', 'children'),
     Output('plot-pg', 'children'),
     Output('plot-tsne', 'children'),
     Output('plot-pca', 'children'),
     Output('feature-value', 'children'),
     Output('normalized-value', 'children'),
     Output('switchSel-value', 'children'),
     Output('tsne-value', 'children'),
     Output('feature_list', 'value'),
     Output('divFeat', 'children'),
     Output('upload-data', 'contents')
     ],
    [Input('test-df', 'children'),
     Input('tsne', 'selectedData'),
     Input('pca', 'selectedData'),
     Input('graphPG', 'restyleData'),
     Input('tsne-train-button', 'n_clicks'),
     Input('plot-button', 'n_clicks'),
     Input('data-tabs', 'value')
     ],
    [State('tsne-value', 'children'),
     State('switchSel-value', 'children'),
     State('feature-value', 'children'),
     State('normalized-value', 'children'),
     State('processed-df', 'children'),
     State('graphPG', 'figure'),
     State('perplexity-state', 'value'),
     State('n-iter-state', 'value'),
     State('feature_list', 'value'),
     State('pca', 'figure'),
     State('tsne', 'figure')], prevent_initial_call=True)
def display_selected_data(test_df, selected_tsne, selected_pca, selected_pg, n_click, p_click,
                          data_tab, tsne, switch_sel, dimensions_df, normalized_df, processed_df,
                          pg_figure, perplexity, n_iter, feature_list, pca_figure, tsne_figure):

    if n_click or test_df or p_click or tsne:
        pass
    the_input = [p['prop_id'] for p in dash.callback_context.triggered][0]
    dimen_df = pd.read_json(dimensions_df, orient='split')
    df_pca = dimen_df[['pca-one', 'pca-two']]
    df_tsne = dimen_df[['tsne-2d-one', 'tsne-2d-two']]
    clusters_ = dimen_df[['clusters']]
    del dimen_df
    df_cycle_main_ = pd.read_json(processed_df, orient='split')
    normal_df = pd.read_json(normalized_df, orient='split')
    try:
        normalized_df_ = normal_df[feature_list]
    except Exception as e:
        print(e)
        normalized_df_ = process_df(feature_list, df_cycle_main_)

    df_cycle = df_cycle_main_[normalized_df_.columns]
    pca_fig = pca_figure['data'][0]
    tsne_fig = tsne_figure['data'][0]

    pca_sel = pca_fig.get('selectedpoints', None)
    tsne_sel = tsne_fig.get('selectedpoints', None)
    if the_input == 'test-df.children':
        try:
            df_cycle = df_cycle_main_[feature_list]
            normalized_df_ = process_df(feature_list, df_cycle_main_)
            # Set PCA components and cluster colors, and Plot PCA Graph
            df_pca, clusters_ = pca(normalized_df_)
            clust_ = normalized_df_[['clusters']]
            normalized_df_ = normalized_df_.drop(['pca-one', 'pca-two', 'clusters'], axis=1)
            graph_pca = pca_graph(clust_, normalized_df_.index, df_pca)

            # Train t-SNE, and Plot Scatter
            n_iter = int(n_iter)
            perplexity = int(perplexity)
            tsne = tsne_train(normalized_df_, perplexity, n_iter)
            df_tsne = tsne_fit(normalized_df_, tsne)
            normalized_df_ = normalized_df_.drop(['tsne-2d-one', 'tsne-2d-two'], axis=1)
            graph_tsne = tsne_graph(clust_, normalized_df_.index, df_tsne)
            switch_sel = False
            graph_hm = hm_graph(clust_, normalized_df_)
            graph_pg = pg_graph(df_cycle, df_cycle.index, df_cycle, clust_)
            dim_df = df_pca.merge(df_tsne, left_index=True, right_index=True)
            dim_df.loc[:, 'clusters'] = clusters_

            return "Dataset is calculated!", dcc.Graph(id="heatmap",
                                                       figure=graph_hm,
                                                       config={'modeBarButtonsToRemove': [
                                                           'zoom2d', 'select2d', 'autoScale2d',
                                                           'hoverClosestCartesian',
                                                           'hoverCompareCartesian']}), \
                   dcc.Graph(id="graphPG", figure=graph_pg, config={'modeBarButtonsToRemove': [
                       'zoom2d', 'select2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian',
                       'hoverCompareCartesian']}), \
                   dcc.Graph(id="tsne", figure=graph_tsne, config={'modeBarButtonsToRemove': [
                       'zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d', 'hoverClosestCartesian',
                       'hoverCompareCartesian']}), \
                   dcc.Graph(id="pca", figure=graph_pca, config={'modeBarButtonsToRemove': [
                       'zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d', 'hoverClosestCartesian',
                       'hoverCompareCartesian']}), dim_df.to_json(date_format='iso', orient='split'), \
                   normalized_df_.to_json(date_format='iso', orient='split'), switch_sel, tsne, dash.no_update, \
                   dash.no_update, ''

        except Exception as e:
            print(e)
            raise PreventUpdate
    elif the_input == 'data-tabs.value':
        if data_tab == 'extruder':
            feature_sel = ['exSpeed', 'tempFZ', 'tempZA1', 'tempZA10', 'tempZA11', 'tempZA2', 'tempZA3', 'exTorque',
                           'meltTemp', 'exPressure', 'totalTime']
            normalized_df_ = process_df(feature_sel, df_cycle_main_)
            df_cycle = df_cycle_main_[normalized_df_.columns]

            # Set PCA components and cluster colors, and Plot PCA Graph
            df_pca, clusters_ = pca(normalized_df_)
            clust_ = normalized_df_[['clusters']]
            normalized_df_ = normalized_df_.drop(['pca-one', 'pca-two', 'clusters'], axis=1)
            graph_pca = pca_graph(clust_, normalized_df_.index, df_pca)

            # Train t-SNE, and Plot Scatter
            n_iter = int(n_iter)
            perplexity = int(perplexity)
            tsne = tsne_train(normalized_df_, perplexity, n_iter)
            df_tsne = tsne_fit(normalized_df_, tsne)
            normalized_df_ = normalized_df_.drop(['tsne-2d-one', 'tsne-2d-two'], axis=1)
            graph_tsne = tsne_graph(clust_, normalized_df_.index, df_tsne)
            switch_sel = False
            graph_hm = hm_graph(clust_, normalized_df_)
            graph_pg = pg_graph(df_cycle, df_cycle.index, df_cycle, clust_)
            dim_df = df_pca.merge(df_tsne, left_index=True, right_index=True)
            dim_df.loc[:, 'clusters'] = clusters_

            return "All plots are updated! You are now examining the extruder data.", \
                   dcc.Graph(id="heatmap", figure=graph_hm, config={
                       'modeBarButtonsToRemove': ['zoom2d', 'select2d', 'autoScale2d', 'hoverClosestCartesian',
                                                  'hoverCompareCartesian']}), \
                   dcc.Graph(id="graphPG", figure=graph_pg, config={'modeBarButtonsToRemove': [
                       'zoom2d', 'select2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian',
                       'hoverCompareCartesian']}), \
                   dcc.Graph(id="tsne", figure=graph_tsne, config={'modeBarButtonsToRemove': [
                       'zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d', 'hoverClosestCartesian',
                       'hoverCompareCartesian']}), \
                   dcc.Graph(id="pca", figure=graph_pca, config={'modeBarButtonsToRemove': [
                       'zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d', 'hoverClosestCartesian',
                       'hoverCompareCartesian']}), dim_df.to_json(date_format='iso', orient='split'),\
                   normalized_df_.to_json(date_format='iso', orient='split'), switch_sel, tsne, dash.no_update, \
                   dcc.Checklist(id="feature_list", inputStyle={}, inputClassName='checkBoxInput',
                                 options=[{'label': x, 'value': x}
                                          for x in normalized_df_.columns],
                                 value=normalized_df_.columns, labelClassName='checkBoxLabel'), dash.no_update
        elif data_tab == 'accumulator':
            feature_sel = ['RWTC1', 'RWTC2', 'suction', 'WTC', 'dieTempZ1', 'dieTempZ2',
                           'dieTempZ3', 'headTempZ1', 'headTempZ2', 'headTempZ3', 'ejectTime', 'totalTime']
            normalized_df_ = process_df(feature_sel, df_cycle_main_)
            df_cycle = df_cycle_main_[normalized_df_.columns]

            # Set PCA components and cluster colors, and Plot PCA Graph
            df_pca, clusters_ = pca(normalized_df_)
            clust_ = normalized_df_[['clusters']]
            normalized_df_ = normalized_df_.drop(['pca-one', 'pca-two', 'clusters'], axis=1)
            graph_pca = pca_graph(clust_, normalized_df_.index, df_pca)

            # Train t-SNE, and Plot Scatter
            n_iter = int(n_iter)
            perplexity = int(perplexity)
            tsne = tsne_train(normalized_df_, perplexity, n_iter)
            df_tsne = tsne_fit(normalized_df_, tsne)
            normalized_df_ = normalized_df_.drop(['tsne-2d-one', 'tsne-2d-two'], axis=1)
            graph_tsne = tsne_graph(clust_, normalized_df_.index, df_tsne)
            switch_sel = False
            graph_hm = hm_graph(clust_, normalized_df_)
            graph_pg = pg_graph(df_cycle, df_cycle.index, df_cycle, clust_)
            dim_df = df_pca.merge(df_tsne, left_index=True, right_index=True)
            dim_df.loc[:, 'clusters'] = clusters_

            return "All plots are updated! You are now examining the EjectCylinder data.", \
                   dcc.Graph(id="heatmap", figure=graph_hm,
                             config={'modeBarButtonsToRemove': ['zoom2d', 'select2d', 'autoScale2d',
                                                                'hoverClosestCartesian', 'hoverCompareCartesian']}), \
                   dcc.Graph(id="graphPG", figure=graph_pg, config={'modeBarButtonsToRemove': [
                       'zoom2d', 'select2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian',
                       'hoverCompareCartesian']}), \
                   dcc.Graph(id="tsne", figure=graph_tsne, config={'modeBarButtonsToRemove': [
                       'zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d', 'hoverClosestCartesian',
                       'hoverCompareCartesian']}), \
                   dcc.Graph(id="pca", figure=graph_pca, config={'modeBarButtonsToRemove': [
                       'zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d', 'hoverClosestCartesian',
                       'hoverCompareCartesian']}), dim_df.to_json(date_format='iso', orient='split'), \
                   normalized_df_.to_json(date_format='iso', orient='split'),\
                   switch_sel, tsne, dash.no_update, \
                   dcc.Checklist(id="feature_list", inputStyle={}, inputClassName='checkBoxInput',
                                 options=[{'label': x, 'value': x}
                                          for x in normalized_df_.columns],
                                 value=normalized_df_.columns, labelClassName='checkBoxLabel'), dash.no_update
        elif data_tab == 'all':
            feature_sel = ['exSpeed', 'tempFZ', 'tempZA1', 'tempZA10', 'tempZA11', 'tempZA2', 'tempZA3',
                           'exTorque', 'meltTemp', 'exPressure', 'RWTC1', 'RWTC2', 'suction', 'WTC', 'dieTempZ1',
                           'dieTempZ2', 'dieTempZ3', 'headTempZ1', 'headTempZ2', 'headTempZ3', 'ejectTime',
                           'totalTime']
            normalized_df_ = process_df(feature_sel, df_cycle_main_)
            df_cycle = df_cycle_main_[normalized_df_.columns]

            # Set PCA components and cluster colors, and Plot PCA Graph
            df_pca, clusters_ = pca(normalized_df_)
            clust_ = normalized_df_[['clusters']]
            normalized_df_ = normalized_df_.drop(['pca-one', 'pca-two', 'clusters'], axis=1)
            graph_pca = pca_graph(clust_, normalized_df_.index, df_pca)

            # Train t-SNE, and Plot Scatter
            n_iter = int(n_iter)
            perplexity = int(perplexity)
            tsne = tsne_train(normalized_df_, perplexity, n_iter)
            df_tsne = tsne_fit(normalized_df_, tsne)
            normalized_df_ = normalized_df_.drop(['tsne-2d-one', 'tsne-2d-two'], axis=1)
            graph_tsne = tsne_graph(clust_, normalized_df_.index, df_tsne)
            switch_sel = False
            graph_hm = hm_graph(clust_, normalized_df_)
            graph_pg = pg_graph(df_cycle, df_cycle.index, df_cycle, clust_)
            dim_df = df_pca.merge(df_tsne, left_index=True, right_index=True)
            dim_df.loc[:, 'clusters'] = clusters_

            return "All plots are updated! You are now examining the All the features.", \
                   dcc.Graph(id="heatmap", figure=graph_hm,
                             config={'modeBarButtonsToRemove': ['zoom2d', 'select2d', 'autoScale2d',
                                                                'hoverClosestCartesian', 'hoverCompareCartesian']}), \
                   dcc.Graph(id="graphPG", figure=graph_pg, config={'modeBarButtonsToRemove': [
                       'zoom2d', 'select2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian',
                       'hoverCompareCartesian']}), \
                   dcc.Graph(id="tsne", figure=graph_tsne, config={'modeBarButtonsToRemove': [
                       'zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d', 'hoverClosestCartesian',
                       'hoverCompareCartesian']}), \
                   dcc.Graph(id="pca", figure=graph_pca, config={'modeBarButtonsToRemove': [
                       'zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d', 'hoverClosestCartesian',
                       'hoverCompareCartesian']}), dim_df.to_json(date_format='iso', orient='split'), \
                   normalized_df_.to_json(date_format='iso', orient='split'), \
                   switch_sel, tsne, dash.no_update, \
                   dcc.Checklist(id="feature_list", inputStyle={}, inputClassName='checkBoxInput',
                                 options=[{'label': x, 'value': x}
                                          for x in normalized_df_.columns],
                                 value=normalized_df_.columns, labelClassName='checkBoxLabel'), \
                   dash.no_update
        else:
            raise PreventUpdate
    elif the_input == 'tsne-train-button.n_clicks':
        try:
            n_iter = int(n_iter)
            perplexity = int(perplexity)

            if n_iter > 1000:
                n_iter = 1000
            elif n_iter < 200:
                n_iter = 200

            if perplexity > 100:
                perplexity = 100
            elif perplexity < 5:
                perplexity = 5

            switch_sel = False
            tsne = tsne_train(normalized_df_, perplexity, n_iter)
            df_tsne = tsne_fit(normalized_df_, tsne)
            normalized_df_ = normalized_df_.drop(['tsne-2d-one', 'tsne-2d-two'], axis=1)
            graph_tsne = tsne_graph(clusters_, normalized_df_.index, df_tsne)
            graph_pg = pg_graph(df_cycle, normalized_df_.index, df_cycle, clusters_)
            graph_pca = pca_graph(clusters_, normalized_df_.index, df_pca)
            graph_hm = hm_graph(clusters_, normalized_df_)
            dim_df = df_pca.merge(df_tsne, left_index=True, right_index=True)
            dim_df.loc[:, 'clusters'] = clusters_

            return "T-SNE is trained, and plots are updated!", \
                   dcc.Graph(id="heatmap", figure=graph_hm, config={'modeBarButtonsToRemove': [
                    'zoom2d', 'select2d', 'autoScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian']}), \
                dcc.Graph(id="graphPG", figure=graph_pg, config={'modeBarButtonsToRemove': [
                    'zoom2d', 'select2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian',
                    'hoverCompareCartesian']}), \
                dcc.Graph(id="tsne", figure=graph_tsne, config={'modeBarButtonsToRemove': [
                    'zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d', 'hoverClosestCartesian',
                    'hoverCompareCartesian']}), \
                dcc.Graph(id="pca", figure=graph_pca, config={'modeBarButtonsToRemove': [
                    'zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d', 'hoverClosestCartesian',
                    'hoverCompareCartesian']}), dim_df.to_json(date_format='iso', orient='split'), dash.no_update,\
                   switch_sel, tsne, dash.no_update, dash.no_update, dash.no_update
        except Exception:
            raise PreventUpdate
    elif the_input == 'plot-button.n_clicks':
        try:
            df_cycle = df_cycle_main_[normalized_df_.columns]

            # Set PCA components and cluster colors, and Plot PCA Graph
            df_pca, clusters_ = pca(normalized_df_)
            clust_ = normalized_df_[['clusters']]
            normalized_df_ = normalized_df_.drop(['pca-one', 'pca-two', 'clusters'], axis=1)
            graph_pca = pca_graph(clust_, normalized_df_.index, df_pca)

            # Train t-SNE, and Plot Scatter
            n_iter = int(n_iter)
            perplexity = int(perplexity)
            tsne = tsne_train(normalized_df_, perplexity, n_iter)
            df_tsne = tsne_fit(normalized_df_, tsne)
            normalized_df_ = normalized_df_.drop(['tsne-2d-one', 'tsne-2d-two'], axis=1)
            graph_tsne = tsne_graph(clust_, normalized_df_.index, df_tsne)
            switch_sel = False
            graph_hm = hm_graph(clust_, normalized_df_)
            graph_pg = pg_graph(df_cycle, df_cycle.index, df_cycle, clust_)
            dim_df = df_pca.merge(df_tsne, left_index=True, right_index=True)
            dim_df.loc[:, 'clusters'] = clusters_

            return "All plots are updated!", dcc.Graph(id="heatmap", figure=graph_hm,
                                                       config={'modeBarButtonsToRemove': [
                                                           'zoom2d', 'select2d', 'autoScale2d', 'hoverClosestCartesian',
                                                           'hoverCompareCartesian']}), \
                dcc.Graph(id="graphPG", figure=graph_pg, config={'modeBarButtonsToRemove': [
                    'zoom2d', 'select2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian',
                    'hoverCompareCartesian']}), \
                dcc.Graph(id="tsne", figure=graph_tsne, config={'modeBarButtonsToRemove': [
                    'zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d', 'hoverClosestCartesian',
                    'hoverCompareCartesian']}), \
                dcc.Graph(id="pca", figure=graph_pca, config={'modeBarButtonsToRemove': [
                    'zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d', 'hoverClosestCartesian',
                    'hoverCompareCartesian']}), dim_df.to_json(date_format='iso', orient='split'), dash.no_update,\
                   switch_sel, tsne, dash.no_update, dash.no_update, dash.no_update
        except Exception:
            raise PreventUpdate
    elif the_input == 'graphPG.restyleData' and selected_pg:
        figure = pg_figure['data'][0]
        curr_dims = figure.get('dimensions', None)
        pg_labels = []
        for i in curr_dims:
            pg_labels.append(i['label'])

        if pca_sel is None or tsne_sel is None:
            switch_sel = False

        test_key = str(list(selected_pg[0].keys())[0])
        test_str = "constraint"

        if test_str in test_key:
            try:
                if switch_sel is True:
                    len(pca_sel)
                    points = normalized_df_.index[pca_sel]
                else:
                    points = normalized_df_.index
            except Exception as e:
                print(e)
                points = normalized_df_.index
            x = 0
            constraint_range = ''

            temp_df = df_cycle.loc[points]

            for i in pg_labels:
                try:
                    const_r = curr_dims[x]['constraintrange']
                    constraint_range = constraint_range + "      " + i + ": " + str(const_r)
                    if type(const_r[0]) == list:
                        temp_points = []
                        for pg_range in const_r:
                            temp_points.extend(temp_df.loc[(temp_df[i] >= pg_range[0]) & (temp_df[i] <=
                                                                                          pg_range[1])].index)
                        temp_df = temp_df.loc[temp_points]
                    else:
                        temp_df = temp_df.loc[(temp_df[i] >= const_r[0]) & (temp_df[i] <= const_r[1])]

                    x = x + 1
                except Exception as e:
                    print(e)
                    x = x + 1
            f_selected = len(points)
            points = temp_df.index
            graph_hm = hm_graph(clusters_, normalized_df_.loc[points, :])

            text_sel = str(len(temp_df)) + "/" + str(f_selected)
            text_sel = text_sel + " points are selected in the Parallel Coordinates Plot. "
            text_sel = text_sel + "{} is the range of selection.".format(constraint_range)

            if switch_sel is True:
                return text_sel, dcc.Graph(id="heatmap", figure=graph_hm, config={'modeBarButtonsToRemove': [
                    'zoom2d', 'select2d', 'autoScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian']}),\
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, switch_sel, \
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            else:
                graph_pca = pca_graph(clusters_, points, df_pca)
                graph_tsne = tsne_graph(clusters_, points, df_tsne)
                return text_sel, dcc.Graph(id="heatmap", figure=graph_hm, config={'modeBarButtonsToRemove': [
                    'zoom2d', 'select2d', 'autoScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian']}), \
                    dash.no_update, dcc.Graph(id="tsne", figure=graph_tsne, config={'modeBarButtonsToRemove': [
                        'zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d', 'hoverClosestCartesian',
                        'hoverCompareCartesian']}), dcc.Graph(id="pca", figure=graph_pca, config={
                            'modeBarButtonsToRemove': ['zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d',
                                                       'hoverClosestCartesian', 'hoverCompareCartesian']}), \
                    dash.no_update, dash.no_update, switch_sel, dash.no_update, dash.no_update, dash.no_update, \
                    dash.no_update

        else:
            raise PreventUpdate
    elif the_input == 'pca.selectedData':
        switch_sel = True
        try:
            points = []
            for point in selected_pca["points"]:
                points.append(point["hovertext"])
            graph_hm = hm_graph(clusters_, normalized_df_.loc[points, :])
            graph_pg = pg_graph(df_cycle, normalized_df_.loc[points, :].index, df_cycle.loc[points, :],
                                clusters_.loc[points])
            graph_tsne = tsne_graph(clusters_, points, df_tsne)
            text_sel = str(len(normalized_df_.loc[points, :])) + "/" + str(len(normalized_df_))
            text_sel = text_sel + " points are Selected in the PCA plot."
            return text_sel, dcc.Graph(id="heatmap", figure=graph_hm, config={'modeBarButtonsToRemove': [
                'zoom2d', 'select2d', 'autoScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian']}), \
                dcc.Graph(id="graphPG", figure=graph_pg, config={'modeBarButtonsToRemove': [
                       'zoom2d', 'select2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian',
                       'hoverCompareCartesian']}), dcc.Graph(id="tsne", figure=graph_tsne, config={
                        'modeBarButtonsToRemove': ['zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d',
                                                   'hoverClosestCartesian', 'hoverCompareCartesian']}), \
                dash.no_update, dash.no_update, dash.no_update, switch_sel, dash.no_update, \
                dash.no_update, dash.no_update, dash.no_update
        except Exception as e:
            print(e)
            graph_hm = hm_graph(clusters_, normalized_df_)
            graph_pg = pg_graph(df_cycle, normalized_df_.index, df_cycle, clusters_)
            graph_tsne = tsne_graph(clusters_, normalized_df_.index, df_tsne)
            return "All Points are Selected", dcc.Graph(id="heatmap", figure=graph_hm, config={
                'modeBarButtonsToRemove': ['zoom2d', 'select2d', 'autoScale2d', 'hoverClosestCartesian',
                                           'hoverCompareCartesian']}), \
                dcc.Graph(id="graphPG", figure=graph_pg, config={'modeBarButtonsToRemove': [
                    'zoom2d', 'select2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian',
                    'hoverCompareCartesian']}), \
                dcc.Graph(id="tsne", figure=graph_tsne, config={'modeBarButtonsToRemove': [
                    'zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d', 'hoverClosestCartesian',
                    'hoverCompareCartesian']}), dash.no_update, dash.no_update, dash.no_update, switch_sel, \
                   dash.no_update, dash.no_update, dash.no_update, dash.no_update
    elif the_input == 'tsne.selectedData':
        raw_df = ''
        print(raw_df)
        switch_sel = True
        try:
            points = []
            for point in selected_tsne["points"]:
                points.append(point["hovertext"])
            graph_hm = hm_graph(clusters_, normalized_df_.loc[points, :])
            graph_pca = pca_graph(clusters_, points, df_pca)
            graph_pg = pg_graph(df_cycle, normalized_df_.loc[points, :].index, df_cycle.loc[points, :],
                                clusters_.loc[points])

            text_sel = str(len(normalized_df_.loc[points, :])) + "/" + str(len(normalized_df_))
            text_sel = text_sel + " points are Selected in the t-SNE plot."
            return text_sel, dcc.Graph(id="heatmap", figure=graph_hm, config={'modeBarButtonsToRemove': [
                'zoom2d', 'select2d', 'autoScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian']}), \
                dcc.Graph(id="graphPG", figure=graph_pg, config={'modeBarButtonsToRemove': [
                       'zoom2d', 'select2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian',
                       'hoverCompareCartesian']}), dash.no_update, dcc.Graph(id="pca", figure=graph_pca, config={
                        'modeBarButtonsToRemove': ['zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d',
                                                   'hoverClosestCartesian', 'hoverCompareCartesian']}), \
                dash.no_update, dash.no_update, switch_sel, dash.no_update, \
                dash.no_update, dash.no_update, dash.no_update
        except Exception as e:
            print(e)
            graph_hm = hm_graph(clusters_, normalized_df_)
            graph_pg = pg_graph(df_cycle, normalized_df_.index, df_cycle, clusters_)
            graph_pca = pca_graph(clusters_, normalized_df_.index, df_pca)
            return "All Points are Selected", dcc.Graph(id="heatmap", figure=graph_hm, config={
                'modeBarButtonsToRemove': ['zoom2d', 'select2d', 'autoScale2d', 'hoverClosestCartesian',
                                           'hoverCompareCartesian']}), \
                   dcc.Graph(id="graphPG", figure=graph_pg, config={'modeBarButtonsToRemove': [
                       'zoom2d', 'select2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian',
                       'hoverCompareCartesian']}), dash.no_update, \
                   dcc.Graph(id="pca", figure=graph_pca, config={'modeBarButtonsToRemove': [
                       'zoom2d', 'select2d', 'toggleSpikelines', 'autoScale2d', 'hoverClosestCartesian',
                       'hoverCompareCartesian']}), dash.no_update, dash.no_update, switch_sel, dash.no_update,\
                   dash.no_update, dash.no_update, dash.no_update
    else:
        raise PreventUpdate


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
    # FOR SERVER
    # app.run_server(host='0.0.0.0', port=5100, debug=True, use_reloader=False)
