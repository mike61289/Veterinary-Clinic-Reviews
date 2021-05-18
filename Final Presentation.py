import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from jupyter_dash import JupyterDash
import base64

hvoe = pd.read_csv('hvoe.csv')

mo = pd.read_csv('Montreal Ouest.csv')

passi = pd.read_csv('Passionimo.csv')

monkland = pd.read_csv('Monkland.csv')

csl = pd.read_csv('CSL.csv')

ahc = pd.read_csv('Animal Health Clinic.csv')

combined = pd.concat((hvoe, mo, passi, monkland, csl, ahc))

def generate_dataframe(dataframe, max_rows = 10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

hvoe_nlp_text = '''
### HVOE NLP

After cleaning up the dataframe, the elbow method was done to establish that 3 is the best number of clusters.
Through TFIDF, training the data, and logisitic regression, I was able to achieve a highest accuracy score of 82%.
'''

combined_nlp_text = '''
### Surrounding Vet Clinics NLP

After cleaning up the dataframe, the elbow method was done to establish that 3 is (again) the best number of clusters.
Through TFIDF, training the data, and logisitic regression, I was able to achieve a highest accuracy score of 80%.
'''

shap_text = '''
### Shapley Dataframes Summary

Specific words were chosen and the ratings associated with those words were written into dataframes for both the
Bird and Exotic Animal Hospital and for the vet clinics in the surrounding area.
One issue with these dataframes is that some words were said more often than others. To properly remove the 
Nans without removing data and without changing the results of the data, the average of words were taken individually
and the Nans were replaced with the mean rating associated with those words.
'''

hvoe_shap = pd.read_csv('HVOE Shapley.csv')
hvoe_shap = hvoe_shap.drop('Clinic', axis = 1)
combined_shap = pd.read_csv('Combined Shapley.csv')

#Graphs

hvoe_elbow = 'Elbow.png'
encoded_elbow = base64.b64encode(open(hvoe_elbow, 'rb').read()).decode()

hvoe_bar_shap = 'HVOE Bar Shap.png'
encoded_hvoe_bar_shap = base64.b64encode(open(hvoe_bar_shap, 'rb').read()).decode()

hvoe_shap_png = 'HVOE Shap.png'
encoded_hvoe_shap = base64.b64encode(open(hvoe_shap_png, 'rb').read()).decode()

combined_bar_shap = 'Combined Bar Shap.png'
encoded_combined_bar_shap = base64.b64encode(open(combined_bar_shap, 'rb').read()).decode()

combined_shap_png = 'Combined Shap.png'
encoded_combined_shap = base64.b64encode(open(combined_shap_png, 'rb').read()).decode()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

app.layout = html.Div([
    
    html.H1('Analysis of the Google Reviews of Vet Clinics',
           style = {'text-align' : 'center'}),
    
    html.Div(children='''
        Bird and Exotic Animal Hospital: Average rating = 4.5
    ''', style={
            'textAlign': 'center'}),
    
    generate_dataframe(hvoe),
    
    html.Img(src='data:image/png;base64,{}'.format(encoded_elbow)),
    
    dcc.Markdown(children = hvoe_nlp_text, style={
            'textAlign': 'center'}),
    
    html.Div(children='''
        Ratings of Vet Clinics in Surrounding Area: Average rating = 4.5
    ''', style={
            'textAlign': 'center', 'marginTop': '2em'}),
    
    generate_dataframe(combined),
    
    html.Img(src='data:image/png;base64,{}'.format(encoded_elbow)),
    
    dcc.Markdown(children = combined_nlp_text, style={
            'textAlign': 'center'}),
    
     html.Div(children='''
        Words and Associated Ratings at the Bird and Exotic Animal Hospital
    ''', style={
            'textAlign': 'center', 'marginTop': '2em'}),
    
    generate_dataframe(hvoe_shap),
    
    html.Img(src='data:image/png;base64,{}'.format(encoded_hvoe_bar_shap)),
    
    html.Img(src='data:image/png;base64,{}'.format(encoded_hvoe_shap)),
    
     html.Div(children='''
        Words and Associated Ratings at the Clinics in the Surrounding Area
    ''', style={
            'textAlign': 'center', 'marginTop': '2em'}),
    
    generate_dataframe(combined_shap),
    
    html.Img(src='data:image/png;base64,{}'.format(encoded_combined_bar_shap)),
    
    html.Img(src='data:image/png;base64,{}'.format(encoded_combined_shap)),
    
    dcc.Markdown(children = shap_text, style={
            'textAlign': 'center'}),
])

if __name__ == '__main__':
    app.run_server(debug = True)