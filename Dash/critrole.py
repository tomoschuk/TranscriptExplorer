# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.plotly as py
import plotly.figure_factory as ff
from sklearn.feature_extraction.text import TfidfVectorizer


app = dash.Dash(__name__)
server = app.server
app.title = "Transcript Explorer"

data = pd.read_pickle('cleandata.pk1')

episodemarks = {str(episode): str(episode) for episode in range(data['episode'].min()+4,data['episode'].max(), 5)}
episodemarks[str(data['episode'].min())] = str(data['episode'].min())
episodemarks[str(data['episode'].max())] = str(data['episode'].max())


app.layout = html.Div(children=[
    html.Div([html.H1(children='Critical Role Transcript Explorer')]),

    html.Div([html.Div(children =
        "Welcome to the Critical Role transcript explorer! Critical Role is a populuar vodcast in which voice actors play Dungeons & Dragons (check them out on critrole.com). Because D&D is a game that takes place almost entirely through language, I thought it'd be a cool place to help people explore language patterns! Enter any number of words or phrases separated by a comma to see how frequently each member of the cast said that word or phrase. Also, use the slider below the graph to filter the episodes over which the graph is generated. Submitting just one word or one speaker will let you compare on a single graph."
    )]),
    html.Br(),
    html.Div([
        html.Div(children = "Speaker(s):"),
        dcc.Input(id='speaker_input', type='text', value='Sam, Laura, Travis, Matt, Liam, Marisha, Taliesin, Ashley', size = 60, minlength = 1),
        html.Br(),
        html.Br(),
        html.Div(children = "Word(s):"),
        dcc.Input(id='word_input', type='text', value='Bigby, Trinket, rage, dagger, Elemental, Bad News, Sarenrae', size = 60, minlength = 1),
        html.Br(),
        html.Br(),
        html.Button(id='submit_button', n_clicks = 0, children='Update')]),
    
        html.Div([dcc.Graph(id='fig')],
            style = {'display':'block'}),

        html.Div(children = "Episode Filter:",
        style={'width': '80%', 'margin': 'auto'}),
        html.Div([dcc.RangeSlider(
            id = 'episode_slider',
            min = data['episode'].min(),
            max = data['episode'].max(),
            value = [data['episode'].min(),data['episode'].max()],
            step = 1,
            marks = episodemarks
            )],
        style={'width': '80%', 'margin': 'auto', 'text-align':'center'}),
        html.Br(),
        html.Br(),
        html.Div([html.Div(children =
        '* As calculated by TF-IDF (Term Frequency - Inverse Document Frequency) - a metric that tells you how distinguishing a word is among a set of words and speakers. It gets HIGHER as a certain person says the word, but LOWER when everyone else says the word. So words like "ok" will not have high scores, even when someone says them a lot... but words that one person uses a lot could have very high scores! Enjoy exploring!'
        )]),
        html.Div([html.Div(children =
        "Below is a link to a walkthrough of the code used used to turn the raw subtitle files into this graph, as well as the github repo containing all of the data and scripts. This script can be easily adapted for use on any folder of txt subtitle files."
        )]),
        html.Br(),
        html.Div([html.A('Walkthrough', href='http://acsweb.ucsd.edu/~btomosch/critrole.html')]),
        html.Div([html.A('Github', href='https://github.com/tomoschuk/TranscriptExplorer')]),
        html.Br(),
        html.Br(),
        html.Br()
        ])









@app.callback(
    dash.dependencies.Output('fig', 'figure'),
    [dash.dependencies.Input('submit_button', 'n_clicks'),
    dash.dependencies.Input('episode_slider','value')],
    [dash.dependencies.State('word_input','value'),
    dash.dependencies.State('speaker_input','value')]
    )

def update_figure(n_clicks, episode_slider, word_input, speaker_input):

    fullmerge = {}
    data_subset = data[data['episode'] >= episode_slider[0]][data['episode'] <= episode_slider[1]]
    #Merge all episodes into one long string
    peeps = list(filter(None,speaker_input.upper().replace(" ","").split(",")))
    for peep in peeps:
        fullmerge.update({peep:' '.join(data_subset[data_subset.Speaker == peep]['cleaned'])})
    finaldata = pd.DataFrame.from_dict(fullmerge, orient = 'index').reset_index().rename(columns = {'index':'speaker',0:'text'})


    totalwords = pd.DataFrame()
    for peep in peeps:
        totalwords = totalwords.append(pd.Series([peep, ' ', finaldata[finaldata['speaker'] == peep].text.str.count(' ').iloc[0]]), ignore_index = True)
    totalwords = totalwords.rename(index=str, columns={0: "speaker", 1: "word",2: "total"})




    words = list(filter(None,word_input.lower().split(",")))
    #Calculate frequency of words in finaldata
    df = pd.DataFrame()
    for peep in peeps:
        for word in words:
            df = df.append(pd.Series([peep, word, finaldata[finaldata['speaker'] == peep].text.str.count(word).iloc[0]]), ignore_index = True)
    df = df.rename(index=str, columns={0: "speaker", 1: "word",2: "amount"})


    #Calculate rate per 1000 words
    df = pd.merge(df, totalwords[['speaker','total']], on='speaker')
    df['Number of times said per 1000 words'] = (df['amount']/df['total'])*1000

    #Sort data by rate and then speaker
    df = df.sort_values(by=['Number of times said per 1000 words','speaker'], ascending = [True,False])
    

    #Graph
    if len(words) != 1:
        if len(peeps) != 1:
            #Apply the tfidf function, and find the most "distinguishing" word among the given words and speakers
            tfidf = TfidfVectorizer(stop_words='english', vocabulary = words)
            tfs = tfidf.fit_transform(finaldata['text'])
            matrix = pd.DataFrame(tfs.todense(), index = peeps, columns = tfidf.get_feature_names()).transpose()
            matrix['word'] = matrix.index
            matrix = pd.melt(matrix, id_vars = 'word')
            matrix = matrix.rename(index=str, columns={'value': "tfidf",'variable': "speaker"})
            distWord = matrix.loc[matrix['tfidf'].idxmax()]['word']
            distSpeaker = matrix.loc[matrix['tfidf'].idxmax()]['speaker']
            tfidfSent = ("Most distinguishing: '" + distWord + "' by " + distSpeaker + ".*")

            fig = ff.create_facet_grid(
                df,
                x='Number of times said per 1000 words',
                y='word',
                facet_col='speaker',
                color_name='speaker',
                trace_type='bar',
                orientation = 'h',
                scales = 'free',
                width = 1200
            )
            for i in range(len(peeps)+1):
                if i == 0:
                    fig.layout.xaxis.update({'range': [df['Number of times said per 1000 words'].min(), (df['Number of times said per 1000 words'].max()+(.15 * df['Number of times said per 1000 words'].max()))]})
                else:
                    exec('fig.layout.xaxis' + str(i)+".update({'range': [df['Number of times said per 1000 words'].min(), (df['Number of times said per 1000 words'].max()+(.15 * df['Number of times said per 1000 words'].max()))]})")
            fig.layout.xaxis.title = tfidfSent
            fig.layout.update(plot_bgcolor='rgba(230,230,230,90)')

        elif len(peeps) == 1: 
            fig = ff.create_facet_grid(
            df,
            x='word',
            y='Number of times said per 1000 words',
            color_name='word',
            trace_type='bar',
            scales = 'free',
            width = 1200
            )
            fig.layout.update(plot_bgcolor='rgba(230,230,230,90)')
    elif len(words) == 1:
        fig = ff.create_facet_grid(
            df,
            x='speaker',
            y='Number of times said per 1000 words',
            color_name='speaker',
            trace_type='bar',
            scales = 'free',
            width = 1200
        )
        fig.layout.update(plot_bgcolor='rgba(230,230,230,90)')
    return {
    'data': fig

    }





if __name__ == '__main__':
    app.run_server(debug=True)