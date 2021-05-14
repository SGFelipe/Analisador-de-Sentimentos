import joblib
import os
import pandas as pd
import nltk
import re
import string
import spacy

from nltk.corpus import stopwords


from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import FloatField, StringField
from wtforms.validators import DataRequired

from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
app.config['SECRET_KEY']='wP4xQ8hUljJ5oI1c'
bootstrap = Bootstrap(app)



class InputForm(FlaskForm):
    frase = StringField('Frase:', validators=[DataRequired()])

@app.route('/', methods=['GET', 'POST'])
def index():
    data = pd.read_csv('data\imdb-reviews-pt-br.zip')


    form = InputForm(request.form)
    specie = 'No-image'
    if form.validate_on_submit():
        x = [form.frase.data]
        vectorizer = CountVectorizer(binary=True, max_features = 5000)
        x_bow = vectorizer.fit_transform(data['text_pt'])
        X_bow = vectorizer.transform(x)
        #X_bow.toarray()

        specie = make_prediction(X_bow)
        
    return render_template('index.html', form=form, specie=specie)

def remove_stopwords (texto):
    pln = spacy.load('pt_core_news_sm')
    stopwords_pt = stopwords.words("portuguese")
    texto = re.sub(r"[\W\s]+", " ", texto)
    texto = [pal for pal in texto.split() if pal not in stopwords_pt]
    pln_texto = pln(" ".join(texto))
    tokens = [word.lemma_ if word.lemma_ != "-PRON-" else word.lower_ for word in pln_texto]
  
    return " ".join(tokens)

def make_prediction(x):    
    filename = os.path.join('model', 'finalized_model.sav')
    model = joblib.load(filename)
    return model.predict(x)[0]