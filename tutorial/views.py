from pyramid.view import (
    view_config,
    view_defaults
    )

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import re
model=load_model('C:/Users/dasso/Desktop/routing/models/model.h5')

import pickle
with open('C:/Users/dasso/Desktop/routing/models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def Predict(text):
    text=text.lower()
    text=re.sub('[^a-zA-z0-9\s]','',text)
    text = tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, maxlen=31, dtype='int32', value=0)
    sentiment = model.predict(text,batch_size=1,verbose = 0)[0][0]
    if sentiment<0.5:
        return("negative ---> Probability: {}".format(sentiment))
    else:
        return("positive ---> Probability:{}".format(sentiment))

@view_defaults(renderer='json')
class TutorialViews:
    def __init__(self, request):
        self.request = request

    @view_config(route_name='home')
    def home(self):
        # first = self.request.matchdict['first']
        first = self.request.GET['sentence']
        #last = self.request.matchdict['last']
        return {
            'Text': first,
            'Sentiment': Predict(first),
            #'last': last
        }
