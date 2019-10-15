import nltk # NLP
import feedparser
import re # Regex

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors # Zum Laden des Modells
import pandas as pd
from sklearn.cluster import OPTICS

# Feed URLs
# Politik
feed_urls = [
  'http://newsfeed.zeit.de/politik/index',
  'http://rss.sueddeutsche.de/rss/Politik',
  'http://www.tagesschau.de/xml/rss2',
  'https://www.faz.net/rss/aktuell/politik/',
  'https://www.tagesspiegel.de/politik/rss',
  'https://taz.de/!p4615;rss/'
]

corpus_array = []

print('Lade und parse Feed URLs… ', end='')
try:
  for url in feed_urls:
    f = feedparser.parse(url)
    feed_title = f['feed']['title']
    for entry in f['entries']:
      entry_title = entry['title']
      # HTML-Tags entfernen
      entry_summary = re.sub('<[^<]+?>', '', entry['summary'])

      corpus_array.append({
        'feed': feed_title, 
        'entry': entry_title + ': ' + entry_summary
      })
  
  print('OK.')
except:
  print('FAIL.')

df = pd.DataFrame.from_dict(corpus_array)

# Umlaute entfernen, lowercase
def deUmlaut(value):
  value = re.sub('/ä/g', 'ae', value)
  value = re.sub('/ö/g', 'oe', value)
  value = re.sub('/ü/g', 'ue', value)
  value = re.sub('/Ä/g', 'Ae', value)
  value = re.sub('/Ö/g', 'Oe', value)
  value = re.sub('/Ü/g', 'Ue', value)
  value = re.sub('/ß/g', 'ss', value)
  return value

# Modell im C BIN-Format laden
print('Lade Word2Vec-Modell… ', end='')
try:
  model = KeyedVectors.load_word2vec_format('german.model', binary=True)
  print('OK.')
except:
  print('FAIL.')

entry_vectors = []
tokenizer = RegexpTokenizer(r'\w+')
stopwords = set(stopwords.words('german'))
for entry in df['entry']:
  entry = deUmlaut(entry)
  sentence = tokenizer.tokenize(entry)
  # Stopwords entfernen
  sentence[:] = (word for word in sentence if word.lower() not in stopwords)
  
  vectors = []
  for word in sentence:
    if word in model:
      vectors.append(model[word])

  df_vectors = pd.DataFrame(vectors)
  # Wortweise Durchschnitt bilden, sodass der ganze Satz einen einzigen "Durchschnitts-Wortvektor" erhält
  mean_vector = df_vectors.mean(axis=0).values.tolist()

  entry_vectors.append(mean_vector)

df['vector'] = entry_vectors

# Clustering
xi = .07
clust = OPTICS(min_samples=2, xi=xi)
labels = clust.fit_predict(entry_vectors)
df['label'] = labels

pd.set_option('display.max_colwidth', -1) # Lange Strings

# Spalten wählen
df = df.filter(items=['label', 'feed', 'entry'])
# Unkategorisierte Zeilen weglassen
df = df[df['label'] >= 0]
# Sortieren
df = df.sort_values(by='label')

print(df.to_string())
