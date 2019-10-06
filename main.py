import nltk # NLP
import feedparser
import re # Regex

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors # Zum Laden des Modells

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

for url in feed_urls:
  f = feedparser.parse(url)
  # print('')
  feed_title = f['feed']['title']
  # print(feed_title)
  for entry in f['entries']:
    entry_title = entry['title']
    # print(entry_title)

    entry_summary = re.sub('<[^<]+?>', '', entry['summary'])
    # print(entry_summary)
    # print('')

    corpus_array.append({
      'feed': feed_title, 
      'entry': entry_title + ': ' + entry_summary
    })
# print(corpus_array)

# Umlaute entfernen, lowercase
def deUmlaut(value):
  value = value.toLowerCase()
  value = re.sub('/ä/g', 'ae', value)
  value = re.sub('/ö/g', 'oe', value)
  value = re.sub('/ü/g', 'ue', value)
  value = re.sub('/Ä/g', 'Ae', value)
  value = re.sub('/Ö/g', 'Oe', value)
  value = re.sub('/Ü/g', 'Ue', value)
  value = re.sub('/ß/g', 'ss', value)
  return value

# Modell im C BIN-Format laden
print('Lade Word2Vec-Modell…')
model = KeyedVectors.load_word2vec_format('german.model', binary=True)
print('OK.')

tokenizer = RegexpTokenizer(r'\w+')
stopwords = set(stopwords.words('german'))
for entry_dict in corpus_array:
  sentence = tokenizer.tokenize(entry_dict['entry'])
  # Stopwords entfernen
  sentence[:] = (word for word in sentence if word.lower() not in stopwords)
  
  vectors = []
  for word in sentence:
    if word in model:
      vectors.append(model[word])

  print(vectors)
  break


