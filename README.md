# Datenanalyse mit Python: Clustering von RSS-Feeds

In diesem Projekt sollen Artikel der Politikressorts diverser deutscher Online-Zeitungen nach Themen geclustert werden.

## Voraussetzungen
Die Paketabhängigkeiten sind in `requirements.txt` aufgeführt. Diese können mit `pip install -r requirements.txt` installiert werden. Zusätzlich wird ein [deutsches Word2Vec-Modell](http://cloud.devmount.de/d2bc5672c523b086) als `german.model` benötigt.

## Vorgehen
Zunächst werden die Feed-URLs jeweils heruntergeladen und von *feedparser* verarbeitet. Dann werden HTML-Tags aus den Einträgen entfernt. Nach der Tokenisierung werden Umlaute umgewandelt und Stopwords entfernt. 

Danach wird für jedes Wort im Feed-Eintrag mit *Word2Vec* ein Wortvektor gebildet. Aus allen Vektoren der Wörter im Satz wird dann ein Durchschnittsvektor gebildet. 

Anschließend erfolgt Clustering mit [OPTICS](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html). Alles wird in einem Dataframe gesammelt, das dann nach den von OPTICS vergebenen Labels sortiert wird und die Feed-Einträge ausgibt.

## Optimierungsmöglichkeiten
Die Bereinigung der Feed-Einträge könnte sauberer sein, um Elemente zu entfernen, die mit dem Eintrag nichts zu tun haben. Außerdem könnte ein Xi-Wert für das Clustering automatisiert ermittelt werden.
