## Chapter 20 Text Mining

**Original Code Credit:**: Shmueli, Galit; Bruce, Peter C.; Gedeck, Peter; Patel, Nitin R.. Data Mining for Business Analytics Wiley.

*Modifications* have been made from the original textbook examples due to version changes in library dependencies and/or for clarity.

Download this notebook and data [**here**](https://github.com/dcyoung23/msba511/tree/main/resources/examples).

### Import Libraries


```python
import os
from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import nltk
from nltk import word_tokenize
from nltk.stem.snowball import EnglishStemmer
import matplotlib.pylab as plt
from dmba import printTermDocumentMatrix, classificationSummary, liftChart

import matplotlib

%matplotlib inline

# download data required for NLTK
#nltk.download('punkt')
#nltk.download('punkt_tab')
```

### 20.2 The Tabular Representation of Text: Term-Document Matrix and “Bag-of-Words”


```python
text = ['this is the first sentence.',
        'this is a second sentence.',
        'the third sentence is here.']

# learn features based on text
count_vect = CountVectorizer()
counts = count_vect.fit_transform(text)

printTermDocumentMatrix(count_vect, counts)
```

              S1  S2  S3
    first      1   0   0
    here       0   0   1
    is         1   1   1
    second     0   1   0
    sentence   1   1   1
    the        1   0   1
    third      0   0   1
    this       1   1   0
    

### 20.4 Preprocessing the Text


```python
text = ['this is the first     sentence!!',
        'this is a second Sentence :)',
        'the third sentence, is here ',
        'forth of all sentences']

# Learn features based on text. Special characters are excluded in the analysis
count_vect = CountVectorizer()
counts = count_vect.fit_transform(text)

printTermDocumentMatrix(count_vect, counts)
```

               S1  S2  S3  S4
    all         0   0   0   1
    first       1   0   0   0
    forth       0   0   0   1
    here        0   0   1   0
    is          1   1   1   0
    of          0   0   0   1
    second      0   1   0   0
    sentence    1   1   1   0
    sentences   0   0   0   1
    the         1   0   1   0
    third       0   0   1   0
    this        1   1   0   0
    


```python
# Learn features based on text. Include special characters that are part of a word
# in the analysis
count_vect = CountVectorizer(token_pattern='[a-zA-Z!:)]+')
counts = count_vect.fit_transform(text)

printTermDocumentMatrix(count_vect, counts)
```

                S1  S2  S3  S4
    :)           0   1   0   0
    a            0   1   0   0
    all          0   0   0   1
    first        1   0   0   0
    forth        0   0   0   1
    here         0   0   1   0
    is           1   1   1   0
    of           0   0   0   1
    second       0   1   0   0
    sentence     0   1   1   0
    sentence!!   1   0   0   0
    sentences    0   0   0   1
    the          1   0   1   0
    third        0   0   1   0
    this         1   1   0   0
    


```python
stopWords = list(sorted(ENGLISH_STOP_WORDS))
ncolumns = 6; nrows= 30

print('First {} of {} stopwords'.format(ncolumns * nrows, len(stopWords)))
for i in range(0, len(stopWords[:(ncolumns * nrows)]), ncolumns):
    print(''.join(word.ljust(13) for word in stopWords[i:(i+ncolumns)]))
```

    First 180 of 318 stopwords
    a            about        above        across       after        afterwards   
    again        against      all          almost       alone        along        
    already      also         although     always       am           among        
    amongst      amoungst     amount       an           and          another      
    any          anyhow       anyone       anything     anyway       anywhere     
    are          around       as           at           back         be           
    became       because      become       becomes      becoming     been         
    before       beforehand   behind       being        below        beside       
    besides      between      beyond       bill         both         bottom       
    but          by           call         can          cannot       cant         
    co           con          could        couldnt      cry          de           
    describe     detail       do           done         down         due          
    during       each         eg           eight        either       eleven       
    else         elsewhere    empty        enough       etc          even         
    ever         every        everyone     everything   everywhere   except       
    few          fifteen      fifty        fill         find         fire         
    first        five         for          former       formerly     forty        
    found        four         from         front        full         further      
    get          give         go           had          has          hasnt        
    have         he           hence        her          here         hereafter    
    hereby       herein       hereupon     hers         herself      him          
    himself      his          how          however      hundred      i            
    ie           if           in           inc          indeed       interest     
    into         is           it           its          itself       keep         
    last         latter       latterly     least        less         ltd          
    made         many         may          me           meanwhile    might        
    mill         mine         more         moreover     most         mostly       
    move         much         must         my           myself       name         
    namely       neither      never        nevertheless next         nine         
    no           nobody       none         noone        nor          not          
    


```python
# Create a custom tokenizer that will use NLTK for tokenizing and lemmatizing 
# (removes interpunctuation and stop words)
class LemmaTokenizer(object):
    def __init__(self):
        self.stemmer = EnglishStemmer()
        self.stopWords = set(ENGLISH_STOP_WORDS)

    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc) 
                if t.isalpha() and t not in self.stopWords]

# Learn features based on text
count_vect = CountVectorizer(tokenizer=LemmaTokenizer(), token_pattern=None)
counts = count_vect.fit_transform(text)

printTermDocumentMatrix(count_vect, counts)
```

             S1  S2  S3  S4
    forth     0   0   0   1
    second    0   1   0   0
    sentenc   1   1   1   1
    


```python
# Apply CountVectorizer and TfidfTransformer sequentially
count_vect = CountVectorizer()
tfidfTransformer = TfidfTransformer(smooth_idf=False, norm=None)
counts = count_vect.fit_transform(text)
tfidf = tfidfTransformer.fit_transform(counts)

printTermDocumentMatrix(count_vect, tfidf)
```

                     S1        S2        S3        S4
    all        0.000000  0.000000  0.000000  2.386294
    first      2.386294  0.000000  0.000000  0.000000
    forth      0.000000  0.000000  0.000000  2.386294
    here       0.000000  0.000000  2.386294  0.000000
    is         1.287682  1.287682  1.287682  0.000000
    of         0.000000  0.000000  0.000000  2.386294
    second     0.000000  2.386294  0.000000  0.000000
    sentence   1.287682  1.287682  1.287682  0.000000
    sentences  0.000000  0.000000  0.000000  2.386294
    the        1.693147  0.000000  1.693147  0.000000
    third      0.000000  0.000000  2.386294  0.000000
    this       1.693147  1.693147  0.000000  0.000000
    

### 20.6 Example: Online Discussions on Autos and Electronics


```python
# Step 1: import and label records
corpus = []
label = []
with ZipFile(os.path.join('data', 'AutoAndElectronics.zip')) as rawData:
    for info in rawData.infolist():
        if info.is_dir(): 
            continue
        label.append(1 if 'rec.autos' in info.filename else 0)
        corpus.append(rawData.read(info))

# Step 2: preprocessing (tokenization, stemming, and stopwords)
class LemmaTokenizer(object):
    def __init__(self):
        self.stemmer = EnglishStemmer()
        self.stopWords = set(ENGLISH_STOP_WORDS)
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc) 
                if t.isalpha() and t not in self.stopWords]

preprocessor = CountVectorizer(tokenizer=LemmaTokenizer(), encoding='latin1', token_pattern=None)
preprocessedText = preprocessor.fit_transform(corpus)

# Step 3: TF-IDF and latent semantic analysis
tfidfTransformer = TfidfTransformer()
tfidf = tfidfTransformer.fit_transform(preprocessedText)

# Extract 20 concepts using LSA ()
svd = TruncatedSVD(20)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

lsa_tfidf = lsa.fit_transform(tfidf)
```


```python
# split dataset into 60% training and 40% test set
Xtrain, Xtest, ytrain, ytest = train_test_split(lsa_tfidf, label, test_size=0.4,
                                                random_state=42)

# run logistic regression model on training
logit_reg = LogisticRegression(solver='lbfgs')
logit_reg.fit(Xtrain, ytrain)

# print confusion matrix and accuracty
classificationSummary(ytest, logit_reg.predict(Xtest))
```

    Confusion Matrix (Accuracy 0.9600)
    
           Prediction
    Actual   0   1
         0 389   8
         1  24 379
    


```python

```
