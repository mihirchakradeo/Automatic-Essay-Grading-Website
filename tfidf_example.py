# Output a representative example of how we vectorized text

def toMarkDownTable(vect, sentences, sentenceOne):
    # Export a markdown table of the vectorizer output
    
    # Fit
    wordVec = vect.fit_transform(sentences)
    # wordVec = vect.transform(sentenceOne)

    vectWords = []
    vectCounts = []
    dividers = []

    for word in vect.vocabulary_:
        vectWords += '|%s'%word
        dividers += '|---'
    vectWords += '|'
    dividers += '|'

    for count in np.array(wordVec.todense()[0,:]).reshape(-1):
        vectCounts += '|%4.3f'%count
    vectCounts += '|'

    # Print out the results
    print ''.join(vectWords)
    print ''.join(dividers)
    print ''.join(vectCounts)



# Import
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# exampleSentence = 'Did you know that more and more people these days are depending on computers for their safety, natural education, and their social life?'
# sentenceTwo = 'Initially, ones safety while using a computer is at risk.'

exampleSentence = 'This is the first sentence.'
sentenceTwo = 'This is also a sentence.'

# vect = TfidfVectorizer()
vect = CountVectorizer()

wordVec = vect.fit_transform([exampleSentence, exampleSentence]).todense()
vect.vocabulary_

vectWords = []
vectCounts = []
dividers = []

for word in vect.vocabulary_:
    vectWords += '|%s'%word
    dividers += '|---'
vectWords += '|'
dividers += '|'

# print wordVec[0,:][0].reshape(-1)

# for count in wordVec[1,:]:
    # vectCounts += '|%d'%count
    # print count
vectCounts += '|'



    # vectWords += word
    # vectWords += '|'

# print ''.join(vectWords)
# print ''.join(dividers)
# print ''.join(vectCounts)

# print '|%s|'%vect.vocabulary_['and']

# vect2 = TfidfVectorizer()
# print vect2.fit_transform([exampleSentence, sentenceTwo]).todense()

# print vect.vocabulary_
# 

toMarkDownTable(CountVectorizer(), [exampleSentence, sentenceTwo], exampleSentence)

print '\n'

toMarkDownTable(TfidfVectorizer(), [exampleSentence, sentenceTwo], exampleSentence)



