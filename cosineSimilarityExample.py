# Import
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer

def printCosSim(wordMat, idx1, idx2, corpus):
    # Cosine similarity bewteen two sentences

    a = wordMat[idx1, :]
    b = wordMat[idx2, :].T

    cosineSim = np.dot(a, b)[0, 0] / \
        (np.linalg.norm(a, 2) * np.linalg.norm(b, 2))

    # return cosineSim
    print '**"%s"** vs. **"%s"**' % (corpus[idx1], corpus[idx2])
    # print '\n$$\cos\\theta = %4.2f$$\n' % cosineSim
    print '\n$$\mathrm{similarity} = %4.2f$$\n' % cosineSim


# Sentences to use
sentence1 = 'This is the first sentence.'
# sentence2 = 'This is the second sentence.'
sentence2 = 'This intelligent second sentence is related to the first sentence.'
# sentence3 = "A completely unrelated set of words will give a score of zero."
sentence3 = "Completely unrelated nonsense scores zero."

corpus = [sentence1, sentence2, sentence3]


# Fit the vectorizer
vect = TfidfVectorizer()
# vect = CountVectorizer()
wordMat = vect.fit_transform(corpus).todense()

printCosSim(wordMat, 0, 1, corpus)
printCosSim(wordMat, 0, 2, corpus)
printCosSim(wordMat, 0, 0, corpus)

# print '%s vs. %s:\n%4.3f'

# print np.dot(wordMat[0,:], wordMat[2,:].T)[0,0]
