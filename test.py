# %%
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# %%

# Example list of documents
documents = [
    'Emotional Rescue is the 15th British and 17th American studio album by English rock band the Rolling Stones, released on 20 June 1980 by Rolling Stones Records. Following the success of their previous album, Some Girls, their biggest hit to date, the Rolling Stones returned to the studio in early 1979 to start writing and recording its follow-up. Full-time members Mick Jagger (vocals), Keith Richards (guitar), Ronnie Wood (guitar), Bill Wyman (bass) and Charlie Watts (drums) were joined by frequent collaborators Ian Stewart (keyboards), Nicky Hopkins (keyboards), Bobby Keys (saxophone) and Sugar Blue (harmonica).Upon release, the album topped the charts in at least six countries, including the United States, UK, and Canada. Hit singles from it include the title track, which reached No. 1 in Canada, No. 3 in the United States, and No. 9 in the UK and a top-40 single in several countries. The recording sessions for Emotional Rescue were so productive that several tracks left off the album would form the core of the follow-up, 1981s Tattoo You.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
# %%
ids = np.array(documents[0].split(' '))
# %%
print(len(ids))
# %%
# Initialize a TfidfVectorizer object with custom options
vectorizer = TfidfVectorizer(max_features=50000)
# Fit and transform the documents into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(ids)
# %%
print(tfidf_matrix[1][0])

# %%
# Retrieve the vocabulary
vocabulary = vectorizer.get_feature_names_out()
idf = vectorizer.idf_
tf_matrix = tfidf_matrix / np.reshape(idf, (1, -1))
# Retrieve the original text with stop words included
original_text = []
for i in range(len(ids)):
    row = np.asarray(tf_matrix[i,:]).squeeze()
    words = []
    for j in np.where(row > 0)[0]:
        words.append(vocabulary[j])
    text = ' '.join(words)
    original_text.append(text)
# Reconstruct the original text
for text in original_text:
    print(text)
# %%
