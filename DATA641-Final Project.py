#!/usr/bin/env python
# coding: utf-8

# ##### Important Packages and Libraries

# In[113]:


pip install git+https://github.com/ClaudeCoulombe/FrenchLefffLemmatizer.git &> /dev/null
pip install datasets transformers


# In[1]:


import nltk
from datasets import load_dataset
import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import pandas as pd
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from datasets import Dataset, concatenate_datasets, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import random
from sklearn.metrics import f1_score
import os
from huggingface_hub import HfApi, HfFolder
from tqdm.notebook import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB


token = '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'
os.environ["HUGGINGFACE_TOKEN"] = token

api = HfApi()
folder = HfFolder()
folder.save_token(token)
user = api.whoami()
print(user)


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_ru')


# #### Loading and Preprocessing the Data

# In[2]:


# Define a function to clean English text
def clean_english_text(text):
    if isinstance(text, dict):
        values = list(text.values())
        text = ' '.join(str(val) for val in values)
    # Split the text into sentences using NLTK's sentence tokenization
    sentences = nltk.sent_tokenize(text.lower())
    # Initialize lemmatizer object
    lemmatizer = WordNetLemmatizer()
    cleaned_sentences = []
    for sentence in sentences:
        # Split the sentence into words
        words = nltk.word_tokenize(sentence)
        # Remove stopwords using NLTK's pre-defined list for English
        stopwords = set(nltk.corpus.stopwords.words('english'))
        words = [word for word in words if word not in stopwords]
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(word) for word in words]
        # Lemmatize the cleaned words using WordNetLemmatizer
        lem_words = [lemmatizer.lemmatize(word) for word in words]
        # Remove non-alphanumeric characters and extra whitespace
        lem_words = [re.sub('\W+', ' ', word) for word in lem_words]
        lem_words = [re.sub('\s+', ' ', word).strip() for word in lem_words]
        # Remove digits
        lem_words = [re.sub(r'\d+', '', word) for word in lem_words]
        # Remove punctuation
        lem_words = [re.sub(r'[^\w\s]', '', word) for word in lem_words]
        # Join the cleaned and lemmatized words back into a string
        cleaned_sentence = ' '.join(lem_words)
        cleaned_sentences.append(cleaned_sentence)
    # Join the cleaned sentences back into a string
    cleaned_text = ' '.join(cleaned_sentences)
    return cleaned_text

# Define a function to map star ratings to labels
def map_stars(star_rating):
    if star_rating in [4, 5]:
        return "1"
    elif star_rating == 3:
        return "0"
    else:
        return "-1"

# Remove "en_" prefix from review IDs
def remove_en(review_id):
    return review_id.replace("en_", "")

english_datasets = load_dataset("amazon_reviews_multi", "en")
english_datasets = english_datasets.remove_columns(["product_category", "language", "review_title", "product_id", "reviewer_id"])
review_body = pd.DataFrame({'review_body': english_datasets["train"]["review_body"]}).applymap(clean_english_text)
label = pd.DataFrame({'label': english_datasets["train"]["stars"]}).applymap(map_stars).astype(int)
review_id = pd.DataFrame({'review_id': english_datasets["train"]["review_id"]}).applymap(remove_en)
english_datasets["train"] = pd.concat([review_body, label, review_id], axis=1)


# In[3]:


# Define a function to clean French text
def clean_french_text(text):
    if isinstance(text, dict):
        values = list(text.values())
        text = ' '.join(str(val) for val in values)
    # Split the text into sentences using NLTK's sentence tokenization
    sentences = nltk.sent_tokenize(text.lower())
    # Initialize lemmatizer object
    lemmatizer = WordNetLemmatizer()
    cleaned_sentences = []
    for sentence in sentences:
        # Split the sentence into words
        words = nltk.word_tokenize(sentence)
        # Remove stopwords using NLTK's pre-defined list for French
        stopwords = set(nltk.corpus.stopwords.words('french'))
        words = [word for word in words if word not in stopwords]
        stemmer = SnowballStemmer('french')
        words = [stemmer.stem(word) for word in words]
        # Lemmatize the cleaned words using WordNetLemmatizer
        lem_words = [lemmatizer.lemmatize(word) for word in words]
        # Remove non-alphanumeric characters and extra whitespace
        lem_words = [re.sub('\W+', ' ', word) for word in lem_words]
        lem_words = [re.sub('\s+', ' ', word).strip() for word in lem_words]
        # Remove digits
        lem_words = [re.sub(r'\d+', '', word) for word in lem_words]
        # Remove punctuation
        lem_words = [re.sub(r'[^\w\s]', '', word) for word in lem_words]
        # Join the cleaned and lemmatized words back into a string
        cleaned_sentence = ' '.join(lem_words)
        cleaned_sentences.append(cleaned_sentence)
    # Join the cleaned sentences back into a string
    cleaned_text = ' '.join(cleaned_sentences)
    return cleaned_text

# Define a function to map star ratings to labels
def map_stars(star_rating):
    if star_rating in [4, 5]:
        return "1"
    elif star_rating == 3:
        return "0"
    else:
        return "-1"

# Remove "fr_" prefix from review IDs
def remove_fr(review_id):
    return review_id.replace("fr_", "")

french_datasets = load_dataset("amazon_reviews_multi", "fr")
french_datasets = french_datasets.remove_columns(["product_category", "language", "review_title", "product_id", "reviewer_id"])
review_body = pd.DataFrame({'review_body': french_datasets["train"]["review_body"]}).applymap(clean_french_text)
label = pd.DataFrame({'label': french_datasets["train"]["stars"]}).applymap(map_stars).astype(int)
review_id = pd.DataFrame({'review_id': french_datasets["train"]["review_id"]}).applymap(remove_fr)
french_datasets["train"] = pd.concat([review_body, label, review_id], axis=1)


# In[4]:


# Define a function to clean Spanish text
def clean_spanish_text(text):
    if isinstance(text, dict):
        values = list(text.values())
        text = ' '.join(str(val) for val in values)
    # Split the text into sentences using NLTK's sentence tokenization
    sentences = nltk.sent_tokenize(text.lower())
    # Initialize lemmatizer object
    lemmatizer = WordNetLemmatizer()
    cleaned_sentences = []
    for sentence in sentences:
        # Split the sentence into words
        words = nltk.word_tokenize(sentence)
        # Remove stopwords using NLTK's pre-defined list for Spanishish
        stopwords = set(nltk.corpus.stopwords.words('spanish'))
        words = [word for word in words if word not in stopwords]
        stemmer = SnowballStemmer('spanish')
        words = [stemmer.stem(word) for word in words]
        # Lemmatize the cleaned words using WordNetLemmatizer
        lem_words = [lemmatizer.lemmatize(word) for word in words]
        # Remove non-alphanumeric characters and extra whitespace
        lem_words = [re.sub('\W+', ' ', word) for word in lem_words]
        lem_words = [re.sub('\s+', ' ', word).strip() for word in lem_words]
        # Remove digits
        lem_words = [re.sub(r'\d+', '', word) for word in lem_words]
        # Remove punctuation
        lem_words = [re.sub(r'[^\w\s]', '', word) for word in lem_words]
        # Join the cleaned and lemmatized words back into a string
        cleaned_sentence = ' '.join(lem_words)
        cleaned_sentences.append(cleaned_sentence)
    # Join the cleaned sentences back into a string
    cleaned_text = ' '.join(cleaned_sentences)
    return cleaned_text

# Define a function to map star ratings to labels
def map_stars(star_rating):
    if star_rating in [4, 5]:
        return "1"
    elif star_rating == 3:
        return "0"
    else:
        return "-1"
    
# Remove "es_" prefix from review IDs
def remove_es(review_id):
    return review_id.replace("es_", "")

spanish_datasets = load_dataset("amazon_reviews_multi", "es")
spanish_datasets = spanish_datasets.remove_columns(["product_category", "language", "review_title", "product_id", "reviewer_id"])
review_body = pd.DataFrame({'review_body': spanish_datasets["train"]["review_body"]}).applymap(clean_spanish_text)
label = pd.DataFrame({'label': spanish_datasets["train"]["stars"]}).applymap(map_stars).astype(int)
review_id = pd.DataFrame({'review_id': spanish_datasets["train"]["review_id"]}).applymap(remove_es)
spanish_datasets["train"] = pd.concat([review_body, label, review_id], axis=1)


# In[5]:


# Define a function to clean German text
def clean_german_text(text):
    if isinstance(text, dict):
        values = list(text.values())
        text = ' '.join(str(val) for val in values)
    # Split the text into sentences using NLTK's sentence tokenization
    sentences = nltk.sent_tokenize(text.lower())
    # Initialize lemmatizer object
    lemmatizer = WordNetLemmatizer()
    cleaned_sentences = []
    for sentence in sentences:
        # Split the sentence into words
        words = nltk.word_tokenize(sentence)
        # Remove stopwords using NLTK's pre-defined list for German
        stopwords = set(nltk.corpus.stopwords.words('german'))
        words = [word for word in words if word not in stopwords]
        stemmer = SnowballStemmer('german')
        words = [stemmer.stem(word) for word in words]
        # Lemmatize the cleaned words using WordNetLemmatizer
        lem_words = [lemmatizer.lemmatize(word) for word in words]
        # Remove non-alphanumeric characters and extra whitespace
        lem_words = [re.sub('\W+', ' ', word) for word in lem_words]
        lem_words = [re.sub('\s+', ' ', word).strip() for word in lem_words]
        # Remove digits
        lem_words = [re.sub(r'\d+', '', word) for word in lem_words]
        # Remove punctuation
        lem_words = [re.sub(r'[^\w\s]', '', word) for word in lem_words]
        # Join the cleaned and lemmatized words back into a string
        cleaned_sentence = ' '.join(lem_words)
        cleaned_sentences.append(cleaned_sentence)
    # Join the cleaned sentences back into a string
    cleaned_text = ' '.join(cleaned_sentences)
    return cleaned_text

# Define a function to map star ratings to labels
def map_stars(star_rating):
    if star_rating in [4, 5]:
        return "1"
    elif star_rating == 3:
        return "0"
    else:
        return "-1"
    
# Remove "de_" prefix from review IDs
def remove_de(review_id):
    return review_id.replace("de_", "")

german_datasets = load_dataset("amazon_reviews_multi", "de")
german_datasets = german_datasets.remove_columns(["product_category", "language", "review_title", "product_id", "reviewer_id"])
review_body = pd.DataFrame({'review_body': german_datasets["train"]["review_body"]}).applymap(clean_spanish_text)
label = pd.DataFrame({'label': german_datasets["train"]["stars"]}).applymap(map_stars).astype(int)
review_id = pd.DataFrame({'review_id': german_datasets["train"]["review_id"]}).applymap(remove_de)
german_datasets["train"] = pd.concat([review_body, label, review_id], axis=1)


# #### Loading Train Data

# In[6]:


spanish_dataset_train = spanish_datasets["train"].sample(n=3400, random_state=123).reset_index(drop=True)
french_dataset_train = french_datasets["train"].sample(n=3400, random_state=123).reset_index(drop=True)
english_dataset_train = english_datasets["train"].sample(n=3400, random_state=123).reset_index(drop=True)
german_dataset_train = german_datasets["train"].sample(n=3400, random_state=123).reset_index(drop=True)


# In[7]:


spanish_dataset_train


# In[8]:


french_dataset_train


# In[9]:


english_dataset_train


# In[10]:


german_dataset_train


# In[11]:


X_es = spanish_dataset_train.review_body.values
y_es = spanish_dataset_train.label.values

X_train_es, X_val_es, y_train_es, y_val_es =\
    train_test_split(X_es, y_es, test_size=0.1, random_state=2020)

X_fr = french_dataset_train.review_body.values
y_fr = french_dataset_train.label.values

X_train_fr, X_val_fr, y_train_fr, y_val_fr =\
    train_test_split(X_fr, y_fr, test_size=0.1, random_state=2020)

X_en = english_dataset_train.review_body.values
y_en = english_dataset_train.label.values

X_train_en, X_val_en, y_train_en, y_val_en =\
    train_test_split(X_en, y_en, test_size=0.1, random_state=2020)

X_de = german_dataset_train.review_body.values
y_de = german_dataset_train.label.values

X_train_de, X_val_de, y_train_de, y_val_de =\
    train_test_split(X_de, y_de, test_size=0.1, random_state=2020)


# #### Loading Test Data

# In[12]:


spanish_dataset_test = spanish_datasets["train"].sample(n=4600, random_state=123).reset_index(drop=True).drop('label', axis=1)
french_dataset_test = french_datasets["train"].sample(n=4600, random_state=123).reset_index(drop=True).drop('label', axis=1)
english_dataset_test = english_datasets["train"].sample(n=4600, random_state=123).reset_index(drop=True).drop('label', axis=1)
german_dataset_test = german_datasets["train"].sample(n=4600, random_state=123).reset_index(drop=True).drop('label', axis=1)


# In[13]:


spanish_dataset_test


# In[14]:


french_dataset_test


# In[15]:


english_dataset_test


# In[16]:


german_dataset_test


# #### TF-IDF Vectorizer

# In[17]:


X_train_es_arr = np.array(X_train_es)
X_val_es_arr = np.array(X_val_es)

X_train_fr_arr = np.array(X_train_fr)
X_val_fr_arr = np.array(X_val_fr)

X_train_en_arr = np.array(X_train_en)
X_val_en_arr = np.array(X_val_en)

X_train_de_arr = np.array(X_train_de)
X_val_de_arr = np.array(X_val_de)


# In[18]:


# Calculate TF-IDF
tf_idf = TfidfVectorizer(ngram_range=(1, 3),
                         binary=True,
                         smooth_idf=False)


# In[19]:


X_train_tfidf_es = tf_idf.fit_transform(X_train_es_arr)
X_val_tfidf_es = tf_idf.transform(X_val_es_arr)

# Get feature names from the fitted vectorizer
feature_names = tf_idf.get_feature_names_out()

# Sum the values of each term across all the documents
tfidf_scores = X_train_tfidf_es.sum(axis=0).tolist()[0]

# Map feature names to their corresponding TF-IDF scores
features = list(zip(feature_names, tfidf_scores))
sorted_features = sorted(features, key=lambda x: x[1], reverse=True)

# Plot the top 20 features and their corresponding scores
top_features = sorted_features[:20]
x = [f[0] for f in top_features]
y = [f[1] for f in top_features]
plt.bar(x, y)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('TF-IDF Scores')
plt.show()


# In[20]:


X_train_tfidf_de = tf_idf.fit_transform(X_train_de_arr)
X_val_tfidf_de = tf_idf.transform(X_val_de_arr)

# Get feature names from the fitted vectorizer
feature_names = tf_idf.get_feature_names_out()

# Sum the values of each term across all the documents
tfidf_scores = X_train_tfidf_de.sum(axis=0).tolist()[0]

# Map feature names to their corresponding TF-IDF scores
features = list(zip(feature_names, tfidf_scores))
sorted_features = sorted(features, key=lambda x: x[1], reverse=True)

# Plot the top 20 features and their corresponding scores
top_features = sorted_features[:20]
x = [f[0] for f in top_features]
y = [f[1] for f in top_features]
plt.bar(x, y)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('TF-IDF Scores')
plt.show()


# In[21]:


X_train_tfidf_fr = tf_idf.fit_transform(X_train_fr_arr)
X_val_tfidf_fr = tf_idf.transform(X_val_fr_arr)

# Get feature names from the fitted vectorizer
feature_names = tf_idf.get_feature_names_out()

# Sum the values of each term across all the documents
tfidf_scores = X_train_tfidf_fr.sum(axis=0).tolist()[0]

# Map feature names to their corresponding TF-IDF scores
features = list(zip(feature_names, tfidf_scores))
sorted_features = sorted(features, key=lambda x: x[1], reverse=True)

# Plot the top 20 features and their corresponding scores
top_features = sorted_features[:20]
x = [f[0] for f in top_features]
y = [f[1] for f in top_features]
plt.bar(x, y)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('TF-IDF Scores')
plt.show()


# In[22]:


X_train_tfidf_en = tf_idf.fit_transform(X_train_en_arr)
X_val_tfidf_en = tf_idf.transform(X_val_en_arr)

# Get feature names from the fitted vectorizer
feature_names = tf_idf.get_feature_names_out()

# Sum the values of each term across all the documents
tfidf_scores = X_train_tfidf_en.sum(axis=0).tolist()[0]

# Map feature names to their corresponding TF-IDF scores
features = list(zip(feature_names, tfidf_scores))
sorted_features = sorted(features, key=lambda x: x[1], reverse=True)

# Plot the top 20 features and their corresponding scores
top_features = sorted_features[:20]
x = [f[0] for f in top_features]
y = [f[1] for f in top_features]
plt.bar(x, y)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('TF-IDF Scores')
plt.show()


# #### Random Forest Classifier

# In[23]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 100)  
  
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train_tfidf_es, y_train_es)
  
# performing predictions on the test dataset
y_pred_es = clf.predict(X_val_tfidf_es)
  
# metrics are used to find accuracy or error
from sklearn import metrics  
print()
  
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_val_es, y_pred_es))
cm = confusion_matrix(y_val_es, y_pred_es)
# plot the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# In[24]:


clf = RandomForestClassifier(n_estimators = 100)  
  
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train_tfidf_en, y_train_en)
  
# performing predictions on the test dataset
y_pred_en = clf.predict(X_val_tfidf_en)
  
# metrics are used to find accuracy or error
from sklearn import metrics  
print()
  
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_val_en, y_pred_en))
cm = confusion_matrix(y_val_en, y_pred_en)
# plot the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# In[25]:


clf = RandomForestClassifier(n_estimators = 100)  
  
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train_tfidf_fr, y_train_fr)
  
# performing predictions on the test dataset
y_pred_fr = clf.predict(X_val_tfidf_fr)
  
# metrics are used to find accuracy or error
from sklearn import metrics  
print()
  
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_val_fr, y_pred_fr))
cm = confusion_matrix(y_val_fr, y_pred_fr)
# plot the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# In[26]:


clf = RandomForestClassifier(n_estimators = 100)  
  
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train_tfidf_de, y_train_de)
  
# performing predictions on the test dataset
y_pred_de = clf.predict(X_val_tfidf_de)
  
# metrics are used to find accuracy or error
from sklearn import metrics  
print()
  
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_val_de, y_pred_de))
cm = confusion_matrix(y_val_de, y_pred_de)
# plot the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# ##### Naive Bayes Classifier

# In[27]:


nb = MultinomialNB()


# In[28]:


get_ipython().run_line_magic('time', 'nb.fit(X_train_tfidf_es, y_train_es)')

# Make predictions on the validation data
y_pred_es = nb.predict(X_val_tfidf_es)
accuracy_es = metrics.accuracy_score(y_val_es, y_pred_es)
print("Accuracy:", accuracy_es)

cm = confusion_matrix(y_val_es, y_pred_es)
# plot the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# In[29]:


get_ipython().run_line_magic('time', 'nb.fit(X_train_tfidf_en, y_train_en)')

# Make predictions on the validation data
y_pred_en = nb.predict(X_val_tfidf_en)
accuracy_en = metrics.accuracy_score(y_val_en, y_pred_en)
print("Accuracy:", accuracy_en)

cm = confusion_matrix(y_val_en, y_pred_en)
# plot the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# In[30]:


get_ipython().run_line_magic('time', 'nb.fit(X_train_tfidf_fr, y_train_fr)')

# Make predictions on the validation data
y_pred_fr = nb.predict(X_val_tfidf_fr)
accuracy_fr = metrics.accuracy_score(y_val_fr, y_pred_fr)
print("Accuracy:", accuracy_fr)

cm = confusion_matrix(y_val_fr, y_pred_fr)
# plot the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# In[31]:


get_ipython().run_line_magic('time', 'nb.fit(X_train_tfidf_de, y_train_de)')

# Make predictions on the validation data
y_pred_de = nb.predict(X_val_tfidf_de)
accuracy_de = metrics.accuracy_score(y_val_de, y_pred_de)
print("Accuracy:", accuracy_de)

cm = confusion_matrix(y_val_de, y_pred_de)
# plot the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# #### kNN Classifier

# In[32]:


knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train_tfidf_es, y_train_es)

# creating a confusion matrix
knn_predictions = knn.predict(X_val_tfidf_es)
# accuracy on X_test
accuracy = knn.score(X_val_tfidf_es, y_val_es)
print( accuracy)
 
cm = confusion_matrix(y_val_es, knn_predictions)
# plot the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# In[33]:


knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train_tfidf_fr, y_train_fr)

# creating a confusion matrix
knn_predictions = knn.predict(X_val_tfidf_fr)
# accuracy on X_test
accuracy = knn.score(X_val_tfidf_fr, y_val_fr)
print( accuracy)
 
cm = confusion_matrix(y_val_fr, knn_predictions)
# plot the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# In[34]:


knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train_tfidf_en, y_train_en)

# creating a confusion matrix
knn_predictions = knn.predict(X_val_tfidf_en)
# accuracy on X_test
accuracy = knn.score(X_val_tfidf_en, y_val_en)
print( accuracy)
 
cm = confusion_matrix(y_val_en, knn_predictions)
# plot the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# In[35]:


knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train_tfidf_de, y_train_de)

# creating a confusion matrix
knn_predictions = knn.predict(X_val_tfidf_de)
# accuracy on X_test
accuracy = knn.score(X_val_tfidf_de, y_val_de)
print( accuracy)
 
cm = confusion_matrix(y_val_de, knn_predictions)
# plot the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# ##### Classifiers Accuracies by Language graph

# In[36]:


accuracies = {'Spanish': {'Random Forest': 0.6411764705882353, 'Naive Bayes': 0.6794117647058824, 'KNN': 0.5676470588235294},
              'French': {'Random Forest': 0.6823529411764706, 'Naive Bayes': 0.7058823529411765, 'KNN': 0.65},
              'English': {'Random Forest': 0.6764705882352942, 'Naive Bayes': 0.6941176470588235, 'KNN': 0.6323529411764706},
              'German': {'Random Forest': 0.6647058823529411, 'Naive Bayes': 0.6882352941176471, 'KNN': 0.638235294117647}}

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))  # Increase the figure size

# Set the x-axis labels and positions
langs = list(accuracies.keys())
x_pos = [i for i in range(len(langs))]
ax.set_xticks(x_pos)
ax.set_xticklabels(langs, fontsize=12)  # Increase font size of x-axis labels

# Set the bar width and positions
bar_width = 0.15
bar_pos_rf = [i - bar_width for i in x_pos]
bar_pos_nb = x_pos
bar_pos_knn = [i + bar_width for i in x_pos]

# Plot the bars with custom colors for each classifier
ax.bar(bar_pos_rf, [accuracies[lang]['Random Forest'] for lang in langs], bar_width, 
       label='Random Forest', color='cornflowerblue')
ax.bar(bar_pos_nb, [accuracies[lang]['Naive Bayes'] for lang in langs], bar_width, 
       label='Naive Bayes', color='darkorange')
ax.bar(bar_pos_knn, [accuracies[lang]['KNN'] for lang in langs], bar_width, 
       label='KNN', color='forestgreen')

# Set the plot title, legend, and y-axis label
ax.set_title('Classifier Accuracies by Language', fontsize=16)
ax.legend(ncol=1, fontsize=12, bbox_to_anchor=(1.2, 1.22))
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('Language', fontsize=12)

# Show the plot
plt.show()


# In[37]:


df = pd.DataFrame.from_dict(accuracies, orient='index')

# Print the dataframe
df


# #### Fine-Tuning BERT

# In[38]:


# Concatenate train data and test data
all_spanish_data = np.concatenate([spanish_dataset_train.review_body.values, spanish_dataset_test.review_body.values])
all_french_data = np.concatenate([french_dataset_train.review_body.values, french_dataset_test.review_body.values])
all_english_data = np.concatenate([english_dataset_train.review_body.values, english_dataset_test.review_body.values])
all_german_data = np.concatenate([german_dataset_train.review_body.values, german_dataset_test.review_body.values])


# In[39]:


def map_polarity(label):
    if label == 1:
        return 'positive'
    elif label == 0:
        return 'neutral'
    else:
        return 'negative'
    
label_dict = {}


# In[40]:


# Apply the function to the 'label' column to get the 'polarity' column
spanish_dataset_train['polarity'] = spanish_dataset_train['label'].apply(map_polarity)
spanish_dataset_train= spanish_dataset_train.drop(['label', 'review_id'], axis=1)

possible_labels = spanish_dataset_train.polarity.unique()

for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

spanish_dataset_train['label'] = spanish_dataset_train.polarity.replace(label_dict)

# Training and Validation split
X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(spanish_dataset_train.index.values, 
                                                 spanish_dataset_train.label.values, 
                                                  test_size=0.15, 
                                                  random_state=42, 
                                                  stratify=spanish_dataset_train.label.values)

spanish_dataset_train['data_type'] = ['not_set']*spanish_dataset_train.shape[0]

spanish_dataset_train.loc[X_train_es, 'data_type'] = 'train'
spanish_dataset_train.loc[X_val_es, 'data_type'] = 'val'

spanish_dataset_train.groupby(['polarity', 'label', 'data_type']).count()


# In[41]:


# Apply the function to the 'label' column to get the 'polarity' column
french_dataset_train['polarity'] = french_dataset_train['label'].apply(map_polarity)
french_dataset_train= french_dataset_train.drop(['label', 'review_id'], axis=1)

possible_labels = french_dataset_train.polarity.unique()

for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

french_dataset_train['label'] = french_dataset_train.polarity.replace(label_dict)

# Training and Validation split
X_train_fr, X_val_fr, y_train_fr, y_val_fr = train_test_split(french_dataset_train.index.values, 
                                                 french_dataset_train.label.values, 
                                                  test_size=0.15, 
                                                  random_state=42, 
                                                  stratify=french_dataset_train.label.values)

french_dataset_train['data_type'] = ['not_set']*french_dataset_train.shape[0]

french_dataset_train.loc[X_train_fr, 'data_type'] = 'train'
french_dataset_train.loc[X_val_fr, 'data_type'] = 'val'

french_dataset_train.groupby(['polarity', 'label', 'data_type']).count()


# In[42]:


# Apply the function to the 'label' column to get the 'polarity' column
english_dataset_train['polarity'] = english_dataset_train['label'].apply(map_polarity)
english_dataset_train= english_dataset_train.drop(['label', 'review_id'], axis=1)

possible_labels = english_dataset_train.polarity.unique()

for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

english_dataset_train['label'] = english_dataset_train.polarity.replace(label_dict)

# Training and Validation split
X_train_en, X_val_en, y_train_en, y_val_en = train_test_split(english_dataset_train.index.values, 
                                                 english_dataset_train.label.values, 
                                                  test_size=0.15, 
                                                  random_state=42, 
                                                  stratify=english_dataset_train.label.values)

english_dataset_train['data_type'] = ['not_set']*english_dataset_train.shape[0]

english_dataset_train.loc[X_train_en, 'data_type'] = 'train'
english_dataset_train.loc[X_val_en, 'data_type'] = 'val'

english_dataset_train.groupby(['polarity', 'label', 'data_type']).count()


# In[43]:


# Apply the function to the 'label' column to get the 'polarity' column
german_dataset_train['polarity'] = german_dataset_train['label'].apply(map_polarity)
german_dataset_train= german_dataset_train.drop(['label', 'review_id'], axis=1)

possible_labels = german_dataset_train.polarity.unique()

for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

german_dataset_train['label'] = german_dataset_train.polarity.replace(label_dict)

# Training and Validation split
X_train_de, X_val_de, y_train_de, y_val_de = train_test_split(german_dataset_train.index.values, 
                                                 german_dataset_train.label.values, 
                                                  test_size=0.15, 
                                                  random_state=42, 
                                                  stratify=german_dataset_train.label.values)

german_dataset_train['data_type'] = ['not_set']*german_dataset_train.shape[0]

german_dataset_train.loc[X_train_en, 'data_type'] = 'train'
german_dataset_train.loc[X_val_en, 'data_type'] = 'val'

german_dataset_train.groupby(['polarity', 'label', 'data_type']).count()


# In[44]:


spanish_dataset_train['polarity'].value_counts()


# In[45]:


french_dataset_train['polarity'].value_counts()


# In[46]:


english_dataset_train['polarity'].value_counts()


# In[47]:


german_dataset_train['polarity'].value_counts()


# #### Tokenizer and Encoding the Data

# In[48]:


tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=True)


# In[49]:


# Encode our concatenated data
encoded_spanish_data = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_spanish_data]
encoded_french_data = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_french_data]
encoded_english_data = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_english_data]
encoded_german_data = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_german_data]


# Find the maximum length
max_len_es = max([len(sent) for sent in encoded_spanish_data])
max_len_fr = max([len(sent) for sent in encoded_french_data])
max_len_en = max([len(sent) for sent in encoded_english_data])
max_len_de = max([len(sent) for sent in encoded_german_data])
print('Max length for Spanish data: ', max_len_es)
print('Max length for French data: ', max_len_fr)
print('Max length for English data: ', max_len_en)
print('Max length for German data: ', max_len_de)


# In[50]:


encoded_data_train_es = tokenizer.batch_encode_plus(
    spanish_dataset_train[spanish_dataset_train.data_type=='train'].review_body.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=max_len_es - 4, 
    return_tensors='pt'
)

encoded_data_val_es = tokenizer.batch_encode_plus(
    spanish_dataset_train[spanish_dataset_train.data_type=='val'].review_body.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=max_len_es - 4, 
    return_tensors='pt'
)


input_ids_train_es = encoded_data_train_es['input_ids']
attention_masks_train_es = encoded_data_train_es['attention_mask']
labels_train_es = torch.tensor(spanish_dataset_train[spanish_dataset_train.data_type=='train'].label.values)

input_ids_val_es = encoded_data_val_es['input_ids']
attention_masks_val_es = encoded_data_val_es['attention_mask']
labels_val_es = torch.tensor(spanish_dataset_train[spanish_dataset_train.data_type=='val'].label.values)


# In[51]:


encoded_data_train_fr = tokenizer.batch_encode_plus(
    french_dataset_train[french_dataset_train.data_type=='train'].review_body.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=max_len_fr - 4, 
    return_tensors='pt'
)

encoded_data_val_fr = tokenizer.batch_encode_plus(
    french_dataset_train[french_dataset_train.data_type=='val'].review_body.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=max_len_fr - 4, 
    return_tensors='pt'
)


input_ids_train_fr = encoded_data_train_fr['input_ids']
attention_masks_train_fr = encoded_data_train_fr['attention_mask']
labels_train_fr = torch.tensor(french_dataset_train[french_dataset_train.data_type=='train'].label.values)

input_ids_val_fr = encoded_data_val_fr['input_ids']
attention_masks_val_fr = encoded_data_val_fr['attention_mask']
labels_val_fr = torch.tensor(french_dataset_train[french_dataset_train.data_type=='val'].label.values)


# In[52]:


encoded_data_train_en = tokenizer.batch_encode_plus(
    english_dataset_train[english_dataset_train.data_type=='train'].review_body.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=max_len_en - 4, 
    return_tensors='pt'
)

encoded_data_val_en = tokenizer.batch_encode_plus(
    english_dataset_train[english_dataset_train.data_type=='val'].review_body.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=max_len_en - 4, 
    return_tensors='pt'
)


input_ids_train_en = encoded_data_train_en['input_ids']
attention_masks_train_en = encoded_data_train_en['attention_mask']
labels_train_en = torch.tensor(english_dataset_train[english_dataset_train.data_type=='train'].label.values)

input_ids_val_en = encoded_data_val_en['input_ids']
attention_masks_val_en = encoded_data_val_en['attention_mask']
labels_val_en = torch.tensor(english_dataset_train[english_dataset_train.data_type=='val'].label.values)


# In[53]:


encoded_data_train_de = tokenizer.batch_encode_plus(
    german_dataset_train[german_dataset_train.data_type=='train'].review_body.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=500, 
    return_tensors='pt'
)

encoded_data_val_de = tokenizer.batch_encode_plus(
    german_dataset_train[german_dataset_train.data_type=='val'].review_body.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=500, 
    return_tensors='pt'
)


input_ids_train_de = encoded_data_train_de['input_ids']
attention_masks_train_de = encoded_data_train_de['attention_mask']
labels_train_de = torch.tensor(german_dataset_train[german_dataset_train.data_type=='train'].label.values)

input_ids_val_de = encoded_data_val_de['input_ids']
attention_masks_val_de = encoded_data_val_de['attention_mask']
labels_val_de = torch.tensor(german_dataset_train[german_dataset_train.data_type=='val'].label.values)


# In[54]:


dataset_train_es = TensorDataset(input_ids_train_es, attention_masks_train_es, labels_train_es)
dataset_val_es = TensorDataset(input_ids_val_es, attention_masks_val_es, labels_val_es)

dataset_train_fr = TensorDataset(input_ids_train_fr, attention_masks_train_fr, labels_train_fr)
dataset_val_fr = TensorDataset(input_ids_val_fr, attention_masks_val_fr, labels_val_fr)

dataset_train_en = TensorDataset(input_ids_train_en, attention_masks_train_en, labels_train_en)
dataset_val_en = TensorDataset(input_ids_val_en, attention_masks_val_en, labels_val_en)

dataset_train_de = TensorDataset(input_ids_train_de, attention_masks_train_de, labels_train_de)
dataset_val_de = TensorDataset(input_ids_val_de, attention_masks_val_de, labels_val_de)


# In[55]:


len(dataset_train_es), len(dataset_val_es)


# In[56]:


len(dataset_train_fr), len(dataset_val_fr)


# In[57]:


len(dataset_train_en), len(dataset_val_en)


# In[58]:


len(dataset_train_de), len(dataset_val_de)


# In[59]:


dataset_val_es.tensors


# In[60]:


dataset_val_fr.tensors


# In[61]:


dataset_val_en.tensors


# In[62]:


dataset_val_de.tensors


# ##### BERT Pre-trained Model

# In[63]:


model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)


# In[64]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(device)


# ##### Data Loaders

# In[65]:


batch_size = 32

dataloader_train_es = DataLoader(dataset_train_es, 
                              sampler=RandomSampler(dataset_train_es), 
                              batch_size=batch_size)

dataloader_validation_es = DataLoader(dataset_val_es, 
                                   sampler=SequentialSampler(dataset_val_es), 
                                   batch_size=batch_size)

dataloader_train_fr = DataLoader(dataset_train_fr, 
                              sampler=RandomSampler(dataset_train_fr), 
                              batch_size=batch_size)

dataloader_validation_fr = DataLoader(dataset_val_fr, 
                                   sampler=SequentialSampler(dataset_val_fr), 
                                   batch_size=batch_size)

dataloader_train_en = DataLoader(dataset_train_en, 
                              sampler=RandomSampler(dataset_train_en), 
                              batch_size=batch_size)

dataloader_validation_en = DataLoader(dataset_val_en, 
                                   sampler=SequentialSampler(dataset_val_en), 
                                   batch_size=batch_size)

dataloader_train_de = DataLoader(dataset_train_de, 
                              sampler=RandomSampler(dataset_train_de), 
                              batch_size=batch_size)

dataloader_validation_de = DataLoader(dataset_val_de, 
                                   sampler=SequentialSampler(dataset_val_de), 
                                   batch_size=batch_size)


# ##### Optimizer

# In[66]:


optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)


# ##### Performance Metrics

# In[67]:


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_overall(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    total_correct = 0
    total_samples = len(labels_flat)

    for i in range(total_samples):
        if (preds_flat[i] == labels_flat[i]):
            total_correct += 1
    
    overall_accuracy = total_correct / total_samples

    print(f'Overall accuracy: {overall_accuracy:.4f}\n')


# ##### Traing Loop

# In[68]:


seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# In[69]:


epochs = 3

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train_es)*epochs)


# In[70]:


def evaluate(dataloader_val_es):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val_es:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val_es) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals


# In[71]:


for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train_es, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    #torch.save(model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train_es)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation_es)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')


# In[72]:


accuracy_overall(predictions, true_vals)


# In[73]:


model.push_to_hub("Worgu/Final_Project_finetuned_bert-base-multilingual-cased_spanish")


# In[ ]:


tokenizer.save_pretrained('tokenizer_dir')
tokenizer.push_to_hub("Worgu/Final_Project_finetuned_bert-base-multilingual-cased_spanish")


# In[63]:


model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)


# In[64]:


optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)


# In[72]:


epochs = 3

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train_fr)*epochs)


# In[73]:


def evaluate(dataloader_val_fr):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val_fr:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val_fr) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals


# In[74]:


for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train_fr, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    #torch.save(model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train_fr)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation_fr)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')


# In[75]:


accuracy_overall(predictions, true_vals)


# In[79]:


model.push_to_hub("Worgu/Final_Project_finetuned_bert-base-multilingual-cased_french")


# In[ ]:


tokenizer.save_pretrained('tokenizer_dir')
tokenizer.push_to_hub("Worgu/Final_Project_finetuned_bert-base-multilingual-cased_french")


# In[69]:


epochs = 3

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train_en)*epochs)


# In[70]:


def evaluate(dataloader_val_en):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val_en:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val_en) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals


# In[71]:


for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train_en, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    #torch.save(model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train_en)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation_en)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')


# In[72]:


accuracy_overall(predictions, true_vals)


# In[73]:


model.push_to_hub("Worgu/Final_Project_finetuned_bert-base-multilingual-cased_english")


# In[ ]:


tokenizer.save_pretrained('tokenizer_dir')
tokenizer.push_to_hub("Worgu/Final_Project_finetuned_bert-base-multilingual-cased_english")


# In[69]:


epochs = 3

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train_de)*epochs)


# In[70]:


def evaluate(dataloader_val_de):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val_de:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val_de) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals


# In[71]:


for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train_de, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    #torch.save(model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train_de)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation_de)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')


# In[72]:


accuracy_overall(predictions, true_vals)


# In[73]:


model.push_to_hub("Worgu/Final_Project_finetuned_bert-base-multilingual-cased_german")


# In[84]:


tokenizer.save_pretrained('tokenizer_dir')
tokenizer.push_to_hub("Worgu/Final_Project_finetuned_bert-base-multilingual-cased_german")


# ##### Classifier Accuracies by Language

# In[74]:


accuracies = {'Spanish': {'Random Forest': 0.6411764705882353, 'BERT': 0.6235},
              'French': {'Random Forest': 0.6823529411764706, 'BERT': 0.6333},
              'English': {'Random Forest': 0.6764705882352942, 'BERT': 0.6431},
              'German': {'Random Forest': 0.6647058823529411, 'BERT': 0.6725}}

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))  # Increase the figure size

# Set the x-axis labels and positions
langs = list(accuracies.keys())
x_pos = [i for i in range(len(langs))]
ax.set_xticks(x_pos)
ax.set_xticklabels(langs, fontsize=12)  # Increase font size of x-axis labels

# Set the bar width and positions
bar_width = 0.15
bar_pos_rf = [i - bar_width for i in x_pos]
bar_pos_nb = x_pos
bar_pos_knn = [i + bar_width for i in x_pos]

# Plot the bars with custom colors for each classifier
ax.bar(bar_pos_rf, [accuracies[lang]['Random Forest'] for lang in langs], bar_width, 
       label='Random Forest', color='cornflowerblue')
ax.bar(bar_pos_nb, [accuracies[lang]['BERT'] for lang in langs], bar_width, 
       label='BERT', color='darkorange')

# Set the plot title, legend, and y-axis label
ax.set_title('Classifier Accuracies by Language', fontsize=16)
ax.legend(ncol=1, fontsize=12, bbox_to_anchor=(1.2, 1.22))
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('Language', fontsize=12)

# Show the plot
plt.show()


# In[75]:


df = pd.DataFrame.from_dict(accuracies, orient='index')

# Print the dataframe
df


# ##### Evaluation Metrics by Language

# In[107]:


import pandas as pd

# Create a list of dictionaries to hold the data
data = [
    {'Language': 'Spanish', 'Epoch': 1, 'Training Loss': 1.00524238046709, 'Validation Loss': 0.9599836468696594, 'F1 Score (Weighted)': 0.4917485718048764},
    {'Language': 'Spanish', 'Epoch': 2, 'Training Loss': 0.8638723564671946, 'Validation Loss': 0.8875471688807011, 'F1 Score (Weighted)': 0.5484364042119974},
    {'Language': 'Spanish', 'Epoch': 3, 'Training Loss': 0.7803521012211894, 'Validation Loss': 0.8829927667975426, 'F1 Score (Weighted)': 0.5580235917007454},
    {'Language': 'French', 'Epoch': 1, 'Training Loss': 1.014892307611612, 'Validation Loss': 1.0046335607767105, 'F1 Score (Weighted)': 0.4696118522904601},
    {'Language': 'French', 'Epoch': 2, 'Training Loss': 0.86506346817855, 'Validation Loss': 0.8765472173690796, 'F1 Score (Weighted)': 0.551510876033942},
    {'Language': 'French', 'Epoch': 3, 'Training Loss': 0.7815885507798457, 'Validation Loss': 0.8848339430987835, 'F1 Score (Weighted)': 0.5663558887607331},
    {'Language': 'English', 'Epoch': 1, 'Training Loss': 1.009612310718704, 'Validation Loss': 0.9073016569018364, 'F1 Score (Weighted)': 0.5412463451210331},
    {'Language': 'English', 'Epoch': 2, 'Training Loss': 0.857522264286712, 'Validation Loss': 0.8431726545095444, 'F1 Score (Weighted)': 0.5655537931156673},
    {'Language': 'English', 'Epoch': 3, 'Training Loss':  0.7942181766688169, 'Validation Loss': 0.8328190259635448, 'F1 Score (Weighted)': 0.5655537931156673},
    {'Language': 'German', 'Epoch': 1, 'Training Loss':  0.9352489218607054, 'Validation Loss': 0.83482476323843, 'F1 Score (Weighted)': 0.5780703150730558},
    {'Language': 'German', 'Epoch': 2, 'Training Loss':  0.7607808054148496, 'Validation Loss': 0.7899614498019218, 'F1 Score (Weighted)': 0.5956314616967727},
    {'Language': 'German', 'Epoch': 3, 'Training Loss':  0.6839471256339943, 'Validation Loss': 0.7857535295188427, 'F1 Score (Weighted)': 0.6216210395878077}
]

# Create the dataframe
df = pd.DataFrame(data)

# Print the dataframe
grouped_df = df.groupby('Language')

for name, group in grouped_df:
    print(f"\n{name}")
    print(group)
grouped_df

