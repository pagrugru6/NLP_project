from sklearn.metrics import accuracy_score 
import nltk
import pandas as pd
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
nltk.download('punkt')
nltk.download('stopwords')

train_data = pd.concat([pd.read_parquet('../../Translated_Questions/translated_ja_rows.parquet')
                        ,pd.read_parquet('../../Translated_Questions/translated_fi_rows.parquet')
                        ,pd.read_parquet('../../Translated_Questions/translated_ru_rows.parquet')])
# valid_data = pd.read_parquet('../../Translated_Questions/validation.parquet')

train_data, valid_data = train_test_split(train_data, test_size=0.1, random_state=42)

filtered_train_data = train_data[(train_data.lang.isin(["ja","fi","ru"]))]
filtered_valid_data = valid_data[(valid_data.lang.isin(["ja","fi","ru"]))]

ftda = pd.concat([filtered_train_data[(filtered_train_data.answerable.isin([True]))]["question"], filtered_train_data[(filtered_train_data.answerable.isin([True]))]["context"]])
ftdna = pd.concat([filtered_train_data[(filtered_train_data.answerable.isin([True]))]["question"], filtered_train_data[(filtered_train_data.answerable.isin([False]))]["context"]])

ansWordList = []
for sentence in list(ftda):
  ansWordList.extend(nltk.tokenize.word_tokenize(sentence))

notAnsWordList = []
for sentence in list(ftdna):
  notAnsWordList.extend(nltk.tokenize.word_tokenize(sentence))

ansWordCountDict = {}
for word in ansWordList:
  if word in ansWordCountDict:
    ansWordCountDict[word] += 1
  else:
    ansWordCountDict[word] = 1

notAnsWordCountDict = {}
for word in notAnsWordList:
  if word in notAnsWordCountDict:
    notAnsWordCountDict[word] += 1
  else:
    notAnsWordCountDict[word] = 1

ansWords = sorted(ansWordCountDict.items(), key=lambda item: item[1], reverse=True)
notAnsWords = sorted(notAnsWordCountDict.items(), key=lambda item: item[1], reverse=True)

fd_ans = nltk.FreqDist(ansWordList)
fd_notAns = nltk.FreqDist(notAnsWordList)

total_ans_words = len(ansWordList)
total_notAns_words = len(notAnsWordList)

stop_words = set(stopwords.words('english'))

max_rel_freq = 0.01
min_abs_freq = 1
ratio = 0.1

frequent_words = {}
frequent_unanswerable_words = {}

for word in fd_ans:
  if word not in stop_words and fd_ans[word] > min_abs_freq:
    rel_freq_ans = fd_ans[word] / total_ans_words
    if rel_freq_ans < max_rel_freq:
      if word in frequent_words:
        frequent_words[word] = frequent_words[word] +1
      else:
        frequent_words[word] = 1
for word in fd_notAns:
  if word not in stop_words and fd_notAns[word] > min_abs_freq:
    rel_freq_notAns = fd_notAns[word] / total_notAns_words
    if rel_freq_notAns < max_rel_freq:
      if word in frequent_unanswerable_words:
        frequent_unanswerable_words[word] = frequent_unanswerable_words[word] +1
      else:
        frequent_unanswerable_words[word] = 1


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_ans = TfidfVectorizer()
vectorizer_notAns = TfidfVectorizer()

tfidf_ans = vectorizer_ans.fit_transform(ftda)
tfidf_notAns = vectorizer_notAns.fit_transform(ftdna)

feature_names_ans = vectorizer_ans.get_feature_names_out()

feature_names_notAns = vectorizer_notAns.get_feature_names_out()


def rule_based_classifier(context):
  tokens = nltk.tokenize.word_tokenize(context)
  ans_score = sum(frequent_words[key] for key in tokens if key in frequent_words)
  not_ans_score = sum(frequent_unanswerable_words[key] for key in tokens if key in frequent_unanswerable_words)

  if ans_score > (not_ans_score + 1):
    return True
  else:
    return False

predictions = []
true_labels = []
valid_list = list(valid_data[['context', 'answerable']].itertuples(index=False, name=None));
random.shuffle(valid_list)
for context, label in valid_list:
  label = True
  if random.randrange(0,2) == 0:
    label = False
  prediction = rule_based_classifier(context)
  predictions.append(prediction)
  true_labels.append(label)

accuracy = accuracy_score(true_labels, predictions)
print(f"Baseline: {accuracy}")

predictions = []
true_labels = []
valid_list = list(valid_data[['context', 'answerable']].itertuples(index=False, name=None));
random.shuffle(valid_list)
for context, label in valid_list:
  prediction = rule_based_classifier(context)
  predictions.append(prediction)
  true_labels.append(label)

accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy}")