!pip install pyspellchecker
!pip install transformers
!pip install clean-text

# IMPORT STATEMENTS
import nltk
import nltk.data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import statistics
import random
import re
import cleantext
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
from spellchecker import SpellChecker

def clean_text(text):
  re_tag = re.compile(r'<[^>]+>')
  text = re_tag.sub('', text.strip())

  text = cleantext.clean(text,
                         to_ascii=False,
                         lower=True,
                         no_line_breaks=False,
                         no_urls=False,
                         no_emails=False,
                         no_phone_numbers=False,
                         no_numbers=False,
                         no_digits=False,
                         no_currency_symbols=False,
                         no_punct=True,
                         replace_with_url="<URL>",
                         replace_with_email="<EMAIL>",
                         replace_with_phone_number="<PHONE>",
                         replace_with_number="<NUMBER>",
                         replace_with_digit="0",
                         replace_with_currency_symbol="<CUR>",
                         lang="en")
  return text

#@title Experiment Options
k = 41#@param {type:"integer"}
data_to_use = "TextFooler" #@param["BAE", "BAE_100", "DeepWordBug", "FasterGeneticAlgo", "Kuleshov", "PWWS", "TextBugger", "TextFooler"]

# Get the specified input file
if data_to_use == "BAE":
  filename = "BAEGarg2019_perturbed_data.txt"
elif data_to_use == "BAE_100":
  filename = "BAEGarg2019_100_data.txt"
elif data_to_use == "DeepWordBug":
  filename = "DeepWordBugGao2018_100_data.txt"
elif data_to_use == "FasterGeneticAlgo":
  filename = "FasterGeneticAlgo_100_data.txt"
elif data_to_use == "Kuleshov":
  filename = "Kuleshov2017_100_data.txt"
elif data_to_use == "PWWS":
  filename = "PWWSRen2019_100_data.txt"
elif data_to_use == "TextBugger":
  filename = "TextBuggerLi2018_100_data.txt"
elif data_to_use == "TextFooler":
  filename = "TextFoolerJin2019_100_data.txt"

# Open the given file for the atatcked data for reading
datafile = open(filename, mode='r')

# Declaring and initializing variables
num_reviews = 100
correct = 0.0
total = 0.0
total_certainty = 0.0
correct_certainty = 0.0
incorrect_certainty = 0.0
modified_sent = ""
i = 0

# nltk punkt sentence detector and spellchecker
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
spell = SpellChecker()
stop_words = set(stopwords.words('english'))

# For each review in dataset 
for i in range(num_reviews):
  review_and_label = datafile.readline()
  seperate = review_and_label.split(sep='", ')
  review = seperate[0]
  correct_label = seperate[1]
  predictions = []
  R_hat = []

  # Printing the original review
  print(f"Original Review: {review}")

  # Sentencize the review and call R
  R = sent_detector.tokenize(review.strip())
  
  # Get N # of sentences for R
  N = len(R)

  for i in range(k):
    # Randomly select a sentences from R
    temp = random.randint(0, N-1)
    sentence = R[temp]

    # Tokenize the sentence and get number of tokens
    tokenized_sentence = sentence.split()
    num_tokens = len(tokenized_sentence)

    # Randomly select token to correct 
    token_to_correct = random.randint(0, num_tokens-1)
    token = tokenized_sentence[token_to_correct]
    while token == "" or token == " ":
      token_to_correct = random.randint(0, num_tokens-1)
      token = tokenized_sentence[token_to_correct]
    token = " " + token + " "

    # Randomly select correction to make
    choice = random.randint(0, 3)
    
    # Spell Check
    if choice == 0:
      list_of_token = []
      list_of_token.append(token)
      misspelled = spell.unknown(list_of_token)
      for tok in misspelled:
        correction = spell.correction(tok)
        correction = " " + correction + " "
        modified_sent = sentence.replace(token, correction)

    # Synonym Substitution
    elif choice == 1:
      synonym = ""
      og_token = token
      token = token.lower()
      token = clean_text(token)
      modified_sent = sentence.replace(og_token, token)
      lemmatized_token = WordNetLemmatizer().lemmatize(token)
      synset = wn.synsets(lemmatized_token)
      synset = [ws.name().split(".")[0].replace('_', ' ') for ws in synset]

      # Create a verified synset without the token itself and duplicates
      if synset:
        verified_synset = []
        for syn in synset:
          if syn != token and syn not in verified_synset:
            verified_synset.append(syn)
        # Randomly select synonym from the verified synset
        if len(verified_synset) > 1:
          syn_to_sub = random.randint(0, len(verified_synset)-1)
          synonym = verified_synset[syn_to_sub]
        elif len(verified_synset) == 1:
          synonym = verified_synset[0]
      else:
        synonym = token
      synonym = " " + synonym + " "
      modified_sent = modified_sent.replace(token, synonym)

    # Drop Word
    elif choice == 2:
      modified_sent = sentence.replace(token, " ")

    # Add modified sentence to R_hat
    R_hat.append(modified_sent)

  # Classify all of the modified sentences in R_hat
  classifier = pipeline("sentiment-analysis")
  for j in range(k):
    if len(R_hat[j]) < 512:
      temp = classifier(R_hat[j])[0]
      temp_pred = temp['label']
      if temp_pred == 'POSITIVE':
        intermediate_pred = 1
      elif temp_pred == 'NEGATIVE':
        intermediate_pred = 0

      # Append intermediate pred to array
      predictions.append(intermediate_pred)
        
  print(str(predictions))
  
  count_0 = predictions.count(0)
  count_1 = predictions.count(1)

  # Majority vote based on the intermediate pred from each R' to get final pred
  if count_0 == count_1:
    final_prediction = random.randint(0, 1)
  else:
    final_prediction = statistics.mode(predictions)

  # Getting the certainty of the final prediction
  certainty = 0.0
  if int(final_prediction) == 0:
    certainty = count_0 / (count_0 + count_1)
  else:
    certainty = count_1 / (count_0 + count_1)

  print(f"Prediction: {final_prediction}  Correct Label: {correct_label}  Certainty: {certainty}\n")

  if int(final_prediction) == int(correct_label):
    correct += 1.0
    total += 1.0
    total_certainty += certainty
    correct_certainty += certainty
  else:
    total += 1.0
    total_certainty += certainty
    incorrect_certainty += certainty

# Calculate the accuracy and certainty of the defended model
accuracy = correct/total
avg_certainty = total_certainty/total
correct_certainty = correct_certainty / correct
incorrect_certainty = incorrect_certainty / (total - correct)

print("Accuracy: ", accuracy)
print("Correct: ", correct)
print("Total: ", total)
print(f"Avg. Certainty: {avg_certainty} Correct Certainty: {correct_certainty}  Incorrect Certainty: {incorrect_certainty}\n")
