import json
import nltk
import string
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset
from collections import Counter

# import the relavant libraries for loggin in to Huggingface
from huggingface_hub import HfApi, HfFolder

# Only one time needed
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download("wordnet")

# Read the JSON file
# with open("./azcorpusv0_chunk_53.json", "r", encoding="utf-8") as file:
#     data = json.load(file)


punctuation = set(string.punctuation)
stop_words = set(stopwords.words("azerbaijani"))

# paste your token
token = "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"



# set api for login and save token
######## NOTE: Uncomment the following 3 lines of code to load the dataset from Huggingface only the first time
# api=HfApi(token=token)
# folder = HfFolder()
# folder.save_token(token)

dataset = load_dataset("azcorpus/azcorpus_v0", split="train")

############ NOTE: As it takes too much time, I just used the first dataset in azcorpus ###########################
#Please remove the following in order to include the whole dataset
dataset = [dataset[0]]

# print(data)


# # Tokenize the sentences
tokenized_sentences = []
for obj in dataset:
    text = obj['text']
    sentences = sent_tokenize(text)
    for sentence in sentences:
        words = word_tokenize(sentence)
        # Filtering out tokens consisting solely of punctuation characters
        words = [
            word.lower()
            for word in words
            if word.isalpha()
            and len(word) > 1 #do not take single characters
            and word not in punctuation #do not include punctuation marks
            and word not in stop_words #do not include stop words like 'amma', 'ki', 'belə'
        ]
        tokenized_sentences.append(words)

print(tokenized_sentences)


########################## Some statistical information ###########################
# all the words
all_words = [word for x in tokenized_sentences for word in x]

# Counting the frequency of each word
word_freq = Counter(all_words)

#Pring the number of all words we have
print("Number of total words in the dataset : ", len(all_words))

# Display the 10 most common words
print("10 Most Common Words:")
print(word_freq.most_common(10))
