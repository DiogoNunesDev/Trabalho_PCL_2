import nltk
from collections import Counter
from nltk import ngrams
import os

categories = {"GEOGRAPHY":"", "MUSIC":"", "LITERATURE":"", "HISTORY":"", "SCIENCE":""}

train_to_list = []

with open("train.txt", encoding="utf-8") as f:
  lines = f.read().split("\n")[:-1]
  for line in lines:
    line_formated = line.split("\t")

    train_to_list.append(line_formated)

    category = line_formated[0]
    question = line_formated[1]
    answear = line_formated[2]

    categories[category]+= question + " "  + answear

nltk.download('punkt_tab')
os.makedirs('counts', exist_ok=True)
os.makedirs('counts2', exist_ok=True)

for c in categories:
    raw_tokens = nltk.word_tokenize(categories[c].lower())

    tokens = [t for t in raw_tokens if t.isalnum()]

    # UNIGRAMAS 
    unigram_counts = Counter(ngrams(tokens, 1))
    filename_uni = f"counts/unigrams_{c.replace(' ', '_')}.txt"

    with open(filename_uni, 'w', encoding='utf-8') as f:
        f.write("Unigramas\n")
        for gram, freq in unigram_counts.most_common():
            f.write(f"{gram[0]} {freq}\n")

    # BIGRAMAS
    bigram_counts = Counter(ngrams(tokens, 2))
    filename_bi = f"counts/bigrams_{c.replace(' ', '_')}.txt"

    with open(filename_bi, 'w', encoding='utf-8') as f:
        f.write("Bigramas\n")
        for gram, freq in bigram_counts.most_common():
            f.write(f"{gram[0]} {gram[1]} {freq}\n")

    print(f"Processado: {c} ({len(tokens)} palavras limpas)")