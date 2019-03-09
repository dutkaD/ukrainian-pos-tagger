import operator
import sys
from collections import defaultdict
from keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from nltk import SnowballStemmer
from ukr_stemmer.ukr_stemmer3 import UkrainianStemmer

DATA = "pickledSTEM"
MODEL = "stemmed_no_embeddingsboth-005.h5"
out = "result.txt"
TEST_DATA = "clean_test.txt"
model = load_model(MODEL)


STEMMER = True  # False if the complete forms of the words should be passed to the tagger
LANGUAGE = "ua"  # "ru" for Russian. "ua" for Ukrainian

tags_correct = 0
sentences_correct = 0
overall_tags = 0
overall_sentences = 0
counter = 0

def evaluate(correct_tags, predicted):
	correct = 0
	sentece = False
	err = False
	for i in range(len(correct_tags)):
		try:
			if correct_tags[i] == predicted[i]:
				correct += 1  # TODO: add a function that counts the result matrix ans saves the result as a pickle file
			if correct == len(predicted):
				sentece = True
		except IndexError:
			pass
	return correct, sentece


def write_tagged(words, tags):
	sentence = ""
	with open(out, "a") as f:
		for i in range(len(words)):
			try:
				sentence += words[i] + "/" + tags[i] + " "
			except:
				print(sentence)
				# TODO: take care of the exception case
		sentence += "\n"
		f.write(sentence)


def get_predicted_tags(prediction, tokenized):
	predicted = []
	for i, pred in enumerate(prediction[0]):
		if i >= len(list(enumerate(prediction[0])))-len(tokenized):
			try:
				predicted.append(int2tag[np.argmax(pred)])
			except KeyError:
				pass

	return predicted

with open(DATA + '.pkl', 'rb') as f:
	X_train, Y_train, word2int, int2word, tag2int, int2tag, tag2instances = pickle.load(f)

	del X_train
	del Y_train

with open(TEST_DATA) as test_f:
	test_corpus = test_f.readlines()

"""
Make two lists: tags and words to be used for later evaluation
"""
words_pro_sent = []  # list of list of words per sentence
correct_tags = []  # list of list of tags per sentence
for line in test_corpus:
	words = []
	tags = []
	if len(line) > 0:
		for word in line.split():
			try:
				w, tag = word.split('/')
				w = w.lower()
				words.append(w)
				tags.append(tag)
			except:
				print("AHA")
	words_pro_sent.append(words)
	correct_tags.append(tags)
tokenized_sentence = []

# stats = evaluation.Stats(int2tag, tag2int, tag2instances)
fails = []
for i in range(len(words_pro_sent)):
	for word in words_pro_sent[i]:
		try:
			if STEMMER:
				if LANGUAGE == "ru":
					stemmer = SnowballStemmer("russian")
					word = stemmer.stem(word)
				if LANGUAGE == "ua":
					stemObj = UkrainianStemmer(word)
					word = stemObj.stem_word()
			tokenized_sentence.append(word2int[word])

		except KeyError:
			tokenized_sentence.append(word2int["<UNKNOWN>"])  # if the word was not found in the dictionary

	np_tokenized = np.asarray([tokenized_sentence])

	padded_tokenized_sentence = pad_sequences(np_tokenized, maxlen=100)
	prediction = model.predict(padded_tokenized_sentence)
	predicted_tags = get_predicted_tags(prediction, words_pro_sent[i])
	corr, sent = evaluate(correct_tags[i], predicted_tags)

	overall_tags += len(correct_tags[i])
	overall_sentences += 1
	tags_correct += corr
	if sent:
		sentences_correct += 1

	write_tagged(words_pro_sent[i], predicted_tags)
	print(i)


print("{}/{} - {}".format(tags_correct, overall_tags, round(tags_correct/overall_tags*100, 2)))
print("{}/{} - {}".format(sentences_correct, overall_sentences, round(sentences_correct/overall_sentences*100, 2)))


# stats.show_results()




