from keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from ukr_stemmer.ukr_stemmer3 import UkrainianStemmer

DATA = "pickledSTEM.pkl"
TEST_DATA = "clean_test.txt"
MODEL = "bestStemmedBoth.h5"
OUT = "tagged.txt"
STEMMER = True
model = load_model(MODEL)




def write_tagged(words, tags):
	sentence = ""
	with open(OUT, "a") as f:
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


with open(DATA, 'rb') as f:
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
				stemObj = UkrainianStemmer(word)
				word = stemObj.stem_word()
			tokenized_sentence.append(word2int[word])

		except KeyError:
			tokenized_sentence.append(word2int["<UNKNOWN>"])  # if the word was not found in the dictionary

	np_tokenized = np.asarray([tokenized_sentence])
	padded_tokenized_sentence = pad_sequences(np_tokenized, maxlen=100)
	prediction = model.predict(padded_tokenized_sentence)
	predicted_tags = get_predicted_tags(prediction, words_pro_sent[i])
	write_tagged(words_pro_sent[i], predicted_tags)
