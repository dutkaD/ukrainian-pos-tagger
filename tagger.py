import pickle
import sys

from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import load_model
from evaluate import Evaluation
from ukr_stemmer.ukr_stemmer3 import UkrainianStemmer

DATA = "pickledSTEM"
MODEL = "ukrainianV1.h5"
out = "result.txt"

# todo: sentencewise
class Tagger:
    def __init__(self, input, evaluate=False, instant=False):
        self.instant = instant
        self.model = load_model(MODEL)
        self.evaluate = evaluate
        if self.evaluate:
            self.evaluator = Evaluation()
        self.X_train, \
        self.Y_train, \
        self.word2int, \
        self.int2word, \
        self.tag2int, \
        self.int2tag, \
        self.tag2instances = load_data()
        self.input_texts = input
        self.words_pro_sent = []
        self.correct_tags = []
        self.tokenized_sentence = []


    def make_lists(self):
        for line in self.input_texts:
            words = []
            tags = []
            if len(line) > 0:
                for word in line.split():
                    if self.evaluate:
                        try:
                            w, tag = word.split('/')
                            w = w.lower()
                            words.append(w)
                            tags.append(tag)
                        except:
                            print("Could not split by / - {}".format(word))
                    else:
                        words.append(word)
                        tags.append("UNK")

            self.words_pro_sent.append(words)
            self.correct_tags.append(tags)

    def get_predicted_tags(self, prediction, tokenized):
        predicted = []
        for i, pred in enumerate(prediction[0]):
            if i >= len(list(enumerate(prediction[0]))) - len(tokenized):
                try:
                    predicted.append(self.int2tag[np.argmax(pred)])
                except KeyError:
                    pass

        return predicted


    def label_data(self):
        self.make_lists()
        for i in range(len(self.words_pro_sent)):
            for word in self.words_pro_sent[i]:
                try:
                    stemObj = UkrainianStemmer(word)
                    word = stemObj.stem_word()
                    self.tokenized_sentence.append(self.word2int[word])

                except KeyError:
                    self.tokenized_sentence.append(self.word2int["<UNKNOWN>"])

            np_tokenized = np.asarray([self.tokenized_sentence])

            padded_tokenized_sentence = pad_sequences(np_tokenized, maxlen=100)
            prediction = self.model.predict(padded_tokenized_sentence)
            predicted_tags = self.get_predicted_tags(prediction, self.words_pro_sent[i])

            if self.evaluate:
                self.evaluator.calculate_correct(self.correct_tags[i], predicted_tags)

            write_tagged(self.words_pro_sent[i], predicted_tags)
            print(i)

        if self.evaluate:
            self.evaluator.print_eval()

def load_data():
    try:
        with open(DATA + '.pkl', 'rb') as f:
             return pickle.load(f)
    except FileNotFoundError:
        print("Could not open pickled data. Make sure the path is correct and file exists.")



def load_input(input):
    # todo: rename variables
    try:
        with open(input) as test_f:
            test_corpus = test_f.readlines()
    except FileNotFoundError:
        print("Could not open input file. Are you sure the " + input + " file exist and the path is correct?")
    return test_corpus


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


if __name__ == "__main__":
    text = []
    try:
        input_file = sys.argv[1]
        text = load_input(input_file)
        tagger = Tagger(text)
        tagger.label_data()
    except IndexError:
        text.append(input("Enter your setntence:"))
        tagger = Tagger(text)
        tagger.label_data()
    if len(sys.argv)>2:
        print("Too many arguments...")
























