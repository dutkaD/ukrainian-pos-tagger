class Evaluation:
    def __init__(self):
        self.tags_correct = 0
        self.sentences_correct = 0
        self.overall_tags = 0
        self.overall_sentences = 0
        self.counter = 0


    def print_eval(self):
        print("{}/{} - {}".format(self.tags_correct, self.overall_tags,
                                  round(self.tags_correct / self.overall_tags * 100, 2)))
        print("{}/{} - {}".format(self.sentences_correct, self.overall_sentences,
                                  round(self.sentences_correct / self.overall_sentences * 100, 2)))

    def calculate_correct(self, correct_tags, predicted):
        corr, sent = evaluate(correct_tags, predicted)
        self.overall_tags += len(correct_tags)
        self.overall_sentences += 1
        self.tags_correct += corr
        if sent:
            self.sentences_correct += 1


def evaluate(correct_tags, predicted):
    correct = 0
    sentece = False
    for i in range(len(correct_tags)):
        try:
            if correct_tags[i] == predicted[i]:
                correct += 1
            if correct == len(predicted):
                sentece = True
        except IndexError:
            pass
    return correct, sentece