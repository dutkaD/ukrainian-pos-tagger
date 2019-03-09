class Evaluation:
    def __init__(self):
        self.tags_correct = 0
        self.sentences_correct = 0
        self.overall_tags = 0
        self.overall_sentences = 0
        self.counter = 0






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