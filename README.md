# ukrainian-pos-tagger
A part-of-speech tagger for Ukrainian

Model is trained on the [Bidirectional LSTM](https://github.com/aneesh-joshi/LSTM_POS_Tagger) implemented by [Aneesh Joshi ](https://github.com/aneesh-joshi). 

The tagger can be in different modes:
* With evaluation / without evaluation
* Instant tagging / tagging input from file

### SETUP

Download `data.pkl` and `ukrainianV1.h5` files from [here](https://github.com/dutkaD/tagger_data) and copy them to your tagger-directory.

### HOW TO RUN

- `python3 tagger.py` program will ask you to enter your sentence from terminal
- `python3 tagger.py filename` will tag the text from the input file






