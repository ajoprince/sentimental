# Sentiment Analysis

**Version 1.0.0**

This is a short project on Sentiment Analysis. This is to showcase some concepts taught in [NLTK with Python 3 for Natural Language](https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL).

Some exapmles of such concepts are:
* Word Tokenization
* Effective Pickling
* Creating Model Vote Classifiers

Requirements:
* NLTK
* Keras

---

## How to use
```
from vote_classifier import sentiment
sentiment("Text to classify")
```
Output will be 0 for negative classification and 4 for positive classification.

---

## Sources
The code in this project is based on the [NLTK Tutorial](https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/) created by [sentdex](https://www.youtube.com/user/sentdex).

---
## Improvements

Output classes can be changed to 'Negative' and 'Positive' to increase ease of use.
