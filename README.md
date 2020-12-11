## Topic Modelling

Spacy is written in cython language, (C extension of Python designed to give C like performance to the python program). Hence is a quite fast library. spaCy provides a concise API to access its methods and properties governed by trained machine (and deep) learning models.We have used spaCy library to do topic modeling on bio pages and include it in the compiled BIO's.

## Setup

Spacy, its data, and its models can be easily installed using python package index and setup tools. Use the following command to install spacy in your machine:
```bash
#Use the following command to install spacy in your machine
sudo pip install spacy

#To download all the data and models, run the following command, after the installation:
python -m spacy.en.download all

#To load nlp for english. Use the below commands
nlp = spacy.load("en_core_web_sm")
```

## Comparison with other libraries

Spacy is very powerful and industrial strength package for almost all natural language processing tasks.
comparison of Spacy with other famous tools to implement nlp in python – CoreNLP and NLTK.

![Feature Availability](Comparison.PNG)

![Speed: Key Functionalities – Tokenizer, Tagging, Parsing](Speed.PNG)

![Accuracy: Entity Extraction](Accuracy.PNG)


## Part of Speech Tagging

Part-of-speech tags are the properties of the word that are defined by the usage of the word in the grammatically correct sentence. We have used part of speech tagging SPacy property in our code . We also added few text cleaning function to remove unwanted information.



```bash
def isNoise(token):
    is_noise = False
    if token.pos_ in noisy_pos_tags:
        is_noise = True
    elif token.is_stop == True:
        is_noise = True
    elif len(token.string) <= min_token_length:
        is_noise = True
    elif token.string.lower().strip() in stop_words:
        is_noise = True
    elif token.string.strip() in ents:
        is_noise = True
    return is_noise
    

def cleanup(token, lower = True):
    if lower:
        token = token.lower()
    return token.strip()
```

Once the Tokenazation and text cleaning are done. we print top five key words modelled out of SPacy in the compiled BIO pages

```bash
    from collections import Counter	
    cleaned_list = [cleanup(word.string) for word in document if not isNoise(word)]
    counts=Counter(cleaned_list).most_common(5)   
```

## Ouput

Output gets stored in all the compiled BIO files in below format once the NLP is done through SPacy

```bash
Key Words based on Topic Modelling through SPacy:
research	compiler	software	students	projects
```


