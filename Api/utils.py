import spacy 
def get_nlp():
    return spacy.load('en_core_web_lg')

#use this utility function to preprocess the text
#1. Remove the stop words
#2. Convert to base form using lemmatisation
def preprocess(text):
    nlp=get_nlp()
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    return ' '.join(filtered_tokens)

def vectorize_log(text):
    nlp=get_nlp()
    return nlp(text).vector
