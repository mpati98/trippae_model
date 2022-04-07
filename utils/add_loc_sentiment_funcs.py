# Get input sentence
def get_input_ner(sent):
    input = nlp_ner_vn(sent)
    result = []
    for token in input:
        if token.tag_ == "Np" and token.dep_ == "obl":
            result.append(token.text)
    return result

