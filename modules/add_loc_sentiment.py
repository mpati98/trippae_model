import argparse
from numpy import inner
import spacy
import json
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from transformers import RobertaForSequenceClassification, RobertaConfig
import pandas as pd
from argparse import Namespace

class ADDING_LOC_SENTIMENT:
    def __init__(self,
                MAX_LEN = 125,
                spacy_module = 'vi_core_news_lg',
                bpe_codes_path = "/home/mpati/Documents/Trippae/phoBERT_Sentinent/transformers/PhoBERT_base_transformers/bpe.codes",
                vocab_path = "/home/mpati/Documents/Trippae/phoBERT_Sentinent/transformers/PhoBERT_base_transformers/dict.txt",
                config_path = "/home/mpati/Documents/Trippae/phoBERT_Sentinent/transformers/PhoBERT_base_transformers/config.json",
                model_path = '/home/mpati/Documents/Trippae/phoBERT_Sentinent/models/PhoBERT_Sentinent.pt',
                device=torch.device('cpu')
                ):
        self.nlp_ner_vn = spacy.load(spacy_module)
        self.bpe = fastBPE(Namespace(bpe_codes=bpe_codes_path))
        self.vocab = Dictionary()
        self.vocab.add_from_file(vocab_path)
        self.MAX_LEN = MAX_LEN
        self.config = RobertaConfig.from_pretrained(
                        config_path, 
                        from_tf=False, 
                        num_labels = 2, 
                        output_hidden_states=False, 
                        map_location=device
                    )
        self.Sentiment_model = RobertaForSequenceClassification.from_pretrained(
                                model_path,
                                config=self.config
                    )
        self.device = device
    def get_text_from_json(self, json_file):
        f = open (json_file, "r")
        text = json.loads(f.read())
        f.close()
        return text
    def get_input_ner(self, text):
        input = self.nlp_ner_vn(text)
        result = []
        for token in input:
            if token.tag_ == "Np" and token.dep_ == "obl":
                result.append(token.text)
        return result
    def preprocess_sent(self, sentence):

        subword = '<s> ' + self.bpe.encode(sentence) + ' </s>'
        encod_sent = self.vocab.encode_line(subword, append_eos=True, add_if_not_exist=False).long().tolist()
        sent_ids = pad_sequences([encod_sent], maxlen=self.MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
        sent_ids = torch.tensor(sent_ids)
        return sent_ids
    def predict_sentiment(self, sentence):
        input_pre = self.preprocess_sent(sentence)
        out = self.Sentiment_model(input_pre.to('cpu'))
        logits = out[0]
        logits = logits.detach().cpu().numpy()
        if logits[0][0] < 1:
            return 0
        else: return 1
    def make_data_input(self, inp, loc, sentiment):
        input_dict = {
        "ID": inp["ID"],
        "Comments": inp["Comments"],
        "like": inp["like"],
        "unlike": inp["unlike"],
        "Ner_LOC": loc,
        "Sentiment": sentiment,
        "Cluster": 0
        }
        df_input = pd.DataFrame(input_dict)
        return df_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='/home/mpati/Documents/Trippae/phoBERT_Sentinent/models/PhoBERT_Sentinent.pt', type=str)
    parser.add_argument("--config_path", default="/home/mpati/Documents/Trippae/phoBERT_Sentinent/transformers/PhoBERT_base_transformers/config.json", type=str)
    parser.add_argument("--vocab_path", default="/home/mpati/Documents/Trippae/phoBERT_Sentinent/transformers/PhoBERT_base_transformers/dict.txt", type=str)
    parser.add_argument("--bpe_codes_path", default="/home/mpati/Documents/Trippae/phoBERT_Sentinent/transformers/PhoBERT_base_transformers/bpe.codes", type=str)
    parser.add_argument("--spacy_module", default='vi_core_news_lg', type=str)
    parser.add_argument("--MAX_LEN", default=125, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--json_file", type=str)
    parser.add_argument("--data_path", default="/home/mpati/Documents/Trippae/phoBERT_Sentinent/test.csv", type=str)

    args, unknown = parser.parse_known_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    adding_loc_sentiment = ADDING_LOC_SENTIMENT(args.MAX_LEN, 
                                                args.spacy_module, 
                                                args.bpe_codes_path, 
                                                args.vocab_path,
                                                args.config_path,
                                                args.model_path,
                                                device)

    inp = adding_loc_sentiment.get_text_from_json(json_file=args.json_file)
    loc = adding_loc_sentiment.get_input_ner(inp["Comments"])
    sentiment = adding_loc_sentiment.predict_sentiment(inp["Comments"])
    df_input = adding_loc_sentiment.make_data_input(inp, loc, sentiment)
    df = pd.read_csv(args.data_path)
    df = df.drop(['Unnamed: 0'], axis=1)
    df = pd.concat([df, df_input])
    df.to_csv('demo.csv')
    