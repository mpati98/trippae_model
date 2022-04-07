import spacy
from utils import add_loc_sentiment_funcs
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes', 
    default="/home/mpati/Documents/Trippae/phoBERT_Sentinent/transformers/PhoBERT_base_transformers/bpe.codes",
    required=False,
    type=str,
    help='path to fastBPE BPE'
)
args, unknown = parser.parse_known_args()
data = args
nlp_ner_vn = spacy.load('vi_core_news_lg')


# JSON file
f = open ('test.json', "r")
 
# Reading from file
data = json.loads(f.read())
 
# Iterating through the json
# list
for i in data:
    print(i)
 
# Closing file
f.close()
text = []
pos_loc = []
for sent in list_sent:
    text.append(sent)
    pos = get_input_ner(sent)
    pos_loc.append(pos)

print(type(text))
print(type(pos_loc[0]))