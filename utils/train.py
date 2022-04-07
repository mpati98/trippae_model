from utils import preprocess
from preprocess import *
# device init
device = 'cpu'
epochs = 10

# file path
bpe_code_path = "/home/mpati/Documents/Trippae/phoBERT_Sentinent/transformers/PhoBERT_base_transformers/bpe.codes"
dict_path = "/home/mpati/Documents/Trippae/phoBERT_Sentinent/transformers/PhoBERT_base_transformers/dict.txt"
VnCoreNLP_path = "/home/mpati/Documents/Trippae/phoBERT_Sentinent/transformers/vncorenlp/VnCoreNLP-1.1.1.jar"
train_path = '/home/mpati/Documents/Trippae/phoBERT_Sentinent/data/train.crash'
test_path = '/home/mpati/Documents/Trippae/phoBERT_Sentinent/data/test.crash'
config_path = "/home/mpati/Documents/Trippae/phoBERT_Sentinent/transformers/PhoBERT_base_transformers/config.json"
pretrain_model_path = "/home/mpati/Documents/Trippae/phoBERT_Sentinent/models/PhoBERT_Sentinent.pt"
model_save_path = "/home/mpati/Documents/Trippae/phoBERT_Sentinent/models/PhoBERT_Sentinent.pt"

# Load encoder and Dic
bpe, vocab, rdrsegmenter = load_encoder_dict(bpe_code_path, dict_path, VnCoreNLP_path)

# Doc du lieu  huan luyen va preprocess voi wordsegmenter
train_id, train_text, train_label, test_id, test_text = read_data(train_path, test_path)

# Make dataloader
train_dataloader, val_dataloader = make_train_dataloader(train_id, train_text, train_label)

# Load model
Bert_model = load_pretrain_model(config_path, pretrain_model_path)

Bert_model = train_model(Bert_model, train_dataloader, val_dataloader)

# Save model
torch.save(Bert_model.state_dict(), model_save_path)