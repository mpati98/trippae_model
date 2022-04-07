from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from vncorenlp import VnCoreNLP
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW
import numpy as np
import random
from tqdm import tqdm_notebook

# Load encoder and Dic
# bpe_code_path = "/home/mpati/Documents/Trippae/phoBERT_Sentinent/transformers/PhoBERT_base_transformers/bpe.codes"
# dict_path = "/home/mpati/Documents/Trippae/phoBERT_Sentinent/transformers/PhoBERT_base_transformers/dict.txt"
# VnCoreNLP_path = "/home/mpati/Documents/Trippae/phoBERT_Sentinent/transformers/vncorenlp/VnCoreNLP-1.1.1.jar"

def load_encoder_dict(bpe_code_path, dict_path,VnCoreNLP_path ):
    bpe = fastBPE(bpe_code_path)
    # Load the dictionary
    vocab = Dictionary()
    vocab.add_from_file(dict_path)
    rdrsegmenter = VnCoreNLP(VnCoreNLP_path, annotators="wseg", max_heap_size='-Xmx500m') 
    return bpe, vocab, rdrsegmenter

# Doc du lieu  huan luyen va preprocess voi wordsegmenter
# train_path = '/home/mpati/Documents/Trippae/phoBERT_Sentinent/data/train.crash'
# test_path = '/home/mpati/Documents/Trippae/phoBERT_Sentinent/data/test.crash'
def read_data(train_path, test_path):
    import re

    # train_path = '/content/drive/MyDrive/NLP/PhoBERT_Sentinent/data/train.crash'
    # test_path = '/content/drive/MyDrive/NLP/PhoBERT_Sentinent/data/test.crash'

    train_id, train_text, train_label = [], [], []
    test_id, test_text = [], []

    with open(train_path, 'r') as f_r:
        data = f_r.read().strip()

        data = re.findall('train_[\s\S]+?\"\n[01]\n\n', data)

        for sample in data:
            splits = sample.strip().split('\n')

            id = splits[0]
            label = int(splits[-1])
            text = ' '.join(splits[1:-1])[1:-1]
            text = rdrsegmenter.tokenize(text)
            text = ' '.join([' '.join(x) for x in text])

            train_id.append(id)
            train_text.append(text)
            train_label.append(label)


    with open(test_path, 'r') as f_r:
        data = f_r.read().strip()
        data = re.findall('train_[\s\S]+?\"\n[01]\n\n', data)

        for sample in data:
            splits = sample.strip().split('\n')

            id = splits[0]
            text = ' '.join(splits[1:])[1:-1]
            text = rdrsegmenter.tokenize(text)
            text = ' '.join([' '.join(x) for x in text])

            test_id.append(id)
            test_text.append(text)
    
    return train_id, train_text, train_label, test_id, test_text

# Make dataloader
def make_train_dataloader(train_id, train_text, train_label):
    train_sents, val_sents, train_labels, val_labels = train_test_split(train_text, train_label, test_size=0.1)

    MAX_LEN = 125

    train_ids = []
    for sent in train_sents:
        subwords = '<s> ' + bpe.encode(sent) + ' </s>'
        encoded_sent = vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
        train_ids.append(encoded_sent)

    val_ids = []
    for sent in val_sents:
        subwords = '<s> ' + bpe.encode(sent) + ' </s>'
        encoded_sent = vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
        val_ids.append(encoded_sent)
        
    train_ids = pad_sequences(train_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
    val_ids = pad_sequences(val_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
    # Tao mot mask co gia tri 0, 1 lam dau vao cho transformer - cho biet gia tri cua chuoi da duoc padding
    train_masks = []
    for sent in train_ids:
        mask = [int(token_id > 0) for token_id in sent]
        train_masks.append(mask)

    val_masks = []
    for sent in val_ids:
        mask = [int(token_id > 0) for token_id in sent]

        val_masks.append(mask)
    
    # Tao dataloader
    train_inputs = torch.tensor(train_ids)
    val_inputs = torch.tensor(val_ids)
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)
    train_masks = torch.tensor(train_masks)
    val_masks = torch.tensor(val_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=32)
    return train_dataloader, val_dataloader

# Load model
# config_path = "/home/mpati/Documents/Trippae/phoBERT_Sentinent/transformers/PhoBERT_base_transformers/config.json"
# pretrain_model_path = "/home/mpati/Documents/Trippae/phoBERT_Sentinent/models/PhoBERT_Sentinent.pt"
def load_pretrain_model(config_path, pretrain_model_path):
    config = RobertaConfig.from_pretrained(
        config_path, from_tf=False, num_labels = 2, output_hidden_states=False, map_location=torch.device('cpu')
    )
    BERT_model = RobertaForSequenceClassification.from_pretrained(
        pretrain_model_path,
        config=config
    )
    return BERT_model

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train_model(Bert_model, train_dataloader, val_dataloader):
    param_optimizer = list(Bert_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False)


    for epoch_i in range(0, epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        total_loss = 0
        Bert_model.train()
        train_accuracy = 0
        nb_train_steps = 0
        predictions , true_labels = [], []
        for step, batch in tqdm_notebook(enumerate(train_dataloader)):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            Bert_model.zero_grad()
            outputs = Bert_model(b_input_ids, 
                token_type_ids=None, 
                attention_mask=b_input_mask, 
                labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()
            
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            true_labels.append(label_ids)
            tmp_train_accuracy = flat_accuracy(logits, label_ids)
            train_accuracy += tmp_train_accuracy
            nb_train_steps += 1
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(Bert_model.parameters(), 1.0)
            optimizer.step()
            
        avg_train_loss = total_loss / len(train_dataloader)
        print(" Accuracy: {0:.4f}".format(train_accuracy/nb_train_steps))
        print(" Average training loss: {0:.4f}".format(avg_train_loss))

        print("Running Validation...")
        Bert_model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in tqdm_notebook(val_dataloader):

            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = Bert_model(b_input_ids, 
                token_type_ids=None, 
                attention_mask=b_input_mask)
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_accuracy = flat_accuracy(logits, label_ids)

                eval_accuracy += tmp_eval_accuracy
                # eval_f1 += tmp_eval_f1
                nb_eval_steps += 1
        print(" Accuracy: {0:.4f}".format(eval_accuracy/nb_eval_steps))
    print("Training complete!")
    return Bert_model

