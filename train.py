# IMPORTS

import pandas as pd
from nltk import tokenize
import time
from sklearn.model_selection import train_test_split
from transformers import BertConfig, BertTokenizer, TFBertModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall


# SET PARAMETERS

DATA_PATH="..."

SAVE_MODELS_TO=".../"


# READ DATA

tab=pd.read_hdf(DATA_PATH)


# SLICE DATA

def slice_data(dataframe, label):
    """Slices dataframe of a structure:
    | text/abstract | label |
    Prepares data for a binary classification
    training. For a given label, creates new
    dataset where number of items belonging
    to the given label equals number of randomly
    generated items from all the other labels items.
    """
    label_data=dataframe[dataframe[label]==1]
    label_data_len=len(label_data)
    temp_data=dataframe.copy()[dataframe[label]!=1].sample(n=label_data_len)
    label_data=label_data[["Abstract", label]]
    label_data=label_data.append(temp_data[["Abstract", label]])
    label_data.columns=["Abstract", "Label"]
    return label_data


# PREPARE DATA FOR BERT

def data_to_values(dataframe):
    """Converts data to values.
    """
    abstracts=dataframe.Abstract.values
    labels=dataframe.Label.values
    return abstracts, labels


def tokenize_abstracts(abstracts):
    """For a given texts, adds '[CLS]' and '[SEP]' tokens
    at the beginning and the end of each sentence, respectively.
    """
    t_abstracts=[]
    for abstract in abstracts:
        t_abstract="[CLS] "
        for sentence in tokenize.sent_tokenize(abstract):
            t_abstract=t_abstract + sentence + " [SEP] "
        t_abstracts.append(t_abstract)
    return t_abstracts


tokenizer=BertTokenizer.from_pretrained('bert-base-multilingual-uncased')


def b_tokenize_abstracts(t_abstracts, max_len=512):
    """Tokenizes sentences with the help
    of a 'bert-base-multilingual-uncased' tokenizer.
    """
    b_t_abstracts=[tokenizer.tokenize(_)[:max_len] for _ in t_abstracts]
    return b_t_abstracts


def convert_to_ids(b_t_abstracts):
    """Converts tokens to its specific
    IDs in a bert vocabulary.
    """
    input_ids=[tokenizer.convert_tokens_to_ids(_) for _ in b_t_abstracts]
    return input_ids


def abstracts_to_ids(abstracts):
    """Tokenizes abstracts and converts
    tokens to their specific IDs
    in a bert vocabulary.
    """
    tokenized_abstracts=tokenize_abstracts(abstracts)
    b_tokenized_abstracts=b_tokenize_abstracts(tokenized_abstracts)
    ids=convert_to_ids(b_tokenized_abstracts)
    return ids


def pad_ids(input_ids, max_len=512):
    """Padds sequences of a given IDs.
    """
    p_input_ids=pad_sequences(input_ids,
                              maxlen=max_len,
                              dtype="long",
                              truncating="post",
                              padding="post")
    return p_input_ids


def create_attention_masks(inputs):
    """Creates attention masks
    for a given seuquences.
    """
    masks=[]
    for sequence in inputs:
        sequence_mask=[float(_>0) for _ in sequence]
        masks.append(sequence_mask)
    return masks


# CREATE MODEL

def create_model(label):
    config=BertConfig.from_pretrained(
                                    "bert-base-multilingual-uncased",
                                     num_labels=2,
                                     hidden_dropout_prob=0.2,
                                     attention_probs_dropout_prob=0.2)
    bert=TFBertModel.from_pretrained(
                                    "bert-base-multilingual-uncased",
                                    config=config)
    bert_layer=bert.layers[0]
    input_ids_layer=Input(
                        shape=(512),
                        name="input_ids",
                        dtype="int32")
    input_attention_masks_layer=Input(
                                    shape=(512),
                                    name="attention_masks",
                                    dtype="int32")
    bert_model=bert_layer(
                        input_ids_layer,
                        input_attention_masks_layer)
    target_layer=Dense(
                    units=1,
                    kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
                    name="target_layer",
                    activation="sigmoid")(bert_model[1])
    model=Model(
                inputs=[input_ids_layer, input_attention_masks_layer],
                outputs=target_layer,
                name="model_"+label.replace(".", "_"))
    optimizer=Adam(
                learning_rate=5e-05,
                epsilon=1e-08,
                decay=0.01,
                clipnorm=1.0)
    model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy", 
                metrics=[BinaryAccuracy(), Precision(), Recall()])
    return model


# THE LOOP

test_scores=[]
elapsed_times=[]

for _ in tab.columns[4:]: # here you have to specify the index where label’s columns start
    print(f"PROCESSING TARGET {_}...")
    start_time=time.process_time()
    data=slice_data(tab, _)
    print("Data sliced.")
    abstracts, labels=data_to_values(data)
    ids=abstracts_to_ids(abstracts)
    print("Abstracts tokenized, tokens converted to ids.")
    padded_ids=pad_ids(ids)
    print("Sequences padded.")
    train_inputs, temp_inputs, train_labels, temp_labels=train_test_split(padded_ids, labels, random_state=1993, test_size=0.3)
    validation_inputs, test_inputs, validation_labels, test_labels=train_test_split(temp_inputs, temp_labels, random_state=1993, test_size=0.5)
    print("Data splited into train, validation, test sets.")
    train_masks, validation_masks, test_masks=[create_attention_masks(_) for _ in [train_inputs, validation_inputs, test_inputs]]
    print("Attention masks created.")
    train_inputs, validation_inputs, test inputs=[convert_to_tensor(_) for _ in [train_inputs, validation_inputs, test_inputs]]
    print("Inputs converted to tensors.")
    train_labels, validation_labels, test_labels=[convert_to_tensor(_) for _ in [train_lables, validation_labels, test_labels]]
    print("Labels converted to tensors.")
    train_masks, validation_masks, test_masks=[convert_to_tensor(_) for _ in [train_masks, validation_masks, test_masks]]
    print("Masks converted to tensors.")
    model=create_model(_)
    print("Model initialized.")
    history=model.fit([train_inputs, train_masks], train_labels,
                        batch_size=3,
                        epochs=3,
                        validation_data=([validation_inputs, validation_masks], validation_labels))
    histories.append(history)
    print(f"Model for {_} target trained.")
    model.save(SAVE_MODELS_TO+_.replace(".", "_")+".h5")
    print(f"Model for target {_} saved.")
    test_score=model.evaluate([test_inputs, test_masks], test_labels,
                                batch_size=3)
    elapsed_times.append(time.process_time()-start_time)
    test_scores.append(test_score)
    print(f"""Model for target {_} tested.
    .
    .
    .""")  


# SAVE STATISTICS

stats=pd.DataFrame(test_scores, columns=["loss", "accuracy", "precision", "recall"])
stats.insert(loc=0, "target", tab.columns[4:])
stats.insert(loc=5, "elapsed_time", elapsed_times)
stats.to_excel(SAVE_MODELS_TO+"_stats.xlsx", index=False)


