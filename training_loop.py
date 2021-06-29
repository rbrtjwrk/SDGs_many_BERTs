# IMPORTS

import pandas as pd
from transformers import BertConfig, BertTokenizer
from nltk import tokenize
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor
from transformers import TFBertModel, BertConfig
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.metrics import BinaryAccuracy, Precision, Recall


# SET PARAMETERS

DATA_PATH=".../SDGs_merged_cleaned_onehot_no_zeros_no_duplicates_no_t13.h5"

SAVE_MODELS_TO=".../"


# READ DATA

tab=pd.read_hdf(DATA_PATH)


# SLICE DATA

def slice_data(dataframe, label):
    label_data=dataframe[dataframe[label]==1]
    label_data_len=len(label_data)
    temp_data=dataframe.copy()[dataframe[label]!=1].sample(n=label_data_len)
    label_data=label_data[["Abstract", label]]
    label_data=label_data.append(temp_data[["Abstract", label]])
    label_data.columns=["Abstract", "Label"]
    return label_data


# PREPARE DATA FOR BERT

def data_to_values(dataframe):
    abstracts=dataframe.Abstract.values
    labels=dataframe.Label.values
    return abstracts, labels


def tokenize_abstracts(abstracts):
    t_abstracts=[]
    for abstract in abstracts:
        t_abstract="[CLS] "
        for sentence in tokenize.sent_tokenize(abstract):
            t_abstract=t_abstract + sentence + " [SEP] "
        t_abstracts.append(t_abstract)
    return t_abstracts


tokenizer=BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

def b_tokenize_abstracts(t_abstracts, max_len=512):
    b_t_abstracts=[tokenizer.tokenize(_)[:max_len] for _ in t_abstracts]
    return b_t_abstracts


def convert_to_ids(b_t_abstracts):
    input_ids=[tokenizer.convert_tokens_to_ids(_) for _ in b_t_abstracts]
    return input_ids


def pad_ids(input_ids, max_len=512):
    p_input_ids=pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")
    return p_input_ids


def create_attention_masks(inputs):
    masks=[]
    for seq in inputs:
        seq_mask=[float(i>0) for i in seq]
        masks.append(seq_mask)
    return masks


# CREATE MODEL

def create_model(label):
    config=BertConfig.from_pretrained("bert-base-multilingual-uncased", num_labels=2)
    bert=TFBertModel.from_pretrained("bert-base-multilingual-uncased", config=config)
    bert_layer=bert.layers[0]
    input_ids_layer=Input(
                        shape=(512),
                        name="input_ids",
                        dtype="int32")
    input_attention_masks_layer=Input(
                                    shape=(512),
                                    name="attention_masks",
                                    dtype="int32")
    bert_model=bert_layer(input_ids_layer, input_attention_masks_layer)
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

histories=[]
test_scores=[]

for _ in tab.columns[4:]:
    print(f"PROCESSING TARGET {_}...")
    data=slice_data(tab, _)
    print("Data sliced.")
    abstracts, labels=data_to_values(data)
    tokenized_abstracts=tokenize_abstracts(abstracts)
    b_tokenized_abstracts=b_tokenize_abstracts(tokenized_abstracts)
    print("Abstracts tokenized.")
    ids=convert_to_ids(b_tokenized_abstracts)
    print("Tokens converted to ids.")
    padded_ids=pad_ids(ids)
    print("Sequences padded.")
    train_inputs, temp_inputs, train_labels, temp_labels=train_test_split(padded_ids, labels, random_state=1993, test_size=0.3)
    validation_inputs, test_inputs, validation_labels, test_labels=train_test_split(temp_inputs, temp_labels, random_state=1993, test_size=0.5)
    print("Data splited into train, validation, test sets.")
    train_masks=create_attention_masks(train_inputs)
    validation_masks=create_attention_masks(validation_inputs)
    test_masks=create_attention_masks(test_inputs)
    print("Attention masks created.")
    train_inputs=convert_to_tensor(train_inputs)
    validation_inputs=convert_to_tensor(validation_inputs)
    test_inputs=convert_to_tensor(test_inputs)
    print("Inputs converted to tensors.")
    train_labels=convert_to_tensor(train_labels)
    validation_labels=convert_to_tensor(validation_labels)
    test_labels=convert_to_tensor(test_labels)
    print("Labels converted to tensors.")
    train_masks=convert_to_tensor(train_masks) 
    validation_masks=convert_to_tensor(validation_masks)
    test_masks=convert_to_tensor(test_masks)
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
    test_scores.append(test_score)
    print(f"""Model for target {_} tested.
    .
    .
    .""")  


# SAVE STATISTICS

stats=pd.DataFrame(list(zip(tab.columns[4:], histories, test_scores)), columns=["target", "history", "test_score"])
stats.to_excel(SAVE_MODELS_TO+"_stats.xlsx", index=False)


