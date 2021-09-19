import pandas as pd
import glob
from nltk import tokenize
from transformers import BertTokenizer, TFBertModel, BertConfig
from transformers.utils.dummy_tf_objects import TFBertMainLayer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import convert_to_tensor
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.metrics import BinaryAccuracy, Precision, Recall


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
    for seq in inputs:
        seq_mask=[float(i>0) for i in seq]
        masks.append(seq_mask)
    return masks


def float_to_percent(float, decimal=3):
    """Takes a float from range 0. to 0.9... as an input
    and converts it to a percentage with specified decimal places.
    """
    return str(float*100)[:(decimal+3)]+"%"

  
def models_predict(directory, inputs, attention_masks, float_to_percent=False):
    """Loads separate .h5 models from a given directory.
    For predictions, inputs are expected to be:
    tensors of token's ids (bert vocab) and tensors of attention masks.
    Output is of format:
    {'model/target N': [the probability of a text N dealing with the target N , ...], ...}
    """
    models=glob.glob(f"{directory}*.h5")
    predictions_dict={}
    for _ in models:
        model=tf.keras.models.load_model(_)
        predictions=model.predict_step([inputs, attention_masks])
        predictions=[float(_) for _ in predictions]
        if float_to_percent==True:
            predictions=[float_to_percent(_) for _ in predictions]
        predictions_dict[model.name]=predictions
        del predictions, model
    return predictions_dict

  
def predictions_dict_to_df(predictions_dictionary):
    """Converts model's predictions of format:
    {'model/target N': [the probability of a text N dealing with the target N , ...], ...}
    to a dataframe of format:
    | text N | the probability of the text N dealing with the target N | ... |
    """
    predictions_df=pd.DataFrame(predictions_dictionary)
    predictions_df.columns=[_.replace("model_", "").replace("_", ".") for _ in predictions_df.columns]
    predictions_df.insert(0, column="text", value=[_ for _ in range(len(predictions_df))])
    return predictions_df
  

def predictions_above_treshold(predictions_dataframe, treshold=0.95):
    """Filters predictions above specified treshold.
    Input is expected to be a dataframe of format:
    | text N | the probability of the text N dealing with the target N | ... |
    Output is of format:
    {text N: [target N dealing with probability > trheshold with text N, ...], ...}
    """
    above_treshold_dict={}
    above_treshold=predictions_dataframe.iloc[:,1:].apply(lambda row: row[row > treshold].index, axis=1)
    for _ in range(len(above_treshold)):
        above_treshold_dict[_]=list(above_treshold[_])
    return above_treshold_dict


