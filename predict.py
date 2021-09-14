import pandas as pd
import glob
import tensorflow as tf

def float_to_percent(float, decimal=3):
    """Takes a float from range 0. to 0.9... as input
    and converts it to a percentage with specified decimal places."""
    return str(float*100)[:(decimal+3)]+"%"

  
def models_predict(directory, inputs, attention_masks, float_to_percent=False):
    """This function loads separate .h5 models from a given directory.
    For predictions, inputs are expected to be:
    tensors of token's ids (bert vocab) and tensors of attention masks.
    Output is of format:
    {'model/target N': [the probability of a text N dealing with the target N , ...],}"""
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
    {'model/target N': [the probability of a text N dealing with the target N , ...],}
    to a dataframe of format:
    | text N | the probability of the text N dealing with the target N | ... |"""
    predictions_tab=pd.DataFrame(predictions_dictionary)
    predictions_tab.columns=[_.replace("model_", "").replace("_", ".") for _ in predictions_tab.columns]
    predictions_tab.insert(0, column="text", value=[_ for _ in range(len(predictions_tab))])
    return predictions_tab
  
