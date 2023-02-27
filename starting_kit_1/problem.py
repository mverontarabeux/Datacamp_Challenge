# -----------------------------------------------------------------------------
# 1. Import relevant modules
# -----------------------------------------------------------------------------
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score

import rampwf as rw
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from rampwf.workflows import Classifier

# from problem import get_train_test

# from rampwf.utils.importing import import_module_from_source
import os
import pandas as pd 
import numpy as np
import time
import os
from typing import List

from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 2. Providing a title
# -----------------------------------------------------------------------------
problem_title = "TV Commercial Classification Challenge"

# -----------------------------------------------------------------------------
# 3. Prediction type : 2 classes (0,1)
# -----------------------------------------------------------------------------
Predictions = rw.prediction_types.make_multiclass(label_names=[0, 1])

# -----------------------------------------------------------------------------
# 4. Workflow definition : basic classification
# -----------------------------------------------------------------------------
workflow = Classifier()

# -----------------------------------------------------------------------------
# 5. Score types
# -----------------------------------------------------------------------------
class PointwisePrecision(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="pw_prec", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        score = precision_score(y_true_label_index, y_pred_label_index)
        return score


class PointwiseRecall(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="pw_rec", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        score = recall_score(y_true_label_index, y_pred_label_index)
        return score

score_types = [
    rw.score_types.ROCAUC(name='auc'),
    rw.score_types.BalancedAccuracy(name='bal_acc'),
    rw.score_types.F1Above(name='f1', threshold=0.5),
    PointwisePrecision(), 
    PointwiseRecall()
]

# -----------------------------------------------------------------------------
# 6. Cross-validation scheme
# -----------------------------------------------------------------------------

def get_cv(X, y, n_splits=5, shuffle=True, random_state=42):
    """This function returns a cross-validation generator 
    for a given dataset.

    Parameters
    ----------
    X : pandas dataframe
        Features dataset
    y pandas series
        Target dataset
    n_splits : int, optional
        Number of folds to generate. Default is 5.
    shuffle : bool, optional
        Whether or not to shuffle the data before splitting. Default is True.
    random_state : int, optional
        Random seed for shuffling the data. Default is 42.

    Returns:
    -------
    A generator that yields indices to split data into training and test sets.
    """
    
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)

# -----------------------------------------------------------------------------
# 7. Provide the I/O methods : Training / testing data reader
# -----------------------------------------------------------------------------
# Read data is implemented in prepare_data : get_train_test

TXT_FILES_NAMES = {
    "public":["BBC", "CNN", "CNNIBN", "NDTV"],
    "private":["TIMESNOW"],
    "test":["BBC"]
}
FEATHER_FILES_NAMES = {
    "train":"train.feather",
    "test":"test.feather"
}
PATH="."


def check_files(mode:str="public") -> bool:
    """Return true if all the needed files exists 

    Parameters
    ----------
    mode : str, optional
        where to look, by default public

    Returns
    -------
    bool
    """
    already_created=True
    for file in FEATHER_FILES_NAMES.keys():
        if mode=="public":
            path_file = os.path.join(PATH,
                                     "data", "public", 
                                     FEATHER_FILES_NAMES[file])
        else:
            path_file = os.path.join(PATH,
                                     "data", 
                                     FEATHER_FILES_NAMES[file])
        if not os.path.exists(path_file):
            already_created=False
            break
    return already_created

def get_text_columns():
    """Set the bins columns
    Motion Distribution (40 bins)
    Frame Difference Distribution (32 bins)
    Text area distribution (15 bins Mean and 15 bins for variance)
    Bag of Audio Words (4000 bins)
    """

    DICT_COLUMNS = {
        "label":"label",
        "channel":"channel",
        1:"shot_length",
        2:"motion_distribution_mean",
        3:"motion_distribution_variance",
        4:"frame_difference_distribution_mean",
        5:"frame_difference_distribution_variance",
        6:"short_time_energy_mean",
        7:"short_time_energy_variance",
        8:"ZCR_mean",
        9:"ZCR_variance",
        10:"spectral_centroid_mean",
        11:"spectral_centroid_variance",
        12:"spectral_roll_off_mean",
        13:"spectral_roll_off_variance",
        14:"spectral_flux_mean",
        15:"spectral_flux_variance",
        16:"fundamental_frequency_mean",
        17:"fundamental_frequency_variance",
        4124:"edge_change_ratio_mean",
        4125:"edge_change_ratio_variance"
    }

    motions = [f"motion_distribution_bin_{bin}" for bin in range(0,41)]
    for i, motion_bin in enumerate(motions, start=18):
        DICT_COLUMNS[i] = motion_bin

    frames = [f"frame_difference_distribution_bin_{bin}" for bin in range(0,33)]
    for i, frame_bin in enumerate(frames, start=59):
        DICT_COLUMNS[i] = frame_bin

    text_means = [f"text_area_distribution_mean_bin_{bin}" for bin in range(0,16)]
    for i, text_bin in enumerate(text_means, start=92):
        DICT_COLUMNS[i] = text_bin

    text_vars = [f"text_area_distribution_var_bin_{bin}" for bin in range(0,16)]
    for i, text_bin in enumerate(text_vars, start=107):
        DICT_COLUMNS[i] = text_bin

    bags = [f"bag_of_audio_words_{bag}" for bag in range(0,4001)]
    for i, bag in enumerate(bags, start=123):
        DICT_COLUMNS[i] = bag

    return DICT_COLUMNS

def get_data_from_txt(file:str='data\BBC.txt') -> pd.DataFrame:
    """ Convert a txt file of the TV News Channel Commercial Detection Dataset  
    to a pandas dataframe
    The zip file containing the data can be downloaded at
    https://archive.ics.uci.edu/ml/machine-learning-databases/00326/ 
    Each line contains the label and a serie of couple (feature_id, value)

    Parameters
    ----------
    file : str, 
        the path of the file to convert, by default 'data\BBC.txt'

    Returns
    -------
    pd.DataFrame 
        The dataframe containing all the .txt data
    """
    data = dict()
    data["label"] = []
    keys = list(range(1,4126))
    for key in keys:
        data[key] = []

    file_name = file.split(sep="\\")[1]
    begin = time.time()
    cpt = 0
    # Open the file for reading
    with open(file, 'r') as f:
        
        # Loop through each line in the file
        for line in f:
            # Remove any leading or trailing whitespace from the line
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Split the line into binary classification 
            # and list of (column id, value) pairs
            parts = line.split()
            data["label"].append(int(parts[0]))
            features = [list(map(float, x.split(':'))) for x in parts[1:]]
            # Convert to dict the previous list 
            features = dict([tuple([int(feature[0]), feature[1]]) for feature in features])

            # Go over the keys and the features
            for key in keys:
                if key in features.keys():
                    data[key].append(features[key])
                else:
                    data[key].append(np.nan)
            
            cpt += 1
            if cpt % 5000 == 0:
                print(f"{cpt} lines extracted from {file_name}")
                      
    print(f"{cpt} lines extracted from {file_name} - {int(time.time()-begin)} seconds\n")

    # Format the output into dataframe with text columns
    data = pd.DataFrame(data)
    DICT_COLUMNS = get_text_columns()
    data.columns = [DICT_COLUMNS[key] for key in data.keys()]

    return data

def create_train_test_feather(mode:str='public', 
                              test_size:float=0.20) -> None:
    """ Converts each file in the files_list to dataframe using get_data_from_txt
    Creates the concatenated version as one dataframe
    Split into train and test
    Finally, save the dataframes as {datafilename}.feather  : pyarrow needed

    Parameters
    ----------
    mode: str, public or private
        if the files have to be public or private
    test_size : float
        train_test_split argument 

    Returns
    -------
    None
    """

    # if one file does not exists, recreate all files
    check_exist = check_files(mode=mode)
    
    if check_exist:
        print(f"Train and test files already exist. Can be used for modelling")
        
    else:
        list_df_temp_files = []
        for file in TXT_FILES_NAMES[mode]:
            # Load each file one by one
            df_temp_file = get_data_from_txt(f"data\{file}.txt")
            # Add the channel name
            df_temp_file["channel"] = [file]*len(df_temp_file["label"])
            list_df_temp_files.append(df_temp_file)

        # Concatenate the list of df and save as feather
        df_all_train_files = pd.concat(list_df_temp_files)
        df_all_train_files.reset_index(inplace=True, drop=True)

        # Split train/test
        train, test = train_test_split(df_all_train_files,
                                        test_size=test_size, 
                                        random_state=42)
        
        if mode=="public":
            path_file = os.path.join(PATH, "data", "public")
        else:
            path_file = os.path.join(PATH, "data")

        print('col', train.columns)

        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        train.to_feather(os.path.join(path_file, FEATHER_FILES_NAMES["train"]))
        test.to_feather(os.path.join(path_file, FEATHER_FILES_NAMES["test"]))

    return None

def get_train_test(mode:str="public") -> pd.DataFrame:
    """Get the train and test data contained in filename 
    Returns it as distincts features and labels

    Parameters
    ----------
    mode : str, optional
        where to look, by default public
        
    Returns
    -------
    pd.DataFrame
        X_train, X_test, y_train, y_test
    """

    # if one file does not exists, recreate all files
    check_exist = check_files(mode=mode)
    if not check_exist:
        create_train_test_feather(mode=mode)
        
    if mode=="public":
        path_file = os.path.join(PATH, "data", "public")
    else:
        path_file = os.path.join(PATH, "data")

    train = pd.read_feather(os.path.join(path_file, 
                                         FEATHER_FILES_NAMES["train"])
    )
    # print('shape', len(train.index))
    test = pd.read_feather(os.path.join(path_file, 
                                        FEATHER_FILES_NAMES["test"])
    )

    # Split between data and labels
    X_train = train.drop("label", inplace=False, axis=1)
    X_test = test.drop("label", inplace=False, axis=1)
    y_train = train["label"]
    y_test = test["label"]
    # Set the labels as (0,1) instead of (-1,1)
    y_train = y_train.replace(-1,0)
    y_test = y_test.replace(-1,0)
    return X_train, X_test, y_train, y_test
    
def get_train_test_private() -> pd.DataFrame:
    """Get the private (scoring) data  
    Returns it as distincts features and labels
    for the train and test set 

    Returns
    -------
    pd.DataFrame
        X_train, X_test, y_train, y_test
    """
    return get_train_test("private")

def get_train_test_public() -> pd.DataFrame:
    """Get the public data  
    Returns it as distincts features and labels 
    for the train and test set 

    Returns
    -------
    pd.DataFrame
        X_train, X_test, y_train, y_test
    """
    return get_train_test("public")

def get_train_data(path="."):
    X_train, _, y_train, _ = get_train_test_public()
    return X_train, y_train

def get_test_data(path="."):
    _, X_test, _, y_test = get_train_test_public()
    return X_test, y_test

