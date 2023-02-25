import pandas as pd 
import numpy as np
import time
import os
from typing import List

from sklearn.model_selection import train_test_split

TRAIN_FILES_NAMES = ["BBC", "CNN", "CNNIBN", "NDTV"]
test = ["BBC"]
SCORE_FILES_NAME = ["TIMESNOW"]
TRAIN_TEST_DATA_FEATHER = "train_test_data.feather"
SCORE_DATA_FEATHER = "score_data.feather"


def get_string_columns():
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


def get_data(file:str='Pub\BBC.txt') -> pd.DataFrame:
    """ Convert a txt file of the TV News Channel Commercial Detection Dataset  
    to a pandas dataframe
    The zip file containing the data can be downloaded at
    https://archive.ics.uci.edu/ml/machine-learning-databases/00326/ 
    Each line contains the label and a serie of couple (feature_id, value)

    Parameters
    ----------
    file : str, 
        the path of the file to convert, by default 'Pub\BBC.txt'

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

    
    data = pd.DataFrame(data)
    DICT_COLUMNS = get_string_columns()
    data.columns = [DICT_COLUMNS[key] for key in data.keys()]
    return data


def create_data_file(datafilename:str, 
                     files_list:list=TRAIN_FILES_NAMES,
                     overwrite:bool=True) -> None:
    """ Converts each file in the files_list to dataframe using get_data
    Creates the concatenated version as one dataframe
    Finally, save it as {datafilename}.feather 

    Parameters
    ----------
    datafilename : str
        name of the file to be created
    files_list : list, optional
        list of file names to concatenate as dataframe and to save as feather, 
        by default TRAIN_FILES_NAMES
    overwrite:bool, optional
        if set to true, and datafilename already exists, overwrite it 
    Returns
    -------
    None
    """

    assert isinstance(files_list, list)
    if os.path.exists(datafilename) and not overwrite:
        print(f"{datafilename} already exists. Can be used as source file")
    else:
        list_df_temp_files = []
        for file in files_list:
            # Load each file one by one
            df_temp_file = get_data(f"Pub\{file}.txt")
            # Add the channel name
            df_temp_file["channel"] = [file]*len(df_temp_file["label"])
            list_df_temp_files.append(df_temp_file)

        # Concatenate the list of df and save as feather
        df_all_train_files = pd.concat(list_df_temp_files)
        df_all_train_files.reset_index(inplace=True, drop=True)
        df_all_train_files.to_feather(datafilename) # need pyarrow installed

    return None


def get_train_test_data(filename:str=TRAIN_TEST_DATA_FEATHER, 
                        test_size:float=None) -> pd.DataFrame:
    """Get the train and test data contained in filename 
    Returns it as distincts features and labels

    Parameters
    ----------
    filename : str, optional
        file name from which to feed, by default TRAIN_TEST_DATA_FEATHER
    test_size : float, optional
        test data size compared to the whole data,
        by default None to not split the data

    Returns
    -------
    pd.DataFrame
        X_train, X_test, y_train, y_test
        if test_size is set to None, X_test and y_test are set as None

    Raises
    ------
    Exception
        If filename is not found 
    """
    if not os.path.exists(filename):
        raise Exception(f"{filename} not found. Use create_file_train_test_data to create it")
    else:
        data = pd.read_feather(filename)
        X = data.drop("label", axis=1)
        y = data["label"]
        if not test_size is None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                test_size=test_size, 
                                                                random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, None, y, None
    
    return X_train, X_test, y_train, y_test
    

def get_score_data(filename:str=SCORE_DATA_FEATHER) -> pd.DataFrame:
    """Get the scoring data contained in filename 
    Returns it as distincts features and labels

    Parameters
    ----------
    filename : str, optional
        file name from which to feed, by default SCORE_DATA_FEATHER

    Returns
    -------
    pd.DataFrame
        X and y 

    Raises
    ------
    Exception
        If filename is not found 
    """
    if not os.path.exists(filename):
        raise Exception(f"{filename} not found. Use create_file_train_test_data to create it")
    else:
        data = pd.read_feather(filename)
        X_score = data.drop("label", axis=1)
        y_score = data["label"]
    return X_score, y_score


if __name__ == '__main__':
    # Create the feather files of train test data 
    create_data_file(datafilename=TRAIN_TEST_DATA_FEATHER, 
                     files_list=TRAIN_FILES_NAMES, 
                     overwrite=False)
    # Create the feather files of scoring data 
    create_data_file(datafilename=SCORE_DATA_FEATHER, 
                     files_list=SCORE_FILES_NAME, 
                     overwrite=False)
    
    # Load the data
    X_train, X_test, y_train, y_test = get_train_test_data(test_size=0.25)
    X_score, y_score = get_score_data()

    # Check shapes
    print(f"X_train shape : {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")
    print(f"X_score shape : {X_score.shape}")
    print(f"y_score shape : {y_score.shape}")