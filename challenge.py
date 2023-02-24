import pandas as pd 
import numpy as np

def get_data(file='Pub\BBC.txt'):
    data = dict()
    data["Label"] = []
    keys = list(range(1,4126))
    for key in keys:
        data[key] = []

    # Open the file for reading
    with open(file, 'r') as f:

        # Loop through each line in the file
        for line in f:
            # Remove any leading or trailing whitespace from the line
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Split the line into binary classification and list of (column id, value) pairs
            parts = line.split()
            label = float(parts[0])
            features = [list(map(float, x.split(':'))) for x in parts[1:]]
            features = dict([tuple([int(feature[0]), feature[1]]) for feature in features])

            # Go over the keys and the features
            for key in keys:
                if key in features.keys():
                    data[key].append(features[key])
                else:
                    data[key].append(np.nan)

    return data


if __name__ == '__main__':
    data = get_data()
    print(data)