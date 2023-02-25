# -----------------------------------------------------------------------------
# 1. Import relevant modules
# -----------------------------------------------------------------------------
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score

import rampwf as rw
from rampwf.score_types.base import ClassifierBaseScoreType
from rampwf.workflows import Classifier

from prepare_data import get_train_test

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
    
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    for train_index, test_index in kf.split(X):
        yield train_index, test_index

# -----------------------------------------------------------------------------
# 7. Provide the I/O methods : Training / testing data reader
# -----------------------------------------------------------------------------
# Read data is implemented in prepare_data : get_train_test

def get_train_data():
    X_train, _, y_train, _ = get_train_test("public")
    return X_train, y_train

def get_test_data():
    _, X_test, _, y_test = get_train_test("public")
    return X_test, y_test

