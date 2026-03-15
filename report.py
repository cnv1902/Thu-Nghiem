import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score

def report(Label, pred):
    precision, recall, fscore, support = score(Label, pred)
    report = {"Label":np.array([-1, 1]),"precision":precision, "recall":recall, "fscore":fscore, "support":support}
    df = pd.DataFrame.from_dict(report)
    return df