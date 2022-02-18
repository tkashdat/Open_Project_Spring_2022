# ----------------------------------------------------------------------------------------------------------------------
# Name:         main.py
# Purpose:      Open Project Fall - 2022: Predict mortality with medical data
#               CodaLab competition: https://competitions.codalab.org/competitions/27605#learn_the_details
#
# Author(s):    David Little
#
# Created:      12/17/2021
# Updated:      12/17/2021
# Update Comment(s):
#
# TO DO:
#
# FUTURE WORK:
#
# BUGS TO FIX:
#
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd

# _______________________ Import __________________________________________________________
all_data = pd.read_csv('mimic_synthetic_train.csv', delimiter=' ', header=None)
col_names = pd.read_csv('mimic_synthetic_feat.csv', delimiter=' ', header=None)
all_data = all_data.iloc[:,1:]
all_data.set_axis(col_names, axis=1, inplace=True)

labels = pd.read_csv('mimic_synthetic_train_labels.csv', delimiter=' ', header=None)
all_data['DIED'] = labels
all_data.fillna('NA', inplace=True)

# _______________________ Identify constant columns_________________________________
# non_dups = []
# for column in all_data:
#     if all_data[column].astype(str).min() == all_data[column].astype(str).max():
#         non_dups.append(column)
#
# all_data.drop(non_dups, axis=1, inplace=True)

non_dups = []
for column in all_data:
    if all_data[column].unique().size == 1:
        non_dups.append(column)

all_data.drop(non_dups, axis=1, inplace=True)

# _______________________ Drop non-informative _________________________________
all_data = all_data.iloc[:,4:]

# _______________________ Just the categorical _________________________________
categoricals =  []
for column in all_data:
    if isinstance(all_data[column][0], str):
        categoricals.append(column)

cats = all_data[categoricals]
all_data.drop(cats, axis=1, inplace=True)

#____________________________ One-hot encoding_________________________________
from sklearn.preprocessing import OneHotEncoder

# Load encoder
enc = OneHotEncoder(handle_unknown='ignore')
# Fit encoding
enc.fit(cats)
# Make conversion
feat = enc.transform(cats).toarray()
feat_names = enc.get_feature_names()
cat_data = pd.DataFrame(feat, columns=feat_names)

all_data = pd.concat([cat_data,all_data], axis=1)

#_______________________test_train split_________________________________________


#____________________________ Upsampling _________________________________



# _______________________ Modeling _________________________________

import xgboost as xgb

dtrain = xgb.DMatrix(all_data_train.iloc[:,:-1], enable_categorical=True, label=all_data_train['DIED'])

print("Booster parameters")
param = {'max_depth': 10, 'eta': 0.2, 'objective': 'binary:hinge'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
print("train xgboost")
num_round = 20
cls = xgb.train(param, dtrain, num_round)
#bst.save_model('xgboost.model')
dtest = xgb.DMatrix(X_test)

preds = cls.predict(dtest)

from sklearn.metrics import f1_score

acc = f1_score(y_test, preds)
print(acc)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, preds)
print(acc)


from sklearn.metrics import balanced_accuracy_score
acc = balanced_accuracy_score(y_test, preds)
print(acc)


xgb.plot_importance(cls, max_num_features=80,grid=False)

#______________________________________ Predict test case & save _________________________________________



