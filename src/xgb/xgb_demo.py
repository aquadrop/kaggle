"""
-------------------------------------------------
   File Name：     xgb_demo
   Description :
   Author :       deep
   date：          18-3-7
-------------------------------------------------
   Change Activity:
                   18-3-7:
                   
   __author__ = 'deep'
-------------------------------------------------
"""

import xgboost as xgb
# read in data
dtrain = xgb.DMatrix('agaricus.txt.train')
dtest = xgb.DMatrix('agaricus.txt.test')
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)

