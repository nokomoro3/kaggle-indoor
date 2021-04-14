import pandas as pd
import numpy as np
import glob
import os
import gc
import json
import lightgbm as lgb
import random
import scipy.stats as stats
from sklearn.model_selection import GroupKFold, KFold

# base_path = '.\\indata'

# # pull out all the buildings actually used in the test set, given current method we don't need the other ones
# ssubm = pd.read_csv('.\\indata\\sample_submission.csv')

# # only 24 of the total buildings are used in the test set, 
# # this allows us to greatly reduce the intial size of the dataset

# ssubm_df = ssubm["site_path_timestamp"].apply(lambda x: pd.Series(x.split("_")))
# used_buildings = sorted(ssubm_df[0].value_counts().index.tolist())

# # dictionary used to map the floor codes to the values used in the submission file. 
# floor_map = {"B2":-2, "B1":-1, "F1":0, "F2": 1, "F3":2, "F4":3, "F5":4, "F6":5, "F7":6,"F8":7, "F9":8,
#              "1F":0, "2F":1, "3F":2, "4F":3, "5F":4, "6F":5, "7F":6, "8F": 7, "9F":8}

# # get only the wifi bssid that occur over 1000 times(this number can be experimented with)
# # these will be the only ones used when constructing features
# bssid = dict()

# for building in used_buildings:
#     break
#     folders = sorted(glob.glob(os.path.join(base_path,'train\\'+building+'\\*')))
#     print(building)
#     wifi = list()
#     for folder in folders:
#         floor = floor_map[folder.split('\\')[-1]]
#         files = glob.glob(os.path.join(folder, "*.txt"))
#         for file in files:
#             with open(file, encoding='utf-8') as f:
#                 txt = f.readlines()
#                 for e, line in enumerate(txt):
#                     tmp = line.strip().split()
#                     if tmp[1] == "TYPE_WIFI":
#                         wifi.append(tmp)
#     df = pd.DataFrame(wifi)
#     #top_bssid = df[3].value_counts().iloc[:500].index.tolist()
#     value_counts = df[3].value_counts()
#     top_bssid = value_counts[value_counts > 1000].index.tolist()
#     print(len(top_bssid))
#     bssid[building] = top_bssid
#     del df
#     del wifi
#     gc.collect()

# with open("bssid_1000.json", "w") as f:
#     json.dump(bssid, f)

# with open("bssid_1000.json") as f:
#     bssid = json.load(f)

# # generate all the training data 
# building_dfs = dict()

# for building in used_buildings:
#     # break
#     folders = sorted(glob.glob(os.path.join(base_path,'train', building +'\\*')))
#     dfs = list()
#     index = sorted(bssid[building]) # all bssid on this site
#     print(building)
#     for folder in folders:
#         floor = floor_map[folder.split('\\')[-1]]
#         files = glob.glob(os.path.join(folder, "*.txt"))
#         print(floor)
#         for file in files:
#             wifi = list()
#             waypoint = list()
#             with open(file, encoding='utf-8') as f:
#                 txt = f.readlines()
#             for line in txt:
#                 line = line.strip().split()
#                 if line[1] == "TYPE_WAYPOINT":
#                     waypoint.append(line)
#                 if line[1] == "TYPE_WIFI":
#                     wifi.append(line)

#             df = pd.DataFrame(np.array(wifi))    

#             # generate a feature, and label for each wifi block
#             for gid, g in df.groupby(0):
#                 dists = list()
#                 for e, k in enumerate(waypoint):
#                     dist = abs(int(gid) - int(k[0]))
#                     dists.append(dist)
#                 nearest_wp_index = np.argmin(dists)
                
#                 g = g.drop_duplicates(subset=3)
#                 tmp = g.iloc[:,3:5] # bssid and rssi
#                 feat = tmp.set_index(3).reindex(index).replace(np.nan, -999).T
#                 feat["x"] = float(waypoint[nearest_wp_index][2])
#                 feat["y"] = float(waypoint[nearest_wp_index][3])
#                 feat["f"] = floor
#                 feat["path"] = file.split('\\')[-1].split('.')[0] # useful for crossvalidation
#                 dfs.append(feat)
                
#     building_df = pd.concat(dfs)
#     building_dfs[building] = df
#     building_df.to_csv(building+"_1000_train.csv")

# # Generate the features for the test set
# ssubm_building_g = ssubm_df.groupby(0)
# feature_dict = dict()

# for gid0, g0 in ssubm_building_g: # loop of site
#     # break
#     index = sorted(bssid[g0.iloc[0,0]])
#     feats = list()
#     print(gid0)
#     for gid,g in g0.groupby(1): # loop of path file

#         # get all wifi time locations, 
#         with open(os.path.join(base_path, 'test\\' + g.iloc[0,1] + '.txt'), encoding='utf-8') as f:
#             txt = f.readlines()

#         wifi = list()

#         for line in txt:
#             line = line.strip().split()
#             if line[1] == "TYPE_WIFI":
#                 wifi.append(line)

#         wifi_df = pd.DataFrame(wifi)
#         wifi_points = pd.DataFrame(wifi_df.groupby(0).count().index.tolist())
        
#         for timepoint in g.iloc[:,2].tolist(): # 

#             deltas = (wifi_points.astype(int) - int(timepoint)).abs()
#             min_delta_idx = deltas.values.argmin()
#             wifi_block_timestamp = wifi_points.iloc[min_delta_idx].values[0]
            
#             wifi_block = wifi_df[wifi_df[0] == wifi_block_timestamp].drop_duplicates(subset=3)
#             feat = wifi_block.set_index(3)[4].reindex(index).fillna(-999)

#             feat['site_path_timestamp'] = g.iloc[0,0] + "_" + g.iloc[0,1] + "_" + timepoint
#             feats.append(feat)
#     feature_df = pd.concat(feats, axis=1).T
#     feature_df.to_csv(gid0+"_1000_test.csv")
#     feature_dict[gid0] = feature_df

N_SPLITS = 10
SEED = 42

from datetime import datetime
class ElapsedTimer():
    def __init__(self) -> None:
        self.mem = datetime.now()
    def get(self) -> int:
        now = datetime.now()
        elapsed = now - self.mem
        self.mem = datetime.now()
        return elapsed

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

# the metric used in this competition
def comp_metric(xhat, yhat, fhat, x, y, f):
    intermediate = np.sqrt(np.power(xhat - x,2) + np.power(yhat-y,2)) + 15 * np.abs(fhat-f)
    return intermediate.sum()/xhat.shape[0]

set_seed(SEED)

feature_dir = ".\\indata\\wifi_features2"

# get our train and test files
train_files = sorted(glob.glob(os.path.join(feature_dir, '*_train.csv')))
test_files = sorted(glob.glob(os.path.join(feature_dir, '*_test.csv')))
ssubm = pd.read_csv('.\\indata\\sample_submission.csv', index_col=0)

lgb_params = {'objective': 'root_mean_squared_error',
              'boosting_type': 'gbdt',
              'n_estimators': 50000,
              'learning_rate': 0.1,
              'num_leaves': 90,
              'colsample_bytree': 0.4,
              'subsample': 0.6,
              'subsample_freq': 2,
              'bagging_seed': SEED,
              'reg_alpha': 8,
              'reg_lambda': 2,
              'random_state': SEED,
              'n_jobs': -1
              }

lgb_f_params = {'objective': 'multiclass',
                'boosting_type': 'gbdt',
                'n_estimators': 50000,
                'learning_rate': 0.1,
                'num_leaves': 90,
                'colsample_bytree': 0.4,
                'subsample': 0.6,
                'subsample_freq': 2,
                'bagging_seed': SEED,
                'reg_alpha': 10,
                'reg_lambda': 2,
                'random_state': SEED,
                'n_jobs': -1
                }

predictions = list()

# loop of sites
y_oof_all = np.empty((0,3))
y_truth_all = np.empty((0,3))

elapsed_timer = ElapsedTimer()
for e, file in enumerate(train_files):

    train_data = pd.read_csv(file, index_col=0).reset_index(drop=True)
    test_data = pd.read_csv(test_files[e], index_col=0).reset_index(drop=True)

    X = train_data.iloc[:,:-4]
    y = train_data[["x", "y", "f"]]
    y_oof = np.zeros(y.values.shape)
    y_test = np.zeros((N_SPLITS, len(test_data), 3))
    y_result = np.zeros((len(test_data), 3))

    # cross validation and predict test data
    path_unique = train_data["path"].unique()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    elapsed_timer_cv = ElapsedTimer()
    for fold, (train_index, valid_index) in enumerate(kf.split(X)):
    # for fold, (train_group_index, valid_group_index) in enumerate(kf.split(path_unique)):

        # train_groups, valid_groups = path_unique[train_group_index], path_unique[valid_group_index]
        # is_train = train_data["path"].isin(train_groups)
        # is_valid = train_data["path"].isin(valid_groups)

        # X_train, X_valid = X[is_train], X[is_valid]
        # y_train, y_valid = y[is_train], y[is_valid]

        X_train, X_valid = X.iloc[train_index,:], X.iloc[valid_index,:]
        y_train, y_valid = y.iloc[train_index,:], y.iloc[valid_index,:]

        modelx = lgb.LGBMRegressor(**lgb_params)
        modelx.fit(X_train, y_train["x"], eval_set=[(X_valid, y_valid["x"])], eval_metric='rmse', verbose=False, early_stopping_rounds=20)
        predx = modelx.predict(X_valid)
    
        modely = lgb.LGBMRegressor(**lgb_params)
        modely.fit(X_train, y_train["y"], eval_set=[(X_valid, y_valid["y"])], eval_metric='rmse', verbose=False, early_stopping_rounds=20)
        predy = modely.predict(X_valid)

        modelf = lgb.LGBMClassifier(**lgb_f_params)
        modelf.fit(X_train, y_train["f"], eval_set=[(X_valid, y_valid["f"])], eval_metric='multi_logloss', verbose=False, early_stopping_rounds=20)
        predf = modelf.predict(X_valid)

        score = comp_metric(predx, predy, predf, y_valid["x"], y_valid["y"], y_valid["f"])
        print(f"site={e+1}/{len(train_files)}:{str(file)}, fold={fold+1}/{N_SPLITS}, score={score}, elapsed_time={elapsed_timer_cv.get()}")

        # y_oof[y_valid.index,0] = predx
        # y_oof[y_valid.index,1] = predy
        # y_oof[y_valid.index,2] = predf

        y_oof[valid_index,0] = predx
        y_oof[valid_index,1] = predy
        y_oof[valid_index,2] = predf

        y_test[fold,:,0] = modelx.predict(test_data.iloc[:,:-1])
        y_test[fold,:,1] = modely.predict(test_data.iloc[:,:-1])
        y_test[fold,:,2] = modelf.predict(test_data.iloc[:,:-1])    
    

    score = comp_metric(y_oof[:,0], y_oof[:,1], y_oof[:,2], y["x"], y["y"], y["f"])

    y_oof_all = np.concatenate([y_oof_all, y_oof])
    y_truth_all = np.concatenate([y_truth_all, train_data[["x", "y", "f"]].values])

    print(f"site={e+1}/{len(train_files)}:{str(file)}, score={score}, elapsed_time={elapsed_timer.get()}")

    y_result[:,1:3] = np.mean(y_test[:,:,0:2], axis=0)
    y_result[:,0] = stats.mode(y_test[:,:,2], axis=0)[0].astype(np.int32).reshape(-1)

    # y_result = np.mean(y_test, axis=0)
    # y_result = y_result[:,[2,0,1]]

    test_preds = pd.DataFrame(y_result)
    test_preds.columns = ssubm.columns
    test_preds.index = test_data["site_path_timestamp"]
    test_preds["floor"] = test_preds["floor"].astype(int)
    predictions.append(test_preds)

score = comp_metric(
    y_oof_all[:,0], y_oof_all[:,1], y_oof_all[:,2], 
    y_truth_all[:,0], y_truth_all[:,1], y_truth_all[:,2],
)
print(f"site=all, score={score}")

# generate prediction file 
all_preds = pd.concat(predictions)
all_preds = all_preds.reindex(ssubm.index)
all_preds.to_csv('submission.csv')
