
import random
import pathlib
from datetime import datetime
import glob
import os
import gc
import json
from collections import OrderedDict
import re

import pandas as pd
import numpy as np
import lightgbm as lgb
import scipy.stats as stats
from sklearn.model_selection import GroupKFold, KFold
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import pickle

from sample.io_f import read_data_file
from sample.compute_f import compute_step_positions
from utils.utils import time_function

class ElapsedTimer():
    def __init__(self) -> None:
        self.mem = datetime.now()
    def get(self) -> int:
        now = datetime.now()
        elapsed = now - self.mem
        self.mem = datetime.now()
        return elapsed

def extract_high_rssi_feature_wrapper(args):
    return extract_high_rssi_feature(*args)

@time_function
def extract_high_rssi_feature(input_file, output_dir, test_flag=False):

    ITEMS_TO_TAKE = 100

    file_name = input_file.name
    output_file = output_dir.joinpath(file_name)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    input_feats_df = pd.read_csv(input_file)

    num_of_lines = input_feats_df.shape[0]

    output_feats = []
    if test_flag == False: # train
        for i in range(num_of_lines):  
            tmp = input_feats_df.iloc[i,1:-5].astype(int).sort_values(ascending=False).head(ITEMS_TO_TAKE) # RSSI順に100個を抽出
            target = input_feats_df.iloc[i,-4:]
            line = [*tmp.index.astype(str), *tmp.values, *target] # 100個のbssid名、100個それぞれのrssi値、正解位置を結合
            output_feats.append(line)

    else: # test
        for i in range(num_of_lines):  
            tmp = input_feats_df.iloc[i,1:-1].astype(int).sort_values(ascending=False).head(ITEMS_TO_TAKE)
            target = input_feats_df.iloc[i, [-1]]
            line = [*tmp.index.astype(str), *tmp.values, *target]
            output_feats.append(line)

    output_feats_df = pd.DataFrame(output_feats)
    if test_flag == False: # train
        output_feats_df.columns = [f'bssid_{str(i)}' for i in range(ITEMS_TO_TAKE)] + [f'rssi_{str(i)}' for i in range(ITEMS_TO_TAKE)] + ['x','y','floor','path']
    else: # test
        output_feats_df.columns = [f'bssid_{str(i)}' for i in range(ITEMS_TO_TAKE)] + [f'rssi_{str(i)}' for i in range(ITEMS_TO_TAKE)] + ['site_path_timestamp']

    output_feats_df.to_csv(output_file)

def calc_wifi_beacon_feature_wrapper(args):
    return calc_wifi_beacon_feature(*args)

@time_function
def calc_wifi_beacon_feature(output_path, site, wifi_id_per_site, beacon_id_per_site, floor_map):

    floor_dir_list = sorted(output_path.joinpath('train', site).glob("*"))
    wifi_id = sorted(wifi_id_per_site[site])
    beacon_id = sorted(beacon_id_per_site[site])

    # count wifi lines (coutn only)
    wifi_lines_num = 0
    beacon_lines_num = 0
    for floor in floor_dir_list:
        floor_num = floor_map[floor.name]
        path_files = floor.glob("*.txt")
        for path_file in path_files:
            with open(path_file, encoding='utf-8') as f:
                lines = f.readlines()

            wifi_lines = [l.strip().split() for l in lines if 'TYPE_WIFI' in l]
            if len(wifi_lines)>0:
                wifi_df = pd.DataFrame(wifi_lines)
                wifi_lines_num = wifi_lines_num + len(wifi_df.groupby(0).count())

            beacon_lines = [l.strip().split() for l in lines if 'TYPE_BEACON' in l]
            if len(beacon_lines)>0:
                beacon_df = pd.DataFrame(beacon_lines)
                beacon_lines_num = beacon_lines_num + len(beacon_df.groupby(0).count())


    # create dataframe
    wifi_feats_np = np.zeros((wifi_lines_num, len(wifi_id) + 4)).astype(np.float32)
    wifi_lines_index = 0
    wifi_path_files_all = []
    beacon_feats_np = np.zeros((beacon_lines_num, len(beacon_id) + 4)).astype(np.float32)
    beacon_lines_index = 0
    beacon_path_files_all = []
    
    wifi_rssi_init = OrderedDict(zip(wifi_id, [-999]*len(wifi_id)))
    beacon_rssi_init = OrderedDict(zip(beacon_id, [-999]*len(beacon_id)))
    for floor in floor_dir_list:
        floor_num = floor_map[floor.name]
        path_files = floor.glob("*.txt")

        for path_file in path_files:

            path_datas = read_data_file(path_file)
            acce_datas = path_datas.acce       # TYPE_ACCELEROMETER
            # magn_datas = path_datas.magn       # TYPE_MAGNETIC_FIELD
            ahrs_datas = path_datas.ahrs       # TYPE_ROTATION_VECTOR
            # wifi_datas = path_datas.wifi       # TYPE_WIFI
            # ibeacon_datas = path_datas.ibeacon # TYPE_BEACON
            posi_datas = path_datas.waypoint   # TYPE_WAYPOINT

            # 加速度センサ、回転ベクトル、正解位置から1stepの位置を計算
            step_positions = compute_step_positions(acce_datas, ahrs_datas, posi_datas)
            with open(path_file, encoding='utf-8') as f:
                lines = f.readlines()

            wifi_lines = [l.strip().split() for l in lines if 'TYPE_WIFI' in l]
            beacon_lines = [l.strip().split() for l in lines if 'TYPE_BEACON' in l]
            waypoint_lines = [l.strip().split() for l in lines if 'TYPE_WAYPOINT' in l] 

            # waypoint_df = pd.DataFrame(waypoint_lines)
            step_position_df = pd.DataFrame(step_positions)

            if len(wifi_lines)>0:
                wifi_df = pd.DataFrame(wifi_lines)

                # generate a feature, and label for each wifi block
                for timestamp, group in wifi_df.groupby(0):
                    # nearest_wp_index = np.argmin(np.abs(int(timestamp) - waypoint_df[0].values.astype(np.int64))) # TODO: step_positionから探す方が良い気がする。
                    nearest_wp_index = np.argmin(np.abs(int(timestamp) - step_position_df[0].values.astype(np.int64))) # TODO: step_positionから探す方が良い気がする。

                    rssi = wifi_rssi_init.copy()
                    rssi.update(dict(zip(group.values[:,3], group.values[:,4])))
                    
                    wifi_feats_np[wifi_lines_index,:-4] = list(rssi.values())
                    wifi_feats_np[wifi_lines_index,-4] = float(timestamp)
                    # wifi_feats_np[wifi_lines_index,-3] = float(waypoint_df[2][nearest_wp_index])
                    # wifi_feats_np[wifi_lines_index,-2] = float(waypoint_df[3][nearest_wp_index])
                    wifi_feats_np[wifi_lines_index,-3] = float(step_position_df[1][nearest_wp_index])
                    wifi_feats_np[wifi_lines_index,-2] = float(step_position_df[2][nearest_wp_index])
                    wifi_feats_np[wifi_lines_index,-1] = floor_num
                    wifi_path_files_all.append(path_file.stem) # useful for crossvalidation


                    wifi_lines_index = wifi_lines_index + 1

            if len(beacon_lines)>0:
                beacon_df = pd.DataFrame(beacon_lines)

                # generate a feature, and label for each beacon block
                for timestamp, group in beacon_df.groupby(0):
                    # nearest_wp_index = np.argmin(np.abs(int(timestamp) - waypoint_df[0].values.astype(np.int64))) # TODO: step_positionから探す方が良い気がする。
                    nearest_wp_index = np.argmin(np.abs(int(timestamp) - step_position_df[0].values.astype(np.int64))) # TODO: step_positionから探す方が良い気がする。

                    ids = [f'{i[3]}_{i[4]}' for i in group.values]
                    rssi = beacon_rssi_init.copy()
                    rssi.update(dict(zip(ids, group.values[:,6])))
                    
                    beacon_feats_np[beacon_lines_index,:-4] = list(rssi.values())
                    beacon_feats_np[beacon_lines_index,-4] = float(timestamp)
                    # beacon_feats_np[beacon_lines_index,-3] = float(waypoint_df[2][nearest_wp_index])
                    # beacon_feats_np[beacon_lines_index,-2] = float(waypoint_df[3][nearest_wp_index])
                    beacon_feats_np[beacon_lines_index,-3] = float(step_position_df[1][nearest_wp_index])
                    beacon_feats_np[beacon_lines_index,-2] = float(step_position_df[2][nearest_wp_index])
                    beacon_feats_np[beacon_lines_index,-1] = floor_num
                    beacon_path_files_all.append(path_file.stem) # useful for crossvalidation

                    beacon_lines_index = beacon_lines_index + 1

    columns = wifi_id
    columns.extend(["timestamp", "x", "y", "f"])
    feature_df = pd.DataFrame(wifi_feats_np, columns=columns)
    feature_df["path"] = wifi_path_files_all
    output_file = output_path.joinpath("indoor-navigation-and-location-wifi-features-2021-05-09", f"{site}_train.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_file)

    columns = beacon_id
    columns.extend(["timestamp", "x", "y", "f"])
    feature_df = pd.DataFrame(beacon_feats_np, columns=columns)
    feature_df["path"] = beacon_path_files_all
    output_file = output_path.joinpath("indoor-navigation-and-location-beacon-features-2021-05-09", f"{site}_train.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_file)

    return

def main():
    base_path = pathlib.Path('indata')
    output_path = pathlib.Path('outdata')

    # pull out all the buildings actually used in the test set, given current method we don't need the other ones
    ssubm = pd.read_csv(base_path.joinpath('sample_submission.csv'))

    # only 24 of the total buildings are used in the test set, 
    # this allows us to greatly reduce the intial size of the dataset

    ssubm_df = ssubm["site_path_timestamp"].apply(lambda x: pd.Series(x.split("_")))
    used_sites = sorted(ssubm_df[0].value_counts().index.tolist())

    # dictionary used to map the floor codes to the values used in the submission file. 
    floor_map = {"B2":-2, "B1":-1, "F1":0, "F2": 1, "F3":2, "F4":3, "F5":4, "F6":5, "F7":6,"F8":7, "F9":8,
                "1F":0, "2F":1, "3F":2, "4F":3, "5F":4, "6F":5, "7F":6, "8F": 7, "9F":8}

    elapsed_timer = ElapsedTimer()

    # #------------------------------------------------
    # # fix end of line CRLF code missing
    # #------------------------------------------------
    # path_files = sorted(base_path.joinpath('train').glob("**/*.txt"))
    # for path_file in tqdm(path_files, desc="path files", ascii=True):
    #     with open(path_file, encoding='utf-8') as f:
    #         lines = f.readlines()

    #     olines = []
    #     for l in lines:
    #         if 'TYPE_BEACON' in l:
    #             olines.append(re.sub("(\d{13}\tTYPE_)", "\n\\1", l).lstrip())
    #         else:
    #             olines.append(l)

    #     out_file = pathlib.Path(str(path_file).replace(str(base_path), str(output_path)))
    #     out_file.parent.mkdir(parents=True, exist_ok=True)
    #     with open(out_file, 'wt', encoding='utf-8') as f:
    #         f.writelines(olines)

    # path_files = sorted(base_path.joinpath('test').glob("**/*.txt"))
    # for path_file in tqdm(path_files, desc="path files", ascii=True):
    #     with open(path_file, encoding='utf-8') as f:
    #         lines = f.readlines()

    #     olines = []
    #     for l in lines:
    #         if 'TYPE_BEACON' in l:
    #             olines.append(re.sub("(\d{13}\tTYPE_)", "\n\\1", l).lstrip())
    #         else:
    #             olines.append(l)

    #     out_file = pathlib.Path(str(path_file).replace(str(base_path), str(output_path)))
    #     out_file.parent.mkdir(parents=True, exist_ok=True)
    #     with open(out_file, 'wt', encoding='utf-8') as f:
    #         f.writelines(olines)

    # #------------------------------------------------
    # # extract unique ssids each sites
    # #------------------------------------------------
    # elapsed_timer.get()
    # wifi_id_per_site = dict()
    # beacon_id_per_site = dict()
    # for i, site in enumerate(used_sites):
    #     wifi_id_hashset = set()
    #     beacon_id_hashset = set()
    #     floor_dir_list = sorted(output_path.joinpath('train', site).glob("*"))
    #     for floor in floor_dir_list:
    #         floor_num = floor_map[floor.name]
    #         path_files = floor.glob("*.txt")
    #         for path_file in path_files:
    #             with open(path_file, encoding='utf-8') as f:
    #                 lines = f.readlines()
    #                 wifi_id = [l.strip().split()[3] for l in lines if 'TYPE_WIFI' in l]
    #                 beacon_id = [f"{l.strip().split()[3]}_{l.strip().split()[4]}" for l in lines if 'TYPE_BEACON' in l]
    #                 wifi_id_hashset |= set(wifi_id)
    #                 beacon_id_hashset |= set(beacon_id)
    #     wifi_id_per_site[site] = sorted(list(wifi_id_hashset))
    #     beacon_id_per_site[site] = sorted(list(beacon_id_hashset))
    #     print(f"site={i+1}/{len(used_sites)}, wifi_num={len(wifi_id_hashset)}, beacon_num={len(beacon_id_hashset)}, elapsed_time={elapsed_timer.get()}")

    # with open(output_path.joinpath("wifi_id_per_site.json"), "w") as f:
    #     json.dump(wifi_id_per_site, f)
    # with open(output_path.joinpath("beacon_id_per_site.json"), "w") as f:
    #     json.dump(beacon_id_per_site, f)

    with open(output_path.joinpath("wifi_id_per_site.json")) as f:
        wifi_id_per_site = json.load(f)
    with open(output_path.joinpath("beacon_id_per_site.json")) as f:
        beacon_id_per_site = json.load(f)

    # #-----------------------------------------------
    # # calc wifi and beacon feature (train)
    # #-----------------------------------------------
    # num_cores = multiprocessing.cpu_count()
    # with Pool(num_cores) as pool:
    #     pool.map(calc_wifi_beacon_feature_wrapper, [ [output_path, site, wifi_id_per_site, beacon_id_per_site, floor_map] for site in used_sites])

    # #-----------------------------------------------
    # # calc wifi and beacon feature (test)
    # #-----------------------------------------------
    # # Generate the features for the test set
    # ssubm_building_g = ssubm_df.groupby(0)

    # for i, (gid0, g0) in enumerate(ssubm_building_g): # loop of site
    #     wifi_index = sorted(wifi_id_per_site[g0.iloc[0,0]])
    #     beacon_index = sorted(beacon_id_per_site[g0.iloc[0,0]])
    #     wifi_feats = list()
    #     beacon_feats = list()
    #     for gid,g in g0.groupby(1): # loop of path file

    #         # get all wifi time locations, 
    #         with open(os.path.join(base_path, 'test\\' + g.iloc[0,1] + '.txt'), encoding='utf-8') as f:
    #             txt = f.readlines()

    #         wifi = list()
    #         beacon = list()

    #         for line in txt:
    #             line = line.strip().split()
    #             if line[1] == "TYPE_WIFI":
    #                 wifi.append(line)
    #             if line[1] == "TYPE_BEACON":
    #                 beacon.append(line)

    #         wifi_df = pd.DataFrame(wifi)
    #         beacon_df = pd.DataFrame(beacon)

    #         for timepoint in g.iloc[:,2].tolist(): # 

    #             # wifi
    #             if len(wifi_df) > 0:
    #                 wifi_points = pd.DataFrame(wifi_df.groupby(0).count().index.tolist())
    #                 deltas = (wifi_points.astype(int) - int(timepoint)).abs()
    #                 min_delta_idx = deltas.values.argmin()
    #                 wifi_block_timestamp = wifi_points.iloc[min_delta_idx].values[0]
    #                 wifi_block = wifi_df[wifi_df[0] == wifi_block_timestamp].drop_duplicates(subset=3)
    #                 wifi_feat = wifi_block.set_index(3)[4].reindex(wifi_index).fillna(-999)
    #             else:
    #                 wifi_feat = pd.DataFrame([-999]*len(wifi_index), index=wifi_index)
    #             wifi_feat['site_path_timestamp'] = g.iloc[0,0] + "_" + g.iloc[0,1] + "_" + timepoint
    #             wifi_feats.append(wifi_feat)

    #             # beacon
    #             if len(beacon_df) > 0:
    #                 beacon_points = pd.DataFrame(beacon_df.groupby(0).count().index.tolist())
    #                 deltas = (beacon_points.astype(int) - int(timepoint)).abs()
    #                 min_delta_idx = deltas.values.argmin()
    #                 beacon_block_timestamp = beacon_points.iloc[min_delta_idx].values[0]
    #                 beacon_block = beacon_df[beacon_df[0] == beacon_block_timestamp].drop_duplicates(subset=3)
    #                 beacon_feat = beacon_block.set_index(3)[6].reindex(beacon_index).fillna(-999)
    #             else:
    #                 beacon_feat = pd.DataFrame([-999]*len(beacon_index), index=beacon_index)
    #             beacon_feat['site_path_timestamp'] = g.iloc[0,0] + "_" + g.iloc[0,1] + "_" + timepoint
    #             beacon_feats.append(beacon_feat)

    #     feature_df = pd.concat(wifi_feats, axis=1).T
    #     output_file = output_path.joinpath("indoor-navigation-and-location-wifi-features-2021-05-09", f"{gid0}_test.csv")
    #     output_file.parent.mkdir(parents=True, exist_ok=True)
    #     feature_df.to_csv(output_file)

    #     feature_df = pd.concat(wifi_feats, axis=1).T
    #     output_file = output_path.joinpath("indoor-navigation-and-location-beacon-features-2021-05-09", f"{gid0}_test.csv")
    #     output_file.parent.mkdir(parents=True, exist_ok=True)
    #     feature_df.to_csv(output_file)

    #     print(f"site={i+1}/{len(ssubm_building_g)}, elapsed_time={elapsed_timer.get()}")

    #-----------------------------
    # extract unified wifi ids
    #-----------------------------
    num_cores = multiprocessing.cpu_count()

    feature_src_dir = output_path.joinpath("indoor-navigation-and-location-wifi-features-2021-05-09")
    train_files = sorted(feature_src_dir.glob('*_train.csv'))
    test_files = sorted(feature_src_dir.glob('*_test.csv'))
    feature_dst_dir = output_path.joinpath("indoor-unified-wifids-2021-05-09")

    with Pool(num_cores) as pool:
        pool.map(extract_high_rssi_feature_wrapper, [ [t, feature_dst_dir] for t in train_files])
    with Pool(num_cores) as pool:
        pool.map(extract_high_rssi_feature_wrapper, [ [t, feature_dst_dir, True] for t in test_files])

    # merge all csv
    for name in ["train", "test"]:
        files = sorted(feature_dst_dir.glob(f'*_{name}.csv'))
        merge_df = pd.DataFrame([])
        for f in files:
            df = pd.read_csv(f, index_col=0)
            site_id = f.stem.split("_")[0]
            df["site_id"] = site_id
            merge_df = pd.concat([merge_df, df])
        merge_df = merge_df.reset_index(drop=True)
        merge_df.to_csv(feature_dst_dir.joinpath(f"{name}_all.csv"))
        merge_df.to_pickle(feature_dst_dir.joinpath(f"{name}_all.pkl"))

    # #-----------------------------
    # # train lightGBM
    # #-----------------------------
    # N_SPLITS = 10
    # SEED = 42

    # def set_seed(seed=42):
    #     random.seed(seed)
    #     os.environ["PYTHONHASHSEED"] = str(seed)
    #     np.random.seed(seed)

    # # the metric used in this competition
    # def comp_metric(xhat, yhat, fhat, x, y, f):
    #     intermediate = np.sqrt(np.power(xhat - x,2) + np.power(yhat-y,2)) + 15 * np.abs(fhat-f)
    #     return intermediate.sum()/xhat.shape[0]

    # set_seed(SEED)

    # feature_dir = output_path.joinpath("indoor-navigation-and-location-wifi-features")

    # # get our train and test files
    # train_files = sorted(feature_dir.glob('*_train.csv'))
    # test_files = sorted(feature_dir.glob('*_test.csv'))
    # ssubm = pd.read_csv(base_path.joinpath('sample_submission.csv'), index_col=0)


    # lgb_params = {'objective': 'root_mean_squared_error',
    #               'boosting_type': 'gbdt',
    #               'n_estimators': 50000,
    #               'learning_rate': 0.1,
    #               'num_leaves': 90,
    #               'colsample_bytree': 0.4,
    #               'subsample': 0.6,
    #               'subsample_freq': 2,
    #               'bagging_seed': SEED,
    #               'reg_alpha': 8,
    #               'reg_lambda': 2,
    #               'random_state': SEED,
    #               'n_jobs': -1
    #               }

    # lgb_f_params = {'objective': 'multiclass',
    #                 'boosting_type': 'gbdt',
    #                 'n_estimators': 50000,
    #                 'learning_rate': 0.1,
    #                 'num_leaves': 90,
    #                 'colsample_bytree': 0.4,
    #                 'subsample': 0.6,
    #                 'subsample_freq': 2,
    #                 'bagging_seed': SEED,
    #                 'reg_alpha': 10,
    #                 'reg_lambda': 2,
    #                 'random_state': SEED,
    #                 'n_jobs': -1
    #                 }

    # predictions = list()

    # # loop of sites
    # y_oof_all = np.empty((0,3))
    # y_truth_all = np.empty((0,3))

    # elapsed_timer = ElapsedTimer()
    # for e, file in enumerate(train_files):

    #     train_data = pd.read_csv(file, index_col=0).reset_index(drop=True)
    #     test_data = pd.read_csv(test_files[e], index_col=0).reset_index(drop=True)

    #     X = train_data.iloc[:,:-4]
    #     y = train_data[["x", "y", "f"]]
    #     y_oof = np.zeros(y.values.shape)
    #     y_test = np.zeros((N_SPLITS, len(test_data), 3))
    #     y_result = np.zeros((len(test_data), 3))

    #     # cross validation and predict test data
    #     path_unique = train_data["path"].unique()
    #     kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        
    #     elapsed_timer_cv = ElapsedTimer()
    #     for fold, (train_index, valid_index) in enumerate(kf.split(X)):
    #     # for fold, (train_group_index, valid_group_index) in enumerate(kf.split(path_unique)):

    #         # train_groups, valid_groups = path_unique[train_group_index], path_unique[valid_group_index]
    #         # is_train = train_data["path"].isin(train_groups)
    #         # is_valid = train_data["path"].isin(valid_groups)

    #         # X_train, X_valid = X[is_train], X[is_valid]
    #         # y_train, y_valid = y[is_train], y[is_valid]

    #         X_train, X_valid = X.iloc[train_index,:], X.iloc[valid_index,:]
    #         y_train, y_valid = y.iloc[train_index,:], y.iloc[valid_index,:]

    #         modelx = lgb.LGBMRegressor(**lgb_params)
    #         modelx.fit(X_train, y_train["x"], eval_set=[(X_valid, y_valid["x"])], eval_metric='rmse', verbose=False, early_stopping_rounds=20)
    #         predx = modelx.predict(X_valid)
        
    #         modely = lgb.LGBMRegressor(**lgb_params)
    #         modely.fit(X_train, y_train["y"], eval_set=[(X_valid, y_valid["y"])], eval_metric='rmse', verbose=False, early_stopping_rounds=20)
    #         predy = modely.predict(X_valid)

    #         modelf = lgb.LGBMClassifier(**lgb_f_params)
    #         modelf.fit(X_train, y_train["f"], eval_set=[(X_valid, y_valid["f"])], eval_metric='multi_logloss', verbose=False, early_stopping_rounds=20)
    #         predf = modelf.predict(X_valid)

    #         score = comp_metric(predx, predy, predf, y_valid["x"], y_valid["y"], y_valid["f"])
    #         print(f"site={e+1}/{len(train_files)}:{str(file)}, fold={fold+1}/{N_SPLITS}, score={score}, elapsed_time={elapsed_timer_cv.get()}")

    #         # y_oof[y_valid.index,0] = predx
    #         # y_oof[y_valid.index,1] = predy
    #         # y_oof[y_valid.index,2] = predf

    #         y_oof[valid_index,0] = predx
    #         y_oof[valid_index,1] = predy
    #         y_oof[valid_index,2] = predf

    #         y_test[fold,:,0] = modelx.predict(test_data.iloc[:,:-1])
    #         y_test[fold,:,1] = modely.predict(test_data.iloc[:,:-1])
    #         y_test[fold,:,2] = modelf.predict(test_data.iloc[:,:-1])    
        

    #     score = comp_metric(y_oof[:,0], y_oof[:,1], y_oof[:,2], y["x"], y["y"], y["f"])

    #     y_oof_all = np.concatenate([y_oof_all, y_oof])
    #     y_truth_all = np.concatenate([y_truth_all, train_data[["x", "y", "f"]].values])

    #     print(f"site={e+1}/{len(train_files)}:{str(file)}, score={score}, elapsed_time={elapsed_timer.get()}")

    #     y_result[:,1:3] = np.mean(y_test[:,:,0:2], axis=0)
    #     y_result[:,0] = stats.mode(y_test[:,:,2], axis=0)[0].astype(np.int32).reshape(-1)

    #     # y_result = np.mean(y_test, axis=0)
    #     # y_result = y_result[:,[2,0,1]]

    #     test_preds = pd.DataFrame(y_result)
    #     test_preds.columns = ssubm.columns
    #     test_preds.index = test_data["site_path_timestamp"]
    #     test_preds["floor"] = test_preds["floor"].astype(int)
    #     predictions.append(test_preds)

    # score = comp_metric(
    #     y_oof_all[:,0], y_oof_all[:,1], y_oof_all[:,2], 
    #     y_truth_all[:,0], y_truth_all[:,1], y_truth_all[:,2],
    # )
    # print(f"site=all, score={score}")

    # # generate prediction file 
    # all_preds = pd.concat(predictions)
    # all_preds = all_preds.reindex(ssubm.index)
    # all_preds.to_csv(output_path.joinpath('submission.csv'))

if __name__ == '__main__':
    main()