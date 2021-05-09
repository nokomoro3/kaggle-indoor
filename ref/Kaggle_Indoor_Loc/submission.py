import pandas as pd
import torch
import os
import neptune
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model.lstm import OgLSTM, SeqLSTM, CustomLSTM
from model.model_comp import IndoorLocModel
from dataset.dataset import IndoorDataModule
from config import Config
from icecream import ic
from datetime import datetime
import numpy as np
import scipy
import multiprocessing
from tqdm import tqdm

from sample.io_f import read_data_file
from sample.compute_f import compute_rel_positions

def correct_path(args):
    path, path_df = args
    
    T_ref  = path_df['timestamp'].values
    xy_hat = path_df[['x', 'y']].values
    
    example = read_data_file(f'../../indata/test/{path}.txt')
    rel_positions = compute_rel_positions(example.acce, example.ahrs)
    if T_ref[-1] > rel_positions[-1, 0]:
        rel_positions = [np.array([[0, 0, 0]]), rel_positions, np.array([[T_ref[-1], 0, 0]])]
    else:
        rel_positions = [np.array([[0, 0, 0]]), rel_positions]
    rel_positions = np.concatenate(rel_positions)
    
    T_rel = rel_positions[:, 0]
    delta_xy_hat = np.diff(scipy.interpolate.interp1d(T_rel, np.cumsum(rel_positions[:, 1:3], axis=0), axis=0)(T_ref), axis=0)

    N = xy_hat.shape[0]
    delta_t = np.diff(T_ref)
    alpha = (8.1)**(-2) * np.ones(N)
    beta  = (0.3 + 0.3 * 1e-3 * delta_t)**(-2)
    A = scipy.sparse.spdiags(alpha, [0], N, N)
    B = scipy.sparse.spdiags( beta, [0], N-1, N-1)
    D = scipy.sparse.spdiags(np.stack([-np.ones(N), np.ones(N)]), [0, 1], N-1, N)

    Q = A + (D.T @ B @ D)
    c = (A @ xy_hat) + (D.T @ (B @ delta_xy_hat))
    xy_star = scipy.sparse.linalg.spsolve(Q, c)

    return pd.DataFrame({
        'site_path_timestamp' : path_df['site_path_timestamp'],
        'floor' : path_df['floor'],
        'x' : xy_star[:, 0],
        'y' : xy_star[:, 1],
    })

def main():
    train_data_dir = os.path.join(Config.DATA_DIR, 'train_all.pkl')
    test_data_dir = os.path.join(Config.DATA_DIR, 'test_all.pkl')
    submit_dir = os.path.join(Config.DATA_DIR, 'sample_submission.csv')
    
    train_data = pd.read_pickle(train_data_dir)
    test_data = pd.read_pickle(test_data_dir)
    submit = pd.read_csv(submit_dir)
    
    idm = IndoorDataModule(train_data, test_data, kfold=False)
    idm.prepare_data()
    idm.setup(stage="test")
    ic(idm.wifi_bssids_size)
    ic(idm.site_id_dim)
    
    model_path_0 = os.path.join(Config.SAVE_DIR, '0/epoch=35-val_loss=6.36-val_metric=6.36.pth.ckpt')
    model_path_1 = os.path.join(Config.SAVE_DIR, '1/epoch=51-val_loss=6.59-val_metric=6.59.pth.ckpt')
    model_path_2 = os.path.join(Config.SAVE_DIR, '2/epoch=36-val_loss=6.40-val_metric=6.40.pth.ckpt')
    model_path_3 = os.path.join(Config.SAVE_DIR, '3/epoch=48-val_loss=6.60-val_metric=6.60.pth.ckpt')
    model_path_4 = os.path.join(Config.SAVE_DIR, '4/epoch=41-val_loss=6.46-val_metric=6.46.pth.ckpt')
    
    
    model0 = IndoorLocModel.load_from_checkpoint(model_path_0, model=SeqLSTM(
            Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))
    model0.eval()
    model1 = IndoorLocModel.load_from_checkpoint(model_path_1, model=SeqLSTM(
            Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))
    model1.eval()
    model2 = IndoorLocModel.load_from_checkpoint(model_path_2, model=SeqLSTM(
            Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))
    model2.eval()
    model3 = IndoorLocModel.load_from_checkpoint(model_path_3, model=SeqLSTM(
            Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))
    model3.eval()
    model4 = IndoorLocModel.load_from_checkpoint(model_path_4, model=SeqLSTM(
            Config.num_wifi_feats, idm.wifi_bssids_size, idm.site_id_dim))
    model4.eval()
    
    
    for i, batch in enumerate(idm.test_dataloader()):
        batch_index = i * Config.val_batch_size
        
        # Make prediction
        output = torch.cat([model0(batch).unsqueeze(1), 
                            model1(batch).unsqueeze(1),
                            model2(batch).unsqueeze(1),
                            model3(batch).unsqueeze(1),], dim=1)
        output = torch.mean(output, 1, keepdim=True).squeeze()
        # output = model0(batch)
        x = output[:, 0].cpu().detach().numpy()
        y = output[:, 1].cpu().detach().numpy()
        f = output[:, 2].cpu().detach().numpy()
    
        submit.iloc[batch_index:batch_index+Config.val_batch_size, -3] = f
        submit.iloc[batch_index:batch_index+Config.val_batch_size, -2] = x
        submit.iloc[batch_index:batch_index+Config.val_batch_size, -1] = y
#     ic(submit)

    #-------------------------------------------------
    # overwrite floor model
    # ref. https://www.kaggle.com/saitodevel01/indoor-post-processing-by-cost-minimization/data
    #-------------------------------------------------
    submit = pd.read_csv("data/submission-99percent-accurate-floor-model.csv")
    submit['floor'] = b['floor']
    dt = datetime.now()
    date_string = dt.strftime("%Y-%m-%d-%H%M%S")
    submit.to_csv(f'data/submission-{date_string}-floor-model.csv', index=False)

    #-------------------------------------------------
    # post process
    # ref. https://www.kaggle.com/saitodevel01/indoor-post-processing-by-cost-minimization/data
    #-------------------------------------------------
    tmp = submit['site_path_timestamp'].apply(lambda s : pd.Series(s.split('_')))
    submit['site'] = tmp[0]
    submit['path'] = tmp[1]
    submit['timestamp'] = tmp[2].astype(float)

    processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=processes) as pool:
        dfs = pool.imap_unordered(correct_path, submit.groupby('path'))
        dfs = tqdm(dfs)
        dfs = list(dfs)
    submit = pd.concat(dfs).sort_values('site_path_timestamp')
    dt = datetime.now()
    date_string = dt.strftime("%Y-%m-%d-%H%M%S")
    submit.to_csv(f'data/submission-{date_string}-post-process.csv', index=False)

if __name__ == '__main__':
    main()