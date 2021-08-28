class Config():
    DATA_DIR = '..\\..\\outdata\\indoor-unified-wifids-beaconids-2021-05-11'
    SAVE_DIR = 'save'
    
    seed = 42
    epochs = 300
    num_wifi_feats = 50
    num_beacon_feats = 50
    fold_num = 5
    train_batch_size = 256
    val_batch_size = 256
    num_workers = 16
    device = 'gpu'
    neptune = False
    lr = 5e-3


