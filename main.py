import json
import pathlib
import re

import pandas as pd
import numpy as np

from sample.io_f import read_data_file
from sample.visualize_f import visualize_trajectory, visualize_heatmap, save_figure_to_html
from sample.main import calibrate_magnetic_wifi_ibeacon_to_position
from sample.main import extract_magnetic_strength, extract_wifi_rssi, extract_wifi_count, extract_ibeacon_rssi

def main(input_path: pathlib.Path, output_path: pathlib.Path):

    output_path.mkdir(parents=True, exist_ok=True)

    # testデータの情報をカラムからパース
    if input_path.joinpath("sample_submission.csv").exists():
        sub_df = pd.read_csv(input_path.joinpath("sample_submission.csv"))
        sub_df["site"]      = [i.split("_")[0] for i in sub_df["site_path_timestamp"]]
        sub_df["path"]      = [i.split("_")[1] for i in sub_df["site_path_timestamp"]]
        sub_df["timestamp"] = [i.split("_")[2] for i in sub_df["site_path_timestamp"]]

        # testデータのsiteリスト取得
        site_list_tt = sorted(sub_df.site.unique())
        with open(output_path.joinpath('site_list_tt.json'), 'wt') as fp:
            json.dump(site_list_tt, fp, indent=4)
    
    # trainデータのsiteリスト取得
    site_list_tr = [re.split('[\\\\/]', str(i))[-2] +"_"+ re.split('[\\\\/]', str(i))[-1] for i in sorted([*input_path.joinpath('train').glob("*/*")])]
    with open(output_path.joinpath('site_list_tr.json'), 'wt') as fp:
        json.dump(site_list_tr, fp, indent=4)

    # metadataのsiteリスト取得
    site_list_meta = [re.split('[\\\\/]', str(i))[-2] +"_"+ re.split('[\\\\/]', str(i))[-1] for i in sorted([*input_path.joinpath('metadata').glob("*/*")])]
    with open(output_path.joinpath('site_list_meta.json'), 'wt') as fp:
        json.dump(site_list_meta, fp, indent=4)

    # フロアのカテゴリをチェック
    floor_category = sorted(list(set([re.split('[\\\\/]', str(i))[-1] for i in sorted([*input_path.joinpath('metadata').glob("*/*")])])))
    with open(output_path.joinpath('floor_category.json'), 'wt') as fp:
        json.dump(floor_category, fp, indent=4)

    # 1サンプルの学習データ確認
    path_file_list = [*input_path.joinpath('train').glob("**/*.txt")]
    path_file = path_file_list[0]
    example = read_data_file(path_file)

    # サンプルの各種情報取得
    site = re.split('[\\\\/]', str(path_file))[-3]
    floor_number = re.split('[\\\\/]', str(path_file))[-2]
    path_name = re.split('[\\\\/]', str(path_file))[-1]
    floor_image = input_path.joinpath('metadata', f'{site}', f'{floor_number}', 'floor_image.png')
    floor_info = input_path.joinpath('metadata', f'{site}', f'{floor_number}', 'floor_info.json')
    with open(floor_info) as fp:
        json_data = json.load(fp)
    width_meter = json_data["map_info"]["width"]
    height_meter = json_data["map_info"]["height"]

    # 軌道の可視化
    trajectory = example.waypoint
    trajectory = trajectory[:, 1:3] # Removes timestamp (we only need the coordinates)

    fig = visualize_trajectory(
        trajectory = trajectory,
        floor_plan_filename = floor_image,
        width_meter = width_meter,
        height_meter = height_meter,
        title = f"{path_name}"
    )
    save_figure_to_html(fig, output_path.joinpath(f'{site}', f'{floor_number}', f'{path_name}', 'trajectory.html'))

    mwi_datas = calibrate_magnetic_wifi_ibeacon_to_position(
        [str(f) for f in path_file_list if str(pathlib.Path("").joinpath(site,floor_number)) in str(f)] # path of a floor wheel data
        # [path_file] # single path data
    )

    # Extracting the magnetic strength
    magnetic_strength = extract_magnetic_strength(mwi_datas)
    
    heat_positions = np.array(list(magnetic_strength.keys()))
    heat_values = np.array(list(magnetic_strength.values()))

    # Visualize the heatmap
    fig = visualize_heatmap(
        heat_positions, 
        heat_values, 
        floor_image,
        width_meter, 
        height_meter, 
        colorbar_title='mu tesla', 
        title='Magnetic Strength',
    )
    save_figure_to_html(fig, output_path.joinpath(f'{site}', f'{floor_number}', 'magneticStrength.html'))

    # Get WiFi data
    wifi_rssi = extract_wifi_rssi(mwi_datas)
    print(f'This floor has {len(wifi_rssi.keys())} wifi aps (access points).')

    wifi_counts = extract_wifi_count(mwi_datas)
    heat_positions = np.array(list(wifi_counts.keys()))
    heat_values = np.array(list(wifi_counts.values()))
    # filter out positions that no wifi detected
    mask = heat_values != 0
    heat_positions = heat_positions[mask]
    heat_values = heat_values[mask]

    # The heatmap
    fig = visualize_heatmap(
        heat_positions, 
        heat_values, 
        floor_image, 
        width_meter, 
        height_meter, 
        colorbar_title='count', 
        title=f'WiFi Count',
    )
    save_figure_to_html(fig, output_path.joinpath(f'{site}', f'{floor_number}', 'wifiCount.html'))

    # Getting the iBeacon data
    ibeacon_rssi = extract_ibeacon_rssi(mwi_datas)
    print(f'This floor has {len(ibeacon_rssi.keys())} ibeacons.')

    ibeacon_ummids = list(ibeacon_rssi.keys())
    target_ibeacon = ibeacon_ummids[0]
    heat_positions = np.array(list(ibeacon_rssi[target_ibeacon].keys()))
    heat_values = np.array(list(ibeacon_rssi[target_ibeacon].values()))[:, 0]

    # The heatmap
    fig = visualize_heatmap(
        heat_positions, 
        heat_values, 
        floor_image, 
        width_meter, 
        height_meter, 
        colorbar_title='dBm', 
        title='iBeacon RSSE',
    )
    save_figure_to_html(fig, output_path.joinpath(f'{site}', f'{floor_number}', 'ibeaconRsse.html'))

    return

if __name__ == '__main__':
    # main(pathlib.Path('./indata_mini'), pathlib.Path('./outdata_mini'))
    main(pathlib.Path('./indata'), pathlib.Path('./outdata'))
