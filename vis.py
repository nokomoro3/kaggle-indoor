import json
import pathlib
import re

import pandas as pd
import numpy as np

from sample.io_f import read_data_file
from sample.visualize_f import visualize_trajectory, visualize_heatmap, save_figure_to_html, save_figure_to_image
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

    site_dir_list = sorted([*input_path.joinpath('train').glob("*")])

    def vis_path_file(path_datas, mwi_datas):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=2, cols=2)
        fig.add_trace( go.Scatter(x=path_datas.acce[:,0], y=path_datas.acce[:,1], name="acce_x"), row=1, col=1)
        fig.add_trace( go.Scatter(x=path_datas.acce[:,0], y=path_datas.acce[:,2], name="acce_y"), row=1, col=1)
        fig.add_trace( go.Scatter(x=path_datas.acce[:,0], y=path_datas.acce[:,3], name="acce_z"), row=1, col=1)

        fig.add_trace( go.Scatter(x=path_datas.magn[:,0], y=path_datas.magn[:,1], name="magn_x"), row=2, col=1)
        fig.add_trace( go.Scatter(x=path_datas.magn[:,0], y=path_datas.magn[:,2], name="magn_y"), row=2, col=1)
        fig.add_trace( go.Scatter(x=path_datas.magn[:,0], y=path_datas.magn[:,3], name="magn_z"), row=2, col=1)

        fig.add_trace( go.Scatter(x=path_datas.ahrs[:,0], y=path_datas.ahrs[:,1], name="ahrs_x"), row=1, col=2)
        fig.add_trace( go.Scatter(x=path_datas.ahrs[:,0], y=path_datas.ahrs[:,2], name="ahrs_y"), row=1, col=2)
        fig.add_trace( go.Scatter(x=path_datas.ahrs[:,0], y=path_datas.ahrs[:,3], name="ahrs_z"), row=1, col=2)

        fig.add_trace( go.Scatter(x=path_datas.waypoint[:,0], y=path_datas.waypoint[:,1], name="waypoint_x"), row=2, col=2)
        fig.add_trace( go.Scatter(x=path_datas.waypoint[:,0], y=path_datas.waypoint[:,2], name="waypoint_y"), row=2, col=2)

        fig.show()
        return

    df = pd.DataFrame([], columns=["site", "floor", "path_name", "sensor_data_len", "waypoint_num", "sensor_num_per_waypoint"])
    for site_dir in site_dir_list:
        site = site_dir.name
        floor_dir_list = [*site_dir.glob("*")]
        for floor_dir in floor_dir_list:
            floor = floor_dir.name
            path_file_list = [*floor_dir.glob("*.txt")]
            for path_file in path_file_list:
                path_name = path_file.stem
                path_datas = read_data_file(path_file)
                mwi_datas = calibrate_magnetic_wifi_ibeacon_to_position(
                    [path_file]
                )
                # vis_path_file(path_datas, mwi_datas)

                floor_image = input_path.joinpath('metadata', f'{site}', f'{floor}', 'floor_image.png')
                floor_info = input_path.joinpath('metadata', f'{site}', f'{floor}', 'floor_info.json')

                with open(floor_info) as fp:
                    json_data = json.load(fp)
                width_meter = json_data["map_info"]["width"]
                height_meter = json_data["map_info"]["height"]
                step_positions = np.array(list(mwi_datas.keys()))
                fig = visualize_trajectory(step_positions, floor_image, width_meter, height_meter, mode='markers', title='Step Positions')
                save_figure_to_html(fig, output_path.joinpath(f'{site}', 'step_positions_all', f'{floor}.html'))
                save_figure_to_image(fig, output_path.joinpath(f'{site}', 'step_positions_all', f'{floor}.png'))


                # s = pd.Series(
                #     [site, floor, path_name, len(path_datas.acce), len(path_datas.waypoint), len(path_datas.acce)/len(path_datas.waypoint)], 
                #     index=["site", "floor", "path_name", "sensor_data_len", "waypoint_num", "sensor_num_per_waypoint"],
                # )
                # df = df.append([s], ignore_index=True)
                # df.to_csv(output_path.joinpath('path_summary.csv'))

                break
            break
        break

    # path_summary: pd.DataFrame = pd.read_csv(output_path.joinpath('path_summary.csv'), index_col=0)
    # path_summary2 = path_summary.groupby(["site", "floor"]).sum()
    # path_summary2["path_count"] = path_summary.groupby(["site", "floor"]).count()["path_name"]
    # path_summary2.to_csv(output_path.joinpath('path_summary2.csv'))
    return



    #------------------------------------------
    # ある建物についての画像出力
    #------------------------------------------
    for site_dir in [site_dir_list[0]]:
        site = site_dir.name

        floor_dir_list = [*site_dir.glob("*")]
        for floor_dir in floor_dir_list:
            floor = floor_dir.name

            floor_image = input_path.joinpath('metadata', f'{site}', f'{floor}', 'floor_image.png')
            floor_info = input_path.joinpath('metadata', f'{site}', f'{floor}', 'floor_info.json')

            with open(floor_info) as fp:
                json_data = json.load(fp)
            width_meter = json_data["map_info"]["width"]
            height_meter = json_data["map_info"]["height"]

            path_file_list = [*floor_dir.glob("*.txt")]

            # path of a floor wheel data
            mwi_datas = calibrate_magnetic_wifi_ibeacon_to_position(
                [str(f) for f in path_file_list if str(pathlib.Path("").joinpath(site, floor)) in str(f)]
            )

            step_positions = np.array(list(mwi_datas.keys()))
            fig = visualize_trajectory(step_positions, floor_image, width_meter, height_meter, mode='markers', title='Step Positions')
            # save_figure_to_html(fig, output_path.joinpath(f'{site}', 'step_positions_all', f'{floor}.html'))
            save_figure_to_image(fig, output_path.joinpath(f'{site}', 'step_positions_all', f'{floor}.png'))

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
            # save_figure_to_html(fig, output_path.joinpath(f'{site}', 'magnetic_strength', f'{floor}.html'))
            save_figure_to_image(fig, output_path.joinpath(f'{site}', 'magnetic_strength', f'{floor}.png'))

            # Wifi Count
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
            save_figure_to_image(fig, output_path.joinpath(f'{site}', 'wificount', f'{floor}.png'))

            # Get WiFi data
            wifi_rssi = extract_wifi_rssi(mwi_datas)
            print(f'This floor has {len(wifi_rssi.keys())} wifi aps (access points).')

            for bssid in list(wifi_rssi.keys()):
                heat_positions = np.array(list(wifi_rssi[bssid].keys()))
                heat_values = np.array(list(wifi_rssi[bssid].values()))[:, 0]

                # The heatmap
                fig = visualize_heatmap(
                    heat_positions, 
                    heat_values, 
                    floor_image, 
                    width_meter, 
                    height_meter, 
                    colorbar_title='dBm', 
                    title='WiFi RSSI',
                )
                save_figure_to_image(fig, output_path.joinpath(f'{site}', 'wifi_rssi', f'{bssid}', f'{floor}.png'))

            # Getting the iBeacon data
            ibeacon_rssi = extract_ibeacon_rssi(mwi_datas)
            print(f'This floor has {len(ibeacon_rssi.keys())} ibeacons.')

            for mmid in list(ibeacon_rssi.keys()):
                heat_positions = np.array(list(ibeacon_rssi[mmid].keys()))
                heat_values = np.array(list(ibeacon_rssi[mmid].values()))[:, 0]

                # The heatmap
                fig = visualize_heatmap(
                    heat_positions, 
                    heat_values, 
                    floor_image, 
                    width_meter, 
                    height_meter, 
                    colorbar_title='dBm', 
                    title='iBeacon RSSI',
                )
                save_figure_to_image(fig, output_path.joinpath(f'{site}', 'ibeacon_rssi', f'{mmid}', f'{floor}.png'))

    return

if __name__ == '__main__':
    # main(pathlib.Path('./indata_mini'), pathlib.Path('./outdata_mini'))
    main(pathlib.Path('./indata'), pathlib.Path('./outdata'))
