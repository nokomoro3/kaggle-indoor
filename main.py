import json
import pathlib

import pandas as pd

from sample.io_f import read_data_file
from sample.visualize_f import visualize_trajectory

def main(input_path: pathlib.Path, output_path: pathlib.Path):

    output_path.mkdir(parents=True, exist_ok=True)

    sub_df = pd.read_csv(input_path.joinpath("sample_submission.csv"))
    sub_df["site"]      = [i.split("_")[0] for i in sub_df["site_path_timestamp"]]
    sub_df["path"]      = [i.split("_")[1] for i in sub_df["site_path_timestamp"]]
    sub_df["timestamp"] = [i.split("_")[2] for i in sub_df["site_path_timestamp"]]

    site_list_tt = sorted(sub_df.site.unique())
    with open(output_path.joinpath('site_list_tt.json'), 'wt') as fp:
        json.dump(site_list_tt, fp, indent=4)
    
    site_list_tr = [str(i).split("/")[-2]+"_"+str(i).split("/")[-1] for i in sorted([*input_path.joinpath('train').glob("*/*")])]
    with open(output_path.joinpath('site_list_tr.json'), 'wt') as fp:
        json.dump(site_list_tr, fp, indent=4)

    site_list_meta = [str(i).split("/")[-2]+"_"+str(i).split("/")[-1] for i in sorted([*input_path.joinpath('metadata').glob("*/*")])]
    with open(output_path.joinpath('site_list_meta.json'), 'wt') as fp:
        json.dump(site_list_meta, fp, indent=4)

    floor_category = sorted(list(set([str(i).split("/")[-1] for i in sorted([*input_path.joinpath('metadata').glob("*/*")])])))
    with open(output_path.joinpath('floor_category.json'), 'wt') as fp:
        json.dump(floor_category, fp, indent=4)

    path_file_list = [*input_path.joinpath('train').glob("**/*.txt")]
    path_file = path_file_list[0]

    #----------------------------
    # visualize sample path
    #----------------------------
    example = read_data_file(path_file)
    trajectory = example.waypoint
    # Removes timestamp (we only need the coordinates)
    trajectory = trajectory[:, 1:3]

    site = str(path_file).split("/")[-3]
    floor_number = str(path_file).split("/")[-2]
    path_name = str(path_file).split("/")[-1]
    floor_image = input_path.joinpath('metadata', f'{site}', f'{floor_number}', 'floor_image.png')
    floor_info = input_path.joinpath('metadata', f'{site}', f'{floor_number}', 'floor_info.json')
    with open(floor_info) as fp:
        json_data = json.load(fp)
    width_meter = json_data["map_info"]["width"]
    height_meter = json_data["map_info"]["height"]

    visualize_trajectory(
        trajectory = trajectory,
        floor_plan_filename = floor_image,
        width_meter = width_meter,
        height_meter = height_meter,
        title = f"{path_name}"
    )

    return

if __name__ == '__main__':
    main(pathlib.Path('./indata'), pathlib.Path('./outdata'))
