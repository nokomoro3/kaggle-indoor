import json
import pathlib

import pandas as pd

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

if __name__ == '__main__':
    main(pathlib.Path('./indata'), pathlib.Path('./outdata'))
