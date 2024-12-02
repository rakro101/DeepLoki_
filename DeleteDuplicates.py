import os
import pandas as pd
from typing import Tuple
import shutil

def create_data_frame_form_folder(folder_path:str)->pd.DataFrame:
    """
    Create a dataframe with all image names
    Args:
        folder_path:

    Returns:

    """
    df = pd.DataFrame()
    full_file_name =[]
    date_l = []
    time_l =[]
    ms_l =[]
    imgnr_l=[]
    y_coord_l =[]
    x_coord_l =[]
    for root, dirs, files in os.walk(folder_path):
        for name in files:
            if name.endswith('.png'):
                full_file_name.append(name)
                name_split = name.split(" ")
                date_l.append(name_split[0])
                time_l.append(name_split[1])
                ms_l.append(name_split[3])
                imgnr_l.append(name_split[5])
                y_coord_l.append(name_split[6])
                x_coord_l.append(name_split[7].split(".")[0])
    df["date"] =date_l
    df["time"]=time_l
    df["ms"]=ms_l
    df["imgnr"]=imgnr_l
    df["y-coord"]=y_coord_l
    df["x-coord"]=x_coord_l
    df["filename"] = full_file_name
    df["path"] = folder_path
    df["ms"] =df["ms"].astype('int')
    df['x-coord'] =df['x-coord'].astype('int')
    df['y-coord'] =df['y-coord'].astype('int')
    df = df.sort_values(["date","time","ms"]).copy()
    df["date_time"] =df["date"].astype('str')+df["time"].astype('str')
    print(df.shape)
    return df

def are_duplicates(img1, img2, theta, alpha, beta):
    if (img2['ms'] - img1['ms'] < theta and
            img2['x-coord'] - img1['x-coord'] < alpha and
            img2['y-coord'] - img1['y-coord'] < beta):
        return True
    return False

def process_duplicate_images(folder_path, theta, alpha, beta):
    df = create_data_frame_form_folder(folder_path)
    df_sort = df.sort_values(by=["date", "time", "ms", "imgnr"])

    grouped = df_sort.groupby(['date', 'time', 'ms'])

    duplicates = {}
    for _, group in grouped:
        filenames = list(group['filename'])
        updated_filenames = []
        for filename in filenames:
            img1 = group[group['filename'] == filename].iloc[0]
            duplicates[filename] = [filename]
            for _, img2 in group.iterrows():
                if img1['filename'] != img2['filename'] and are_duplicates(img1, img2, theta, alpha, beta):
                    duplicates[filename].append(img2['filename'])
                    duplicates[filename].sort()

    df_duplicates = pd.DataFrame({'filename': list(duplicates.keys()), 'duplicates': list(duplicates.values())})
    df_duplicates['dub'] = df_duplicates['duplicates'].apply(lambda x: len(x) - 1)

    keep_map = {}
    for _, row in df_duplicates.iterrows():
        duplicates_list = row['duplicates']
        filename = row['filename']
        if len(duplicates_list) == 1:
            keep_map[filename] = 1
        else:
            if filename != duplicates_list[0]:
                keep_map[filename] = 0
            else:
                keep_map[filename] = 1

    merged_df = pd.merge(df_sort, df_duplicates, on='filename', how='left')
    merged_df['keep'] = merged_df['filename'].map(keep_map).fillna(1)

    return merged_df




if __name__ == "__main__":
    print('Start Delete Duplicates')
    folder_path = "data/5_cruises/PS99.2"
    theta, alpha, beta = 60, 400, 400
    result_df = process_duplicate_images(folder_path, theta, alpha, beta)
    print(result_df)
    result_df.to_csv("output/delete_duplicates.csv", sep=";")