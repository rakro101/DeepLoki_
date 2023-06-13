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
    df = df.sort_values(["date","time","ms"]).copy()
    df["date_time"] =df["date"].astype('str')+df["time"].astype('str')
    return df

def remove_same_object_img(df:pd.DataFrame, treshold:int=50)->Tuple[pd.DataFrame,pd.DataFrame]:
    """
    Select images, that are same date and time, and difference between the ms is smaller then the treshold and a lower y-pos value then the original (first image).
    Args:
        df:
        treshold:

    Returns:

    """
    drop_index_list = []
    df2 = df.loc[df.duplicated(subset=["date","time"], keep=False)]
    for row in df2.iterrows():
        df2_diff =df2[df2['date_time']==row[1]['date_time']].copy()
        print(df2_diff)
        df2_diff["ms"]=df2_diff["ms"].astype('int')
        df2_diff["y-coord"]=df2_diff["y-coord"].astype('int')
        df2_diff["diff"]=df2_diff["ms"].diff()
        df2_diff["diff_ycoord"]=df2_diff["y-coord"].diff()
        df2_diff=df2_diff.dropna()
        print(df2_diff)
        #ToDo Ask if there is only movement in one direction
        if (df2_diff["diff"].values[0] <= treshold):# and (df2_diff["diff_ycoord"].values[0] != 0):
            #print(df2_diff["diff"].values)
            drop_index_list.append(df2_diff.index[0])
    drop_index_list  = list(set(drop_index_list))
    df_clean = df.drop(drop_index_list)
    print(drop_index_list)
    df_drop_zooms = df.loc[drop_index_list]
    df_drop_zooms = df_drop_zooms.drop_duplicates()
    print(f"Droped images: {len(df_drop_zooms)}")
    return df_clean, df_drop_zooms

def copy_no_double_img(df:pd.DataFrame, target_folder_path:str)->None:
    """

    Args:
        df:
        target_folder_path:

    Returns:

    """
    print(len(df))
    for row in df.iterrows():
        root = row[1]["path"]
        name = row[1]["filename"]
        src_file = os.path.join(root, name)
        temp_path = os.path.join(target_folder_path, name)
        print(src_file)
        #print(temp_path)
        #shutil.copy(src_file, temp_path)
    return None


from typing import List
import pandas as pd
import numpy as np


class ImageNameValidator:
    @staticmethod
    def is_valid_image_name(image_name: str) -> bool:
        # Add your validation logic here
        pass


def detect_duplicate_specimens(df: pd.DataFrame, gamma: int, eta: float) -> List[List[str]]:
    """
    Detects multiple pictures of the same specimen based on the specified rules.

    Args:
        df: DataFrame containing image names and attributes.
        gamma: Threshold for the difference in milliseconds.
        eta: Threshold for the Euclidean distance between coordinates.

    Returns:
        A list of lists, where each sublist contains the image names of duplicate specimens.
    """
    duplicates = []
    for _, group in df.groupby(["date", "time"]):
        if len(group) > 1:
            for i in range(len(group)):
                specimen_duplicates = [group["filename"].iloc[i]]
                for j in range(i + 1, len(group)):
                    if abs(group["ms"].astype('int').iloc[i] - group["ms"].astype('int').iloc[j]) <= gamma:
                        coord_i = np.array([group["x-coord"].astype('int').iloc[i], group["y-coord"].astype('int').iloc[i]], dtype=float)
                        coord_j = np.array([group["x-coord"].astype('int').iloc[j], group["y-coord"].astype('int').iloc[j]], dtype=float)
                        distance = np.linalg.norm(coord_i - coord_j)
                        if distance <= eta:
                            specimen_duplicates.append(group["filename"].iloc[j])
                if len(specimen_duplicates) > 1:
                    duplicates.append(specimen_duplicates)
    return duplicates



if __name__ == '__main__':
    df = create_data_frame_form_folder("data/zoomie/export_6238_20230517_1747")
    print(len(detect_duplicate_specimens(df, 60, 280)))
    # df_clean, df_drop_zooms = remove_same_object_img(df=df, treshold=120)
    # print(df_drop_zooms.shape)
    # print(df_clean.shape)
    # copy_no_double_img(df=df_drop_zooms, target_folder_path="copy_test")