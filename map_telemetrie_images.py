import pandas as pd
import os
import numpy as np

# Log data
df_log = pd.read_csv("meta_data_encoding/log_mapping.csv")
df_log_d = df_log[["Index","Name"]]
df_log_d.set_index("Index", inplace=True)
log_dict =df_log_d.to_dict()["Name"]

# Telemetrie
df_telemetrie = pd.read_csv("meta_data_encoding/telemetrie_mapping.csv")
df_telemetrie = df_telemetrie[df_telemetrie["Index"]!="."]
df_telemetrie_d = df_telemetrie[["Index", "Name"]]
df_telemetrie_d.set_index("Index", inplace=True)
telemetrie_dict = df_telemetrie_d.to_dict()["Name"]




def get_tmd_files(root_folder, ends='.tmd'):
    tmd_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(ends):
                if file[:2] != "._": #ToDo check ._ files
                    tmd_files.append(os.path.join(dirpath, file))
    return tmd_files

def read_telemetrie(path):
    df_tel = pd.read_csv(path,sep=";", header=None, engine='python')
    headers = ['index', "value"]
    df_tel.columns = headers
    df_tel["index"] =df_tel["index"].apply(lambda x: str(x)).map(telemetrie_dict)
    df_tel.set_index("index", inplace=True)
    df_tel = df_tel.T
    df_tel["key"] = path.split("/")[-1].split(".")[0]
    df_tel["date"]  =df_tel["key"].apply( lambda x: x.split(" ")[0])
    df_tel["time"]  =df_tel["key"].apply( lambda x: x.split(" ")[1])
    return df_tel

def read_log(path):
    df_log = pd.read_csv(path,sep=";", header=None)
    headers = ['index', "value"]
    df_log.columns = headers
    df_log["index"] =df_log["index"].apply(lambda x: str(x)).map(log_dict)
    df_log.set_index("index", inplace=True)
    df_log = df_log.T
    df_log["key"] = path.split("/")[-1].split(".")[0]
    df_log["date"]  =df_log["key"].apply( lambda x: x.split(" ")[0])
    df_log["time"]  =df_log["key"].apply( lambda x: x.split(" ")[-1])
    return df_log

def build_telemetrie_df(root_folder, save_path="meta_data_encoding/all_telemetrie.csv"):
    tmd_files = get_tmd_files(root_folder, '.tmd')
    print(len(tmd_files))
    list_comp1 = [path for path in tmd_files if path.split("/")[-1][:2] != "._"]
    list_comp1 =np.unique(list_comp1)
    print(list_comp1)
    list_comp =[read_telemetrie(path) for path in list_comp1]
    print(list_comp)
    print(len(list_comp))
    # for l in list_comp:
    #     print(l)
    #     print(l.columns)
    #     print(l.head())
    counter = 0
    for i, df in enumerate(list_comp):
        if df.index.duplicated() or (df.shape != (1,25)):
            print(f"DataFrame {i} - Shape: {df.shape}, Index Duplicates: {df.index.has_duplicates}")
            print("#########################")
            print(df.columns)
            print(df.head())
        else:
            counter += 1

    filtered_df_list = [df for df in list_comp if df.shape == (1, 25)]
    print("Only df with shape 1,25 are kept.")
    print(len(filtered_df_list))
    df_tel_all = pd.concat(filtered_df_list)

    df_tel_all["GPS_LAT"] = df_tel_all["GPS_LAT"].str.replace(",", ".").astype(float)
    df_tel_all["GPS_LONG"] = df_tel_all["GPS_LONG"].str.replace(",", ".").astype(float)
    df_tel_all["LOKI_FRAME"] = df_tel_all["LOKI_FRAME"].str.replace(",", ".").astype(float)
    # List of columns to be converted
    columns_to_convert = ['PRESS', 'TEMP', 'OXY_CON', 'OXY_SAT',
                      'OXY_TEMP', 'COND_COND', 'COND_TEMP', 'COND_SALY',
                      'COND_DENS', 'COND_SSPEED', 'FLOUR_1','LOKI_PIC', 'HOUSE_STAT', 'HOUSE_T1', 'HOUSE_T2', 'HOUSE_VOLT']
    # Replace commas with dots and convert to float for specified columns
    df_tel_all[columns_to_convert] = df_tel_all[columns_to_convert].apply(lambda x: x.str.replace(",", ".").astype(float))

    df_tel_all["root_folder"] = root_folder
    df_tel_all.set_index("key", inplace=True)
    df_tel_all.to_csv(save_path,sep=";")
    print("Number of non matchting files: %s", counter)
    return df_tel_all

def build_log_df(root_folder, save_path="meta_data_encoding/all_log.csv"):
    tmd_files = get_tmd_files(root_folder, '.log')
    print(len(tmd_files))
    df_tel_all = pd.concat([read_log(file_name) for file_name in tmd_files])
    df_tel_all["root_folder"] = root_folder
    df_tel_all.set_index("key", inplace=True)
    df_tel_all.to_csv(save_path,sep=";")
    return df_tel_all

def build_image_df(root_folder, save_path="meta_data_encoding/all_images.csv"):
    file_paths = get_tmd_files(root_folder, ends='.bmp')
    df_files = pd.DataFrame(file_paths, columns=['file_path'])
    df_files['file_name'] = df_files['file_path'].apply(lambda x: os.path.basename(x))
    df_files["key"] = df_files['file_name'].apply(lambda x: x[:15])
    df_files["date"] = df_files["key"].apply(lambda x: x.split(" ")[0])
    df_files["time"] = df_files["key"].apply(lambda x: x.split(" ")[-1])
    df_files.to_csv(save_path, sep=";")
    return df_files

if __name__ == "__main__":
    # Path to the haul folder
    run_name= "haul_004_2024_09_04"
    root_folder = "data/0010_PS121-010-03/Haul 9"
    root_folder =  "/Volumes/T7 Shield/T7/LOKI2/0007_PS143.2_07-4/Haul 4" # "data/0010_PS121-010-03/Haul 9"
    # change the save path to telemetie
    df_all_tele = build_telemetrie_df(root_folder,save_path=f"meta_data_encoding/{run_name}_all_telemetrie.csv")
    # change the save path for all image 
    df_all_images = build_image_df(root_folder, save_path=f"meta_data_encoding/{run_name}_all_images.csv")
    # Merge the DataFrames on 'date' and 'time' columns
    df_joined_img = df_all_images.merge(df_all_tele, on=['date', 'time'], how='left', suffixes=('_img', '_tele'))
    # Save the joined data image and telemetrie
    df_joined_img.to_csv(f"meta_data_encoding/{run_name}_A_image_tele_data.csv", sep=";")
