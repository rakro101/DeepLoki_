import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Helper functions
def is_na_then_empty(col):
    if type(col) != str:
        ret = " "
        return ret
    else:
        return col
def replace_blanks_with_us(col):
    if type(col) == str:
        return col.replace(' ', '_')
    return col

def replace_mult_us_with_us(col):
    if type(col) == str:
        return col.replace('__', '_').replace('__', '_')
    return col

def replace_end_us_with_none(col):
    if type(col) == str:
        if col[-1] =="_":
            return col[:-1]
    return col

def dict_app(col):
    try:
        ret = dict_babsi[col]
    except:
        ret = np.nan
    return ret

def create_all_data_df(data_root_path):
    all_df_list = []
    for root, dirs, files in os.walk(data_root_path):
        for dir in dirs:
            if dir.startswith('export') and not dir.startswith('export_5008'):
                dir = os.path.join(root, dir)
                for temp_root, temp_dirs, temp_files in os.walk(dir):
                    for name in temp_files:
                        if name.endswith('.tsv'):
                            path = os.path.join(temp_root, name)
                            df = pd.read_csv(path, sep='\t')
                            df['root_path'] = temp_root
                            all_df_list.append(df)
    all_df = pd.concat(all_df_list)
    return all_df

def create_validate_df(all_df):
    validated_df = all_df[all_df["object_annotation_status"]=="predicted"].copy()
    return validated_df

def create_label_overview(validated_df):
    validated_df['count'] = 1
    overview_labels = validated_df.groupby('object_annotation_category').agg({'count': 'sum'})
    overview_labels = overview_labels.sort_values('count', ascending=False)
    overview_labels['percent_of_all_img'] = overview_labels['count']/overview_labels.sum().values[0]*100
    overview_labels['cumsum count'] = overview_labels['percent_of_all_img'].cumsum()
    overview_labels['class_count'] =1
    overview_labels['class_count'] =overview_labels['class_count'].cumsum()
    overview_labels.to_csv('output/annotation_overview.csv',";")
    overview_labels_100 =overview_labels.iloc[:100]
    overview_labels_100.to_csv('output/combine_classes_for_training.csv', sep=";")
    return overview_labels, overview_labels_100

def create_agg_labels(path ="data/combine_classes_for_training_2.csv"):
    df_babsi = pd.read_csv(path, sep=";")
    df_babsi["Taxon II"] = df_babsi["Taxon II"].apply(is_na_then_empty)
    df_babsi["Taxon III/part of organism"] = df_babsi["Taxon III/part of organism"].apply(is_na_then_empty)
    df_babsi["B_label"] =df_babsi["Taxon I"]+"_"+df_babsi["Taxon II"]+"_"+df_babsi["Taxon III/part of organism"]
    df_babsi["B_label"] =df_babsi["B_label"].apply(replace_blanks_with_us)
    df_babsi["B_label"] =df_babsi["B_label"].apply(replace_mult_us_with_us)
    df_babsi["B_label"] =df_babsi["B_label"].apply(replace_end_us_with_none)
    df_dict = df_babsi[["object_annotation_category","B_label"]]
    dict_babsi = {}
    for row in df_dict.iterrows():
        dict_babsi[row[1]["object_annotation_category"]] =row[1]["B_label"]
    return dict_babsi

def create_validated_sub_data_set(validated_df,excluded):
    validated_df['label'] = validated_df["object_annotation_category"].apply(dict_app)
    validated_df['new_index']= [ i for i in range(0,validated_df.shape[0])]
    validated_df = validated_df.set_index('new_index')
    validated_df_drop = validated_df[validated_df['label'].isin(excluded)]
    validated_df_drop[['object_annotation_category','label']]
    validated_df_zero = validated_df.copy()
    print(validated_df_zero.shape)
    validated_df_zero= validated_df_zero.drop(validated_df_drop.index)
    print(validated_df_zero.shape)
    validated_df_zero.to_csv('data_csv/data_run_004.csv',sep =";")
    return validated_df_zero

def create_data_split(validated_df_zero):
    df_train, df_val_test = train_test_split(validated_df_zero, test_size=0.3)
    df_test, df_val = train_test_split(df_val_test, test_size=0.5)
    df_train['phase'] = 'train'
    df_val['phase'] = 'val'
    df_test['phase'] = 'test'
    print(df_train.shape)
    print(df_val.shape)
    print(df_test.shape)
    df_phase = pd.concat([df_train,df_val,df_test])
    print(df_phase.shape)
    return df_phase

def create_folder_from_path(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return None

def create_folders(tvt, list_of_folders, run_id= "data_set_001"):
    for t in tvt:
        for s in list_of_folders:
            temp_path = f"data/{run_id}/{t}/{s}"
            print(temp_path)
            create_folder_from_path(temp_path)
    return None

def build_tvt(df_phase, run_id= "data_set_001"):
    counter = 0
    for img in df_phase.iterrows():
        if counter <= df_phase.shape[0]:
            print(img[1]['root_path']+"/"+img[1]['img_file_name'], img[1]['phase'], img[1]['label'])
            counter +=1
            src_file = img[1]['root_path']+"/"+img[1]['img_file_name']
            temp_path = f"data/{run_id}/{img[1]['phase']}/{img[1]['label']}"
            print(src_file,temp_path)
            shutil.copy(src_file, temp_path)
    return None

if __name__ == '__main__':
    data_root_path = "data/ecoTaxa"
    all_df = create_all_data_df(data_root_path)
    all_df.to_csv('output/df_ecoTaxa_dataPS99_0230303.csv')
    validated_df = create_validate_df(all_df)
    overview_labels, overview_labels_100 = create_label_overview(validated_df)
    dict_babsi = create_agg_labels(path ="data/combine_classes_for_training_20230303.csv")
    excluded = ['Badfocus','I_have_no_idea,_leave_out', 'Unknown','Doubles(???)', np.nan]
    validated_df_zero = create_validated_sub_data_set(validated_df,excluded)
    df_phase = create_data_split(validated_df_zero)
    df_phase.to_csv('output/ecoTaxa_dataPS992_0230303.csv')
    list_of_folders = list(np.unique(validated_df_zero['label']))
    print(len(list_of_folders))
    #tvt = ['train', 'val', 'test']
    #create_folders(tvt, list_of_folders, run_id="data_set_005")
    #build_tvt(df_phase, run_id="data_set_005")


