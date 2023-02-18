import pandas as pd
from parameter_holder_hpc import*
import time
__author__ = "Raphael Kronberg Department of MMBS, MatNat Faculty," \
             " Heinrich-Heine-University"
__license__ = "MIT"
__version__ = "1.0.1"
__status__ = "Prototype: This progam/code can not be used as diagnostic tool."
__credits__ = "Pls cite and refer to when using the code: Kronberg R.M.," \
              "Applications of Supervised Deep (Transfer) Learning for Medical Image Classification"


def datestr():
    ''' get the date for naming of files '''
    temp_time = '{:04}_{:02}_{:02}__{:02}_{:02}_{:02}'.format(time.gmtime().tm_year, time.gmtime().tm_mon,
                                                              time.gmtime().tm_mday, time.gmtime().tm_hour,
                                                              time.gmtime().tm_min, time.gmtime().tm_sec)
    return temp_time

if __name__ == '__main__':
    date = datestr()
    train_tile_classes = ['Amphipoda',
                             'Antenna',
                             'Artefact',
                             'Bubble',
                             'Chaetognata',
                             'Chaetognata_Eukronia',
                             'Chaetognata_Eukronia-tail',
                             'Chaetognata_head',
                             'Chaetognata_tail',
                             'Cnidaria',
                             'Cnidaria_Siphonophorae',
                             'Copepoda',
                             'Copepoda_Calanoida',
                             'Copepoda_Calanoida_small',
                             'Copepoda_Calanus',
                             'Copepoda_Calanus_Calanus_hyperboreus',
                             'Copepoda_Gaetanus',
                             'Copepoda_Group_Microclanaus',
                             'Copepoda_Group_Oithona',
                             'Copepoda_Group_Oncaea',
                             'Copepoda_Group_Paraeuchaeta',
                             'Copepoda_Heterorhabdus',
                             'Copepoda_Metridia_longa',
                             'Copepoda_Metridia_longa_large_Cops',
                             'Copepoda_Microcalanus',
                             'Copepoda_Oithona',
                             'Copepoda_Pseudocalanus',
                             'Copepoda_Scaphocalanus',
                             'Copepoda_Scolecithricella',
                             'Copepoda_Spinocalanus',
                             'Copepoda_dead',
                             'Crustacea',
                             'Detritus',
                             'Eggs',
                             'Euphausiacea',
                             'Feces',
                             'Foraminifera',
                             'Nauplii',
                             'Ostracoda',
                             'Polychaeta',
                             'Rhizaria',
                             'Trochophora',
                             'multiples']

    class_names = ['Amphipoda',
                    'Antenna',
                             'Artefact',
                             'Bubble',
                             'Chaetognata',
                             'Chaetognata_Eukronia',
                             'Chaetognata_Eukronia-tail',
                             'Chaetognata_head',
                             'Chaetognata_tail',
                             'Cnidaria',
                             'Cnidaria_Siphonophorae',
                             'Copepoda',
                             'Copepoda_Calanoida',
                             'Copepoda_Calanoida_small',
                             'Copepoda_Calanus',
                             'Copepoda_Calanus_Calanus_hyperboreus',
                             'Copepoda_Gaetanus',
                             'Copepoda_Group_Microclanaus',
                             'Copepoda_Group_Oithona',
                             'Copepoda_Group_Oncaea',
                             'Copepoda_Group_Paraeuchaeta',
                             'Copepoda_Heterorhabdus',
                             'Copepoda_Metridia_longa',
                             'Copepoda_Metridia_longa_large_Cops',
                             'Copepoda_Microcalanus',
                             'Copepoda_Oithona',
                             'Copepoda_Pseudocalanus',
                             'Copepoda_Scaphocalanus',
                             'Copepoda_Scolecithricella',
                             'Copepoda_Spinocalanus',
                             'Copepoda_dead',
                             'Crustacea',
                             'Detritus',
                             'Eggs',
                             'Euphausiacea',
                             'Feces',
                             'Foraminifera',
                             'Nauplii',
                             'Ostracoda',
                             'Polychaeta',
                             'Rhizaria',
                             'Trochophora',
                             'multiples',
                             'NL']
    train_tile_classes = [
        "CVstage<Calanus",
        "female<Calanus",
        "CIVstage<Calanus",
        "CIIIstage<Calanus",
    ]

    class_names = [
        "CVstage<Calanus",
        "female<Calanus",
        "CIVstage<Calanus",
        "CIIIstage<Calanus",
        "NL"
    ]
    arg_dict_data_load = create_arg_dict(reload=False,
                               add_img=False,
                               data_dir='data/data_set_003',#'./data/lowhangingfruits',
                               file_path_train='data/data_set_003', #'./data/lowhangingfruits',
                               result_file_name='data_set_003',
                               model_id='loki_data_set_003',
                               model_name="resnet",
                               tile_size=224,
                               optimizer_name='SGD',
                               train_tile_classes = train_tile_classes,
                               class_names =class_names,
                               )
    args = get_arguments(arg_dict_data_load)
    T2 = Trainer(args)
    data_dict = T2.get_that_tiles()
    # init lists
    model_name_list =[]
    model_path_list = []
    model_arch_list = []
    model_opti_list = []
    model_bs_list = []
    model_lr_list = []
    model_nl_list =[]
    f1_score_list = []
    precision_score_list = []
    recall_score_list = []
    roc_auc_score_list = []
    balanced_accuracy_score_list = []
    jacc_list = []
    accuracy_score_list = []
    T3 = 0
    for model in ["effnet", "resnet152", "vitb"]:
        for opti in ["ADAM"]:
            for lrate in [0.0005, 0.0001, 0.001]:
                for nl in [9, 21]:
                    del T3
                    if model == "effnet" or model == "vitb":
                        bs = 64
                    else:
                        bs = 256
                    arg_dict_train = create_arg_dict(reload=False,
                                               add_img=False,
                                               data_dir='data/data_set_003',#'./data/lowhangingfruits',
                                               file_path_train='data/data_set_003', #'./data/lowhangingfruits',
                                               result_file_name='data_set_003',
                                               model_id='loki_data_set_003',
                                               model_name=model,
                                               tile_size=224,
                                               optimizer_name=opti,
                                               train_tile_classes = train_tile_classes,
                                               class_names =class_names,
                                               batch_size=bs,
                                               num_epochs=50,
                                               learning_rate=lrate,
                                               pixel_cutoff= 256,
                                               early_stop= 5,
                                               lr_step_size=5,
                                               gamma=0.95,
                                               num_train_layers=nl,
                                               )
                    args_train = get_arguments(arg_dict_train)
                    T3 = Trainer(args_train)
                    a, b = T3.model_train(data_dict)
                    model_name_list.append(model)
                    model_path_list.append(a)
                    model_arch_list.append(model)
                    model_opti_list.append(opti)
                    model_lr_list.append(lrate)
                    model_nl_list.append(nl)
                    model_bs_list.append(bs)
                    f1_score_list.append(b[0])
                    precision_score_list.append(b[1])
                    recall_score_list.append(b[2])
                    roc_auc_score_list.append(b[3])
                    balanced_accuracy_score_list.append(b[4])
                    jacc_list.append(b[5])
                    accuracy_score_list.append(b[6])
                    print(a,b)
    try:
        df = pd.DataFrame()
        df = df.assign(**{"model_name":model_name_list,
                     "path":model_path_list,
                     "arch":model_arch_list,
                     "optimizer":model_opti_list,
                     "learningrate":model_lr_list,
                     "tilesize": args_train.tile_size,
                     "batch_size":model_bs_list,
                     "lr_steps":args_train.lr_step_size,
                     "num_train_layers":model_nl_list,
                     "f1-score":f1_score_list,
                     "precission":precision_score_list,
                     "recall":recall_score_list,
                     "roc_auc":roc_auc_score_list,
                     "balanced_acc":balanced_accuracy_score_list,
                     "jaccard":jacc_list,
                     "accuracy":accuracy_score_list,
                     })

        df.to_csv(f"output/43_hyper_parameter_testing_grid_search_{date}.csv", sep=";")
        print(df.head())
    except Exception as err:
        print(err)