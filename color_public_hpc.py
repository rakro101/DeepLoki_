from parameter_holder_hpc import *
__author__ = "Raphael Kronberg Department of MMBS, MatNat Faculty," \
             " Heinrich-Heine-University"
__license__ = "MIT"
__version__ = "1.0.1"
__status__ = "Prototype: This progam/code can not be used as diagnostic tool."
__credits__ = "Pls cite and refer to when using the code: Kronberg R. et al.," \
                "Communicator-driven Data Preprocessing Improves Deep Transfer Learning" \
                "of Histopathological Prediction of Pancreatic Ductal Adenocarcinoma.  , Journal, 2021"
if __name__ == '__main__':
    ttc = ['HLN', 'HP', 'PDAC']
    ttc2 = ['HLN', 'HP', 'PDAC', 'NL']
    pr = 'saved_models/train_model_resnet_Spots_test2022_03_16__10_28_45.pth'
    four_score = []
    arg_dict = create_arg_dict(reload=True, batch_size=1,
                               data_load_shuffle=False,
                               model_name='resnet',
                               session_id='resnet_test_Coms10_{}'.format('final___'), normalize_on=1,#1
                               label_coloring=True,#True
                               Not_TTC='None',
                               treshhold=0.0,
                               reload_path=pr,
                               folder_path= './data/Spots_test/test/HLN', # r'D:\patho_daten_alt\data\ex_val3',
                               train_tile_classes = ttc,
                               class_names = ttc2,
                               save_infer_img=True,
                               tile_size=224,
                               optimizer_name='ADAM',
                               result_file_name = 'Resnet_test',
                               save_colored_dir='./results/colored_img/',

    )

    args = get_arguments(arg_dict)
    T1 = Trainer(args)
    four_score.append(T1.inference_folder_new())

