from parameter_holder_hpc import *
__author__ = "Raphael Kronberg Department of MMBS, MatNat Faculty," \
             " Heinrich-Heine-University"
__license__ = "MIT"
__version__ = "1.0.1"
__status__ = "Prototype: This progam/code can not be used as diagnostic tool."
__credits__ = "Pls cite and refer to when using the code: Kronberg R.M.," \
                "Applications of Supervised Deep (Transfer) Learning for Medical Image Classification"


def main(folder_path, save_class_patches_path):
    ttc = ['Metridia_longa', 'Crustacea', 'Detritus', 'Calanus', 'Paraeuchaeta', 'Multiple', 'Calanoida',
                   'Copepoda', 'Oithona', 'Themisto', 'Bubble', 'Ostracoda', 'Microcalanus', 'Oncaea', 'Chaetognatha',
                   'Oithona_similis', 'Unknown', 'Eukrohnia_hamata', 'Pseudocalanus', 'Feces', 'Acantharia', 'Egg',
                   'Rhizaria']
    ttc2 = ['Metridia_longa', 'Crustacea', 'Detritus', 'Calanus', 'Paraeuchaeta', 'Multiple', 'Calanoida',
                   'Copepoda', 'Oithona', 'Themisto', 'Bubble', 'Ostracoda', 'Microcalanus', 'Oncaea', 'Chaetognatha',
                   'Oithona_similis', 'Unknown', 'Eukrohnia_hamata', 'Pseudocalanus', 'Feces', 'Acantharia', 'Egg',
                   'Rhizaria', 'NL']
    pr = 'saved_models/train_model_resnet50_loki_subclasses2022_09_21__17_30_10.pth'
    four_score = []
    arg_dict = create_arg_dict(reload=True,
                               batch_size=1,
                               data_load_shuffle=False,
                               model_name='resnet50',
                               session_id='resnet_test_Coms10_{}'.format('final___'),
                               label_coloring=True,#True
                               Not_TTC='None',
                               treshhold=0.00,
                               reload_path=pr,
                               folder_path= folder_path,
                               train_tile_classes = ttc,
                               class_names = ttc2,
                               save_infer_img=True,
                               save_class_patches=True,#asb
                               save_class_patches_path = save_class_patches_path,
                               tile_size=224,
                               optimizer_name='ADAM',
                               result_file_name = 'Analye_file',
                               save_colored_dir='./results/colored_img/',

    )

    args = get_arguments(arg_dict)
    T1 = Trainer(args)
    four_score.append(T1.inference_folder_new())
    print(four_score)
    return None

if __name__ == '__main__':
    main(folder_path='analyse/test_data/NL', save_class_patches_path='./sorted_img/')


