from public_modul_hpc import *
import argparse
__author__ = "Raphael Kronberg Department of MMBS, MatNat Faculty," \
             " Heinrich-Heine-University"
__license__ = "MIT"
__version__ = "1.0.1"
__status__ = "Prototype: This progam/code can not be used as diagnostic tool."
__credits__ = "Pls cite and refer to when using the code: Kronberg R.M.," \
              "Applications of Supervised Deep (Transfer) Learning for Medical Image Classification"


def create_arg_dict(
                    data_dir='./data/raw_data/',
                    file_path_train='./data/raw_data/',
                    train_val_test=['train', 'val', 'test'],
                    train_tile_classes=['Loki', 'Other'],
                    class_names=['Loki', 'Other', 'NL'],
                    tile_size=224,
                    dont_save_tiles='yes',
                    model_name="resnet",
                    batch_size=150,
                    num_epochs=2,
                    num_train_layers=3,
                    feature_extract=True,
                    use_pretrained=True,
                    optimizer_name='ADAM',
                    criterion_name='CEL',
                    scheduler_name=None,
                    img_data_dict=None,
                    reload=False,
                    input_size=None,
                    save_path='./saved_models/',
                    folder_path='data/ex_test',
                    reload_path='./saved_models/train_model.pth',
                    result_file_name='base_line_model_new_BackGround__2',
                    model_id='base_line_model_new_BackGround__2',
                    learning_rate=0.0001,
                    pixel_cutoff = 256,
                    tissue_per = 0.3,
                    early_stop = 5,
                    gamma = 0.95,
                    lr_step_size= 3,
                    data_load_shuffle = True,
                    label_coloring = True,
                    save_colored_dir = './results/colored_img',
                    session_id = 'Master',
                    patho_cut_off = False,
                    add_img = False,
                    add_img_path = './data/raw_data/',
                    Not_TTC = 'None',
                    save_class_dir = './results/',
                    treshhold = 0.75,
                    save_class_patches=False,
                    save_class_patches_path='./patches_A/',
                    save_class_patches_mod='infer',
                    save_class_patches_class='Loki',
                    save_infer_img = True,
                    patches_mod = False,
                    ):
    #ToDo Hook for GUI
    arg_dict = {}
    arg_dict['data_dir'] = data_dir
    arg_dict['train_tile_path'] = file_path_train
    arg_dict['train_val_test'] = train_val_test
    arg_dict['train_tile_classes'] = train_tile_classes
    arg_dict['class_names'] = class_names
    arg_dict['tile_size'] = tile_size
    arg_dict['dont_save_tiles'] = dont_save_tiles
    arg_dict['model_name'] = model_name
    arg_dict['batch_size'] = batch_size
    arg_dict['num_epochs'] = num_epochs
    arg_dict['num_train_layers'] = num_train_layers
    arg_dict['feature_extract'] = feature_extract
    arg_dict['use_pretrained'] = use_pretrained
    arg_dict['optimizer_name'] = optimizer_name
    arg_dict['criterion_name'] = criterion_name
    arg_dict['scheduler_name'] = scheduler_name
    arg_dict['img_data_dict'] = img_data_dict
    arg_dict['reload'] = reload
    arg_dict['input_size'] = input_size
    arg_dict['reload_path'] = reload_path
    arg_dict['folder_path'] = folder_path
    arg_dict['save_path'] = save_path
    arg_dict['result_file_name'] = result_file_name
    arg_dict['model_id'] = model_id
    arg_dict['learning_rate'] = learning_rate
    arg_dict['pixel_cutoff'] = pixel_cutoff
    arg_dict['tissue_per'] = tissue_per
    arg_dict['early_stop'] = early_stop
    arg_dict['gamma'] = gamma #
    arg_dict['lr_step_size'] = lr_step_size
    arg_dict['data_load_shuffle'] = data_load_shuffle
    arg_dict['label_coloring'] = label_coloring
    arg_dict['save_colored_dir'] = save_colored_dir
    arg_dict['session_id'] = session_id
    arg_dict['patho_cut_off'] = patho_cut_off
    arg_dict['add_img'] = add_img
    arg_dict['add_img_path'] = add_img_path
    #Not_TTC
    arg_dict['Not_TTC'] = Not_TTC
    arg_dict['save_class_dir'] = save_class_dir
    arg_dict['treshhold'] = treshhold
    arg_dict['save_class_patches'] = save_class_patches
    arg_dict['save_class_patches_path'] = save_class_patches_path
    arg_dict['save_class_patches_mod'] = save_class_patches_mod
    arg_dict['save_class_patches_class'] = save_class_patches_class
    arg_dict['save_infer_img'] = save_infer_img
    arg_dict['patches_mod'] = patches_mod
    print(arg_dict)
    return arg_dict


def get_arguments(arg_dict):
    parser = argparse.ArgumentParser()
    for ent in arg_dict.keys():
        parser.add_argument('--{}'.format(ent), type=type(arg_dict[ent]), default=arg_dict[ent])
    args = parser.parse_args() # "" when running on JupyterHub
    print(args)
    return args

if __name__ == '__main__':
    task = 'Train'
    if task == 'Infer':
        arg_dict = create_arg_dict(reload=True)
        args = get_arguments(arg_dict)
        T1 = Trainer(args)
        T1.inference_folder()
    elif task == 'Train':
        arg_dict = create_arg_dict(reload=False)
        args = get_arguments(arg_dict)
        T2 = Trainer(args)
        T2.model_train()
    else:
        print('No valid task')