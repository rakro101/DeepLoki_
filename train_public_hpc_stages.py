from parameter_holder_hpc import*
import time
__author__ = "Raphael Kronberg Department of MMBS, MatNat Faculty," \
             " Heinrich-Heine-University"
__license__ = "MIT"
__version__ = "1.0.1"
__status__ = "Prototype: This progam/code can not be used as diagnostic tool."
__credits__ = "Pls cite and refer to when using the code: Kronberg R.M.," \
              "Applications of Supervised Deep (Transfer) Learning for Medical Image Classification"

if __name__ == '__main__':
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
    arg_dict = create_arg_dict(reload=False,
                               add_img=False,
                               data_dir='data/data_set_003',#'./data/lowhangingfruits',
                               file_path_train='data/data_set_003', #'./data/lowhangingfruits',
                               result_file_name='data_set_003',
                               model_id='loki_data_set_003',
                               model_name="resnet152",
                               tile_size=224,
                               optimizer_name='SGD',
                               train_tile_classes = train_tile_classes,
                               class_names =class_names,
                               batch_size=256,
                               num_epochs=20,
                               learning_rate=0.001,
                               pixel_cutoff= 256,
                               early_stop= 15,
                               lr_step_size=1,
                               gamma=0.99,
                               num_train_layers= 3,
                               )
    args = get_arguments(arg_dict)
    T2 = Trainer(args)
    T2.model_train()
