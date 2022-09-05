from parameter_holder_hpc import*
import time
__author__ = "Philipp Lang and Raphael Kronberg Department of Molecular Medicine II, Medical Faculty," \
             " Heinrich-Heine-University"
__license__ = "MIT"
__version__ = "1.0.1"
__status__ = "Prototype: This progam/code can not be used as diagnostic tool."
__credits__ = "Pls cite and refer to when using the code: Kronberg R. et al.," \
                "Communicator-driven Data Preprocessing Improves Deep Transfer Learning" \
                "of Histopathological Prediction of Pancreatic Ductal Adenocarcinoma.  , Journal, 2022"

if __name__ == '__main__':
    print('sleep')
    time.sleep(5)
    print('sleep')
    arg_dict = create_arg_dict(reload=False, add_img=False, data_dir='./data/lowhangingfruits',
                               file_path_train='./data/lowhangingfruits', normalize_on=0,
                               result_file_name='lowhangingfruits',
                               model_id='lowhangingfruits',
                               model_name="resnet",
                               tile_size=224,
                               optimizer_name='ADAM',
                               train_tile_classes=['Loki', 'Other'],
                               class_names=['Loki', 'Other', 'NL'],
                               batch_size=150,
                               num_epochs=2,
                               learning_rate=0.0001,
                               pixel_cutoff= 256,
                               early_stop= 5,
                               num_train_layers= 3,
                               )
    args = get_arguments(arg_dict)
    T2 = Trainer(args)
    T2.model_train()