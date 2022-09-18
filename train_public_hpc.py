from parameter_holder_hpc import*
import time
__author__ = "Raphael Kronberg Department of MMBS, MatNat Faculty," \
             " Heinrich-Heine-University"
__license__ = "MIT"
__version__ = "1.0.1"
__status__ = "Prototype: This progam/code can not be used as diagnostic tool."
__credits__ = "Pls cite and refer to when using the code: Kronberg R. et al.," \
                "Communicator-driven Data Preprocessing Improves Deep Transfer Learning" \
                "of Histopathological Prediction of Pancreatic Ductal Adenocarcinoma.  , Journal, 2022"

if __name__ == '__main__':
    arg_dict = create_arg_dict(reload=False, add_img=False, data_dir='./data/loki_subclasses',
                               file_path_train='./data/loki_subclasses',
                               result_file_name='loki_subclasses',
                               model_id='loki_subclasses',
                               model_name="vgg19",
                               tile_size=224,
                               optimizer_name='ADAM',
                               train_tile_classes=['Metridia_longa', 'Crustacea', 'Detritus', 'Calanus', 'Paraeuchaeta', 'Multiple', 'Calanoida', 'Copepoda', 'Oithona', 'Themisto', 'Bubble', 'Ostracoda', 'Microcalanus', 'Oncaea', 'Chaetognatha', 'Oithona_similis', 'Unknown', 'Eukrohnia_hamata', 'Pseudocalanus', 'Feces', 'Acantharia', 'Egg', 'Rhizaria'],
                               class_names=['Metridia_longa', 'Crustacea', 'Detritus', 'Calanus', 'Paraeuchaeta', 'Multiple', 'Calanoida', 'Copepoda', 'Oithona', 'Themisto', 'Bubble', 'Ostracoda', 'Microcalanus', 'Oncaea', 'Chaetognatha', 'Oithona_similis', 'Unknown', 'Eukrohnia_hamata', 'Pseudocalanus', 'Feces', 'Acantharia', 'Egg', 'Rhizaria', 'NL'],
                               batch_size=256,
                               num_epochs=100,
                               learning_rate=0.01,
                               pixel_cutoff= 256,
                               early_stop= 100,
                               num_train_layers= 10,
                               )
    args = get_arguments(arg_dict)
    T2 = Trainer(args)
    T2.model_train()