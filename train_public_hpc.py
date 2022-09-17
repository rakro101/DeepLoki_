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
    arg_dict = create_arg_dict(reload=False, add_img=False, data_dir='./data/loki',
                               file_path_train='./data/loki',
                               result_file_name='loki',
                               model_id='loki',
                               model_name="resnet",
                               tile_size=224,
                               optimizer_name='ADAM',
                               train_tile_classes=['92267', '92782', '13364', '92263', '83742', '92269', '82653', '85026', '92266', '93028', '92245', '92273', '92068', '17263', '92253', '92268', '92767', '85116', '81950', '92247', '92323', '342', '82502', '85078', '80135', '92762', '51958', '85060', '30815', '25828', '92277', '92274', '92270', '85079', '84963', '45074', '80126', '92275', '81965', '80134'],
                               class_names=['92267', '92782', '13364', '92263', '83742', '92269', '82653', '85026', '92266', '93028', '92245', '92273', '92068', '17263', '92253', '92268', '92767', '85116', '81950', '92247', '92323', '342', '82502', '85078', '80135', '92762', '51958', '85060', '30815', '25828', '92277', '92274', '92270', '85079', '84963', '45074', '80126', '92275', '81965', '80134', 'NL'],
                               batch_size=256,
                               num_epochs=100,
                               learning_rate=0.0001,
                               pixel_cutoff= 256,
                               early_stop= 15,
                               num_train_layers= 3,
                               )
    args = get_arguments(arg_dict)
    T2 = Trainer(args)
    T2.model_train()