# DeepLoki - Classification for Cleaning Images
Based on the Paper:  <br>
Data are not avaible. <br>
Implemented in Python using the PyTorch Framework<br>
We provide the code to enable you to to analyse/ classify Images in useful and trash  automatically and quickly.<br>
<br>
# Installation Guide
See https://github.com/

# Usage
<br>
The script always expects the following folder structure: train, test, val  and then the classes.
See patches_A_resnet as example.
<br>
train_public_hpc.py - Script for Training a CNN form the list.
<br>
color_public_hpc.py - Use a trained CNN to classify & color the image patches.
<br>

# Error Handling:
Error -2 tempfile.tif: Cannot read TIFF header. conda install libtiff=4.1.0=h885aae3_4 -c conda-forge or  conda install -c anaconda libtiff<br>

# TBC / Todo
- Open Code for Hyperparametertuning

# Latest features (08/2022)
- train, coloring, automatic_cleaning, clean,

# Support 
If you **really** like this repository and find it useful, please consider (★) **starring** it, so that it can reach a broader audience of like-minded people. It would be highly appreciated !

# Contributing to Auto_cleaner
If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues. 

# License , citation and acknowledgements
Please advice the **LICENSE.md** file. For usage of third party libraries and repositories please advise the respective distributed terms. Please cite our paper, when using this code:

```
@Article{cancers14081964,
AUTHOR = {Kronberg, Raphael M. and Haeberle, Lena and Pfaus, Melanie and Xu, Haifeng C. and Krings, Karina S. and Schlensog, Martin and Rau, Tilman and Pandyra, Aleksandra A. and Lang, Karl S. and Esposito, Irene and Lang, Philipp A.},
TITLE = {Communicator-Driven Data Preprocessing Improves Deep Transfer Learning of Histopathological Prediction of Pancreatic Ductal Adenocarcinoma},
JOURNAL = {Cancers},
VOLUME = {14},
YEAR = {2022},
NUMBER = {8},
ARTICLE-NUMBER = {1964},
URL = {https://www.mdpi.com/2072-6694/14/8/1964},
ISSN = {2072-6694},
ABSTRACT = {Pancreatic cancer is a fatal malignancy with poor prognosis and limited treatment options. Early detection in primary and secondary locations is critical, but fraught with challenges. While digital pathology can assist with the classification of histopathological images, the training of such networks always relies on a ground truth, which is frequently compromised as tissue sections contain several types of tissue entities. Here we show that pancreatic cancer can be detected on hematoxylin and eosin (H&amp;E) sections by convolutional neural networks using deep transfer learning. To improve the ground truth, we describe a preprocessing data clean-up process using two communicators that were generated through existing and new datasets. Specifically, the communicators moved image tiles containing adipose tissue and background to a new data class. Hence, the original dataset exhibited improved labeling and, consequently, a higher ground truth accuracy. Deep transfer learning of a ResNet18 network resulted in a five-class accuracy of about 94% on test data images. The network was validated with independent tissue sections composed of healthy pancreatic tissue, pancreatic ductal adenocarcinoma, and pancreatic cancer lymph node metastases. The screening of different models and hyperparameter fine tuning were performed to optimize the performance with the independent tissue sections. Taken together, we introduce a step of data preprocessing via communicators as a means of improving the ground truth during deep transfer learning and hyperparameter tuning to identify pancreatic ductal adenocarcinoma primary tumors and metastases in histological tissue sections.},
DOI = {10.3390/cancers14081964}
}


```
## Acknowledgements
Based on the implementation of the previous paper:  https://doi.org/10.3390/v13040610 .<br>

# Disclaimer
This progam/code can not be used as diagnostic tool.

