# DeepLoki - Classification for Cleaning Images
Based on the Paper:  <br>
Data are not avaible. <br>
Implemented in Python using the PyTorch Framework<br>
We provide the code to enable you to to analyse/ classify Images in useful and trash  automatically and quickly.<br>
<br>
# Installation Guide
https://pytorch.org/get-started/locally/

See https://github.com/rakro101/deeploki
'''
brew install python@3.10
pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt
'''
# Usage
<br>
The script always expects the following folder structure: train, test, val  and then the classes.
<br>
train_public_hpc.py - Script for Training a CNN form the list.
<br>
color_public_hpc.py - Use a trained CNN to classify & color the image patches.
<br>
To start the app just run the start_app.py
<br>
# Error Handling:
Error -2 tempfile.tif: Cannot read TIFF header. conda install libtiff=4.1.0=h885aae3_4 -c conda-forge or  conda install -c anaconda libtiff<br>

# TBC / Todo
- Currently working on the Pytorch Lightning Implementation
- And the wandb integration

# Latest features (08/2022)
- train, coloring, automatic_cleaning, clean,

# Authors
Raphael Kronberg and Ellen Oldenburg

# Support 
If you **really** like this repository and find it useful, please consider (★) **starring** it, so that it can reach a broader audience of like-minded people. It would be highly appreciated !

# Contributing to DeepLoki
If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues. 

# License , citation and acknowledgements
Please advice the **LICENSE.md** file. For usage of third party libraries and repositories please advise the respective distributed terms. Please cite our paper, when using this code:

```
@phdthesis{kronbergapplications,
  title={Applications of Supervised Deep (Transfer) Learning for Medical Image Classification},
  author={Kronberg, Raphael Marvin}
}
```
## Acknowledgements
Based on the implementation of the previous paper:  https://doi.org/10.3390/v13040610 .<br>

# Disclaimer
This progam/code can not be used as diagnostic tool.

