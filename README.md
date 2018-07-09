# UCD ChestX-ray14 effort

## Getting Started
If you are using lakota then you should setup your own anaconda environment with pytorch installed

    conda create -n pytorch python=3.6

After anaconda builds it you can activate it and install software using

    source activate pytorch
    conda install pytorch torchvision scipy numpy matplotlib pandas ipython
    cd ucd-cxr
    python setup.py develop

## Dataset Preprocessing
If you are using your own machine to run the data then you will need to preprocess the dataset before running it.
First, make sure that you've downloaded and unzipped all CXR images from [NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC).
Next download the files `train_val_list.txt`, `test_list.txt`, and `Data_Entry_2017.csv`. Now you can perform preprocessing.
The first set of preprocessing creates a new file that will have both the image names and the labels associated with them.

    cd cxrlib
    python label_preprocessing.py train_test_preproc /path/to/Data_Entry_2017.csv /path/to/train_val_list.txt
    python label_preprocessing.py train_test_preproc /path/to/Data_Entry_2017.csv /path/to/test_list.txt

After this is complete you will have a file that reflects both the image and the label associated with it. This can be used
in subsequent preprocessing or in PyTorch itself.
## Heading Heading
stub
