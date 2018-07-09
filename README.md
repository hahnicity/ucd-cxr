# UCD ChestX-ray14 effort

## Getting Started
If you are using lakota then you should setup your own anaconda environment with pytorch installed

    conda create -n pytorch python=3.6

After anaconda builds it you can activate it and install software using

    source activate pytorch
    conda install pytorch torchvision scipy numpy matplotlib pandas ipython
    cd ucd-cxr
    python setup.py develop

## Label Preprocessing
If you are using your own machine to run the data then you will need to preprocess the dataset before running it.
First, make sure that you've downloaded and unzipped all CXR images from [NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC).
Next download the files `train_val_list.txt`, `test_list.txt`, and `Data_Entry_2017.csv`. Now you can perform preprocessing.
The first set of preprocessing creates a new file that will have both the image names and the labels associated with them.

    cd cxrlib
    python label_preprocessing.py train_test_preproc /path/to/Data_Entry_2017.csv /path/to/train_val_list.txt
    python label_preprocessing.py train_test_preproc /path/to/Data_Entry_2017.csv /path/to/test_list.txt

After this is complete you will have a file that reflects both the image and the label associated with it. This can be used
in subsequent preprocessing or in PyTorch itself. If you want to create a validation set then you can do so via

    python label_preprocessing.py make_validation_set /path/to/train_val_list.processed /path/to/test_list.processed

This will help for more effective training of the classifier.


## Dataset Preprocessing
There are multiple transformations that the CPU needs to perform on an image before it
can be processed by pytorch. Although these transformations are usually fast they do take
some time, and when you are running tens of thousands of mini-batches for training they
do take time in aggregate. In order to preprocess all batch transforms before the training
process you can utilize `dataset_preprocessing.py`.

    cd cxrlib
    python dataset_preprocessing.py /path/to/input/images /path/to/output/dir --convert-to RGB --labels-path /path/to/img/labels/dir

In experience, training does run slightly faster after performing this operation.

## Heading Heading
stub
