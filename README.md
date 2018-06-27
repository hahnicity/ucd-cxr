# UCD ChestX-ray14 effort

## Getting Started
If you are using lakota then pytorch comes pre-built with anaconda. You can activate it yourself using

    source activate pytorch

Afterwards you can use all software normally. If you are not using lakota then you will have to setup your
pytorch libraries yourself. You can do this by using the pytorch site for instructions. Once you have finished
setting up pytorch run setup.py.

    python setup.py develop

After this all cxr common libraries should be good for use.

## Dataset Preprocessing
If you are using your own machine to run the data then you will need to preprocess the dataset before running it.
First, make sure that you've downloaded and unzipped all CXR images from [NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC).
Next download the files `train_val_list.txt`, `test_list.txt`, and `Data_Entry_2017.csv`. Now you can perform preprocessing.
The first set of preprocessing creates a new file that will have both the image names and the labels associated with them.

    cd cxrlib
    python label_preprocessing.py /path/to/Data_Entry_2017.csv /path/to/<test_list.txt OR train_val_list.txt>

After this is complete you will have a file that reflects both the image and the label associated with it. This can be used
in subsequent preprocessing or in PyTorch itself.
## Heading Heading
stub
