# APPLIED MACHINE LEARNING SYSTEM II ELEC0135 

The code is for the assignment of Applied Machine Learning System II and is implemented by Dongyu Wang (SN:23104424). 

The code is an implementation the of Single Image Super-Resolution. A new model architecture is introduced to effectively 
extract long-range and short-range information within features via combining two kinds of attention-based modules. 
To make the model lightweight, the modules are connected in a decoupled fashion that significantly saves the 
model parameters. 
The project has three modes: "train", "test", and "visual". In the train mode, the programme trains the super-resolution model 
from scratch and saves it as a .pth file. In the test mode, the code tests the performance of the model on various 
datasets. In the visual mode, the programme illustrates the performance of the model by showing the generated SR images.

## Code Structure and Description
main.py is used to launch the code. ./Datasets is the folder containing the data for training and testing. ./runs stores the
log information during training. ./A contains the code dealing with the Single Image Super-resolution. 

In the folder A, launch.py contains all the code for launching the model training, evaluation and visualization. 
cfgs.py stores all the hyperparameters and config information.
./dataset includes the code related to retrieving and preprocessing the data for training/testing. ./models/model_arch.py
stores the code of the model architecture. ./models/metrics.py includes the loss functions and metrics.
## Initialization

Please run anaconda and create the environment with the following command:
```
cd path/to/AMLS2/
conda env create -f environment.yaml
```

## Dataset
Please download the DIV2K dataset via the following link: https://data.vision.ee.ethz.ch/cvl/DIV2K/. 
This project uses the x4 bicubic downscaling LR images (Track1) for super-resolution. The downloaded data should 
be placed at: 

```
./Datasets/your_dataset
```
Please download the other datasets (Set5, Set14, Urban100, B100) via the following link:
https://cv.snu.ac.kr/research/EDSR/benchmark.tar. Place the data in the Datasets folder too.

As a result, the Datasets folder should contain the data in the following format:
```
Datasets ---- DIV2K_train_HR
         |--- DIV2K_train_LR_bicubic
         |--- DIV2K_valid_HR
         |--- DIV2K_valid_LR_bicubic
         |--- benchmark
```
## Launch
Here we can start training of the model:
```
python main.py
```

To adjust hyperparameters, please use the following command:
```
python main.py --mode train --cmid 32 --cup 24 --lr 2e-3
```

To test the model, run the following command:
```
python main.py --mode test --cmid 32 --cup 24 --lr 2e-3 ----test_data B100 (or Urban100, Set5, Set14, DIV2K)
```

To see the log data during training, run the following command:
```
tensorboard --logdir /runs
```

To visualize the model performance, run the following command:
```
python main.py --mode visual --cmid 32 --cup 24 --lr 2e-3 ----test_data B100 (or Urban100, Set5, Set14, DIV2K)
```
