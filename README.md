# MA-SNN

***Exploiting memristive autapse and temporal distillation for training spiking neural networks***

This paper has been submitted to ***Knowledge-Based Systems***. 
## Figures
![/MA-SNN/autapse_negative.png](https://github.com/CHNtao/MA-SNN/blob/main/autapse_negative.png)
* Negative feedback
![/MA-SNN/autapse_positive.png](https://github.com/CHNtao/MA-SNN/blob/main/autapse_positive.png)
* Positive feedback


## Requirements
*  Python 3.9.7
*  Torch 1.10.1
*  Torchvision 0.11.2
*  Numpy 1.22.0


## Datasets
*  [CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html) 
*  [CIFAR100](http://www.cs.toronto.edu/~kriz/cifar.html)
*  [DVS-CIFAR10](https://figshare.com/s/d03a91081824536f12a8)
*  [N-Caltech101](https://www.garrickorchard.com/datasets/n-caltech101)

## The inference weight
* [CIFAR10 (96.10%)](https://drive.google.com/file/d/1mz6dbHzSvkA5-8Pj9JU2ur8BMW5EjnzI/view?usp=drive_link), you can download it from google drive.
## Usage
* Set the dataset path in the test.py
* Download the trained model and set its path in test.py
* Then run test.py to reproduce the result on CIFAR10 of 96.10%.
* If you want to retrain MA-SNN from scratch, try to run my_main.py.



