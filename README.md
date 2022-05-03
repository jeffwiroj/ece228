# ECE228 Project Code

Setup: Create a folder called data, and download pathmnist.npz from https://zenodo.org/record/6496656.

utils/: Contains training/validation function and model.

dataset.py: To load data in torch tensor.

experiments/:
- configs: Contains all the configurations/hyperparameter settings we ran
- results: Contains the training summary of all experiments
  - weights: Contains the weights for each experiment
- train.py: training file given a config .yaml file

