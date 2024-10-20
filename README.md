# Downscaling Satellite Imagery for High Resolution Offshore Winds.

## Description
This project seeks to downscale gridded satellite-based offshore winds to an NDBC buoy.
Different methods of downscaling are compared including a convolutional neural network
using tensorflow/keras, a random forest regressor, bilinear interpolation, and using the
nearest grid point as a baseline. Results show the CNN outperforming the other models.

## Subdirectories

#### Figures
Contains all figures in the paper.

#### Python
Contains all python scripts and notebooks.


## Python Files

#### bilinear.py 
Stores function to perform bilinear interpolation.

#### full_time_series.ipynb
Plots out the full time series (July 1987 - Present) of predicted CNN wind speeds and buoy observations.
Corresponds to figure 8 in the paper. Also used to give some useful metrics given in the paper.

#### NBS_CNN_Case1.ipynb
Trains the model for CNN case 1 in the paper (Normalized between 0-1, MSE and ReLU).
Outputs predictions on test set to csv.

#### NBS_CNN_Case2.ipynb
Trains the model for CNN case 2 (Standardized, MSE, and ReLU)
Outputs predictions on test set to csv.

#### NBS_CNN_Case3.ipynb
Trains the model for CNN case 3 (Standardized, Kernel MSE and leaky ReLU).
Outputs predictions on test set to csv.

#### perform_bilinear_and_take_nearest.py
Performs bilinear interpolation on the test set and grabs the nearest NBS observation 
for each observation in the test set.

#### plot_hyperparameter_oob_score.py
Plots the out of bag errors for each hyperparameter combination for the random forest model.
Corresponds to Figure 3 in the paper.

#### plot_grid.py
Plots the location of the buoy and gridded NBS product near the buoy. Figure 1 in the paper.

#### plot_test_results.py
Plots out the bar charts for errors (Figure 6) and time series comparison of test
set with predictions for all models (includes Figure 7). Figures that made it into
the paper have different names than what is written in the file as they were changed
after they were already put in the paper. 
fig6 = error_bars.png, fig7=CNN_comparison_time_series_test.png
In addition, many values provided in the paper are generated in the output of this script.

#### rf_test.py
Creates final random forest model that is trained on all training data with
optimal hyperparameters. Outputs predictions on test set.

#### split_buoy_five_fold.py
Assigns observations to each fold to be used in cross validation.
Points in folds 0-3 are used as the train data for the CNNs, fold 4 is the 
validation set for the CNN, "fold 5" is the test set.
Plots out time series with each fold its own color. Corresponds to figure 2 in the paper.

#### train_rf.py 
Performs 5 fold cross validation on random forest model and computes out of bag
error for all combinations of hyperparameters.
