#### Summary:
This repository contains classification and regression supervised learning scripts in R and Python using 'iris' data.
R and Python scripts are in the R and Python branches, respectively.
Classification and regression scripts are contained in the 'classification' and 'regression' folders in each branch, respectively.
These scripts get the simplest possible version of models running as quickly as possible, so there is no / minimal feature engineering or parameter tuning.
The small 'iris' dataset is also chosen for ease of use.

Scripts are named in an obvious way. In the R branch, 'classification' folder, there is a script for LIME. LIME is not a supervised learning model but helps to explain predictions. You can read more about it here: https://homes.cs.washington.edu/~marcotcr/blog/lime/. I couldn't get LIME to work for Python.

Where possible, training, cross validation, and test set accuracy are included.

#### Motivation:
I created this project to explore various supervised learning models in R and Python.

#### Dataset Details:
I use the famous 'iris' dataset in R.
The dataset gives the measurements in centimeters of the variables sepal length and width and petal length and width, respectively, for 50 flowers from each of 3 species of iris. The species are Iris setosa, versicolor, and virginica.
The models will be predicting the species.
A csv file of the data is included in each branch. The script to create the csv
More information on the data can be found here:
https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/iris.html

I divide the data into 100 training examples and 50 test examples.
The 'data_creation.R' file in the 'master' branch creates the training and test sets.
Note that this file can only be run in R.
The training and test files are called 'train.csv' and 'test.csv' in all 3 branches.

#### License:
GNU General Public License
