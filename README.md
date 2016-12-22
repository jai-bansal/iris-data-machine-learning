#### Synopsis:
This project creates various supervised learning scripts in R and Python using iris data.
The R and Python scripts are contained in the R and Python branches, respectively.
These scripts are written to get the simplest possible version of the models up and running as quickly as possible, so there is no feature engineering and parameter tuning.
The small iris dataset (150 rows total) is also chosen for ease of use.

Below are the models contained in each branch. I've also included the library or modules used in parentheses.

R:
- Neural Network ('nnet' and 'e1071')
- Random Forest ('randomForest', 'RRF', and 'caret' implementations)
- Adaboost ('adabag')
- Support Vector Machine ('e1071')
- Logistic Regression ('nnet')
- K Nearest Neighbor ('kknn')
- Naive Bayes ('e1071')

Python (all models use 'sklearn'):
- Random Forest
- Adaboost
- Support Vector Machine
- Logistic Regression
- K Nearest Neighbor
- Naive Bayes

For each model, training set accuracy, accuracy derived from cross validation, and test set accuracy will be included.

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
