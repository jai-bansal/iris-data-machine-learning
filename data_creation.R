# This script creates the training and test data sets from the 'iris' data set.
# The training and test sets will then be used in the machine learning scripts in the R and Python branches of the 'iris-data-machine-learning' repository.
# The output of this script is two files: 'train.csv' and 'test.csv' containing the training and test data, respectively.
# 'train.csv' is 100 rows and 'test.csv' is 50 rows.

# Load packages.
library(data.table)

# Load data.
data(iris)
iris_data = data.table(iris)

# Randomly divide the data into training and test data.

  # Set random seed.
  set.seed(1)
  
  # Create 100 values that will be the row indices for the training data.
  training_indices = sample(1:nrow(iris_data), 100, replace = F)
  
  # Create 'train' consisting of rows from 'iris_data' with row indices contained in 'training_indices'.
  train = iris_data[training_indices, ]
  
  # Create 'test' consisting of rows from 'iris_data' with row indices NOT contained in 'training_indices'.
  test = iris_data[(-training_indices), ]
  
# Export 'train' and 'test'.
write.csv(train, 'train.csv', row.numbers = F)
write.csv(test, 'test.csv', row.numbers = F)