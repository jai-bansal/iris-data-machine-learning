# This scripts implements LIME (locally interpretable model-agnostic explanations).
# LIME applies to classification models only.
# At this time (2016-06-12), LIME is honestly not ready for showtime.
# It seems to only currently work for 'caret' models.
# But, it's pretty cool stuff.

# LOAD LIBRARIES ----------------------------------------------------------
# This section loads libraries.
library(readr)
library(data.table)
library(dplyr)
library(caret)
library(lime)

# IMPORT DATA --------------------------------------------------------------
# This section imports data.
# The data files are located in the 'R' branch file, NOT the 'classification' or 'regression' folders.
# The commands below assume the working directory is set to the 'classification' folder.
train = data.table(read_csv('../train.csv'))
test = data.table(read_csv('../test.csv'))

# TRAIN RANDOM FOREST MODEL ------------------------------------------------------------
# This section trains a random forest model.

  # Train a random forest model.
  rf = train(Species ~ ., 
             data = train, 
             method = 'rf', 
             ntree = 100)

# LIME --------------------------------------------------------------------
# This section implements LIME.

  # Create explanation object.
  ex_obj = lime(train, 
                rf)

  # Explain a new observation (the plotting below is probably more intuitive).
  # Obviously, different observations can be viewed by changing the index of "test".
  ex_obj(test[50], 
         n_labels = 1,
         n_features = 4)
  
  # Plot the explanation above.
  # Obviously, different observations can be viewed by changing the index of "test".
  # It's odd the "Species" shows up?
  plot_features(ex_obj(test[10], 
         n_labels = 1,
         n_features = 4))


