# This script shows some capabilities of the H2O platform.
# I train a few classification models.
# I also include grid search, auto-ML, and LIME explanations.

# LOAD LIBRARIES ----------------------------------------------------------
# This section loads relevant libraries for this script.
pacman::p_load(readr, h2o, lime)

# IMPORT DATA -------------------------------------------------------------
# This section imports data.
# The data files are located in the 'R' branch file, NOT the 'classification' or 'regression' folders.
train = read_csv('train.csv')
test = read_csv('test.csv')

# Change variable types
train$Species = as.factor(train$Species)
test$Species = as.factor(test$Species)

# H2O ---------------------------------------------------------------------
# This section shows H2O capabilities.

h2o.init()      # Initialize cluster

# Create H2O data frames
h2o_train = as.h2o(train)
h2o_test = as.h2o(test)

# Set independent and dependent variables
y = "Species"
x = c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")

# Regression models (xgboost is not currently supported on Windows as of 2019-11-11)
m = h2o.glm(x = x, y = y, training_frame = h2o_train, family = "multinomial")                  # Generalized linear model
m = h2o.randomForest(training_frame = h2o_train, x = x, y = y, ntrees = 100, max_depth = 10)     # Random forest
m = h2o.gbm(x = x, y = y, training_frame = h2o_train, ntrees = 100, max_depth = 5)

# Grid search

grid_search = h2o.grid(algorithm = "randomForest", x = x, y = y, training_frame = h2o_train, 
                       hyper_params = list(ntrees = c(50, 100), max_depth = c(10, 15)), grid_id = "test1")

# Sort grid results by metric of choice
r = h2o.getGrid(grid_id = "test1", sort_by = "rmse", decreasing = T)

# AutoML (takes awhile and semi-freezes computer!)
# test parameters to limit time spent per model and total training time!
auto_ml = h2o.automl(x = x, y = y, training_frame = h2o_train, verbosity = "info", max_runtime_secs_per_model = 60)

# Predict
h2o_preds = h2o.predict(m, h2o_test)

h2o.shutdown() # Shut down instance

# LIME --------------------------------------------------------------------
# This section shows how to get LIME explanations for the models trained above.

explainer <- lime(as.data.frame(h2o_test), model = m)
explanation <- lime::explain(as.data.frame(h2o_test[5,]), explainer, n_features = 3, n_labels = 3)
plot_features(explanation)
