# This scripts implements LIME (locally interpretable model-agnostic explanations).
# LIME applies to classification models only.
# At this time (2016-06-12), LIME is honestly not ready for showtime.
# It seems to only currently work for 'caret' models.
# But, it's pretty cool stuff.

################
# IMPORT MODULES
################
# This section imports modules.
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lime
import lime.lime_tabular

#############
# IMPORT DATA
#############
# This section imports data.
# The data files are located in the 'Python' branch file, NOT the 'classification' or 'regression' folders.
# The commands below assume the working directory is set to the 'classification' folder.
train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

################################
# TRAIN RANDOM FOREST CLASSIFIER
################################
# This section trains a random forest classifier.

# Create classifier.
rf = RandomForestClassifier(n_estimators = 100,
                            oob_score = True,
                            random_state = 1)

# Fit classifier on training data.
rf.fit(train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']],
       train['Species'])

######
# LIME
######
# This section implements LIME.

#exp = lime.lime_tabular.LimeTbaularExplainer(train,
#                                             feature_names = list(train.columns.values),
#                                             class_names = ['Species'])
#test = lime.lime_tabular.LimeTabularExplainer()
