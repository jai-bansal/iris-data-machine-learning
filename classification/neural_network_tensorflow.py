# This script creates and evaluates a neural network classification model in Python for the 'iris' data.
# It uses the 'tensorflow' package.
# This iteration gets the model working as quickly as possible; there is no
# cross validation, feature engineering, or parameter tuning.

################
# IMPORT MODULES
################
# This section imports necessary modules.
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import tensorflow as tf

#############
# IMPORT DATA
#############
# This section imports data.
# The data files are located in the 'Python' branch file, NOT the 'classification' or 'regression' folders.
# The commands below assume the working directory is set to the 'classification' folder.
os.chdir('..')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

###########
# PREP DATA
###########
# This section preps the data for model training.

# Get training and test data (not labels).
train_data = train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']].as_matrix()
test_data = test[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']].as_matrix()

# Turn training and test labels into one-hot vectors.

# First, I turn training and test labels into integers instead of characters.

# Create label encoder and fit on training labels.
text_to_num_transformer = LabelEncoder().fit(train.Species)

# Convert training and test labels to integers.
train_labels_num = text_to_num_transformer.transform(train.Species)
test_labels_num = text_to_num_transformer.transform(test.Species)

# Now, turn integer labels into one-hot vectors for model below.
train_labels_one_hot = LabelBinarizer().fit_transform(train_labels_num)
test_labels_one_hot = LabelBinarizer().fit_transform(test_labels_num)

##############
# CREATE MODEL
##############
# This section creates a neural network classification model using 'tensorflow'.

# Specify number of classes (output labels).
classes = 3

# Specify number of hidden nodes.
hidden = 1000

# Set # of iterations to run.
steps = 10

# Set up graph.
graph = tf.Graph()
with graph.as_default():

    # Set random seed.
    tf.set_random_seed(20170907)

    # Input training and test data and labels.
    # I use constant data instead of placeholder data since the 'iris' data is really small.
    train_data = tf.constant(train_data,
                             dtype = tf.float32)
    test_data = tf.constant(test_data,
                            dtype = tf.float32)
    train_labels = tf.constant(train_labels_one_hot)
    test_labels = tf.constant(test_labels_one_hot)

    # Define 2 layers of weights and biases.
    w_1 = tf.Variable(tf.truncated_normal([4, hidden]))
    b_1 = tf.Variable(tf.zeros([hidden]))
    w_2 = tf.Variable(tf.truncated_normal([hidden, classes]))
    b_2 = tf.Variable(tf.zeros([classes]))

    # Specify training computation.
    mm_1 = tf.matmul(train_data, w_1) + b_1
    relu_1 = tf.nn.relu(mm_1)
    mm_2 = tf.matmul(relu_1, w_2) + b_2

    # Specify loss function.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = train_labels,
                                                                  logits = mm_2))

    # Define optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Generate predictions.
    train_pred = tf.nn.softmax(mm_2)
    test_pred = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(test_data, w_1) + b_1), w_2) + b_2)

# Run graph.
with tf.Session(graph = graph) as session:

    # Initialize variables.
    session.run(tf.global_variables_initializer())
    
    # Iterate.
    for step in range(steps):
    
        # Run optimizer.
        _, l, pred = session.run([optimizer, loss, train_pred])
    
        # Print loss every step.
        print('Step ', step, ' Loss: ', l)

        # Print training set accuracy every step.
        print('Step ', step, ' Training Set Accuracy: ',
              tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1),
                                              tf.argmax(train_labels, 1)),
                                     tf.float32)).eval())
        print('')
    
    # Print test set accuracy.
    print('Test Set Accuracy: ',
          tf.reduce_mean(tf.cast(tf.equal(tf.argmax(test_pred, 1),
                                          tf.argmax(test_labels, 1)),
                                 tf.float32)).eval()) 








