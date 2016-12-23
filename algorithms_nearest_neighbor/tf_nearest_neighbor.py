from __future__ import print_function

import numpy as np
import tensorflow as tf
import csv


def load_dataset(filename, split, training_set=[], test_set=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            #print(x)
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if x < split:
                #print("train")
                training_set.append(dataset[x][0:4])
            else:
                #print("test")
                test_set.append(dataset[x][0:4])


# prepare data
Xtr = []
Xte = []
split = 75
load_dataset('iris.data', split, Xtr, Xte)

print("len(Xtr): %s" % len(Xtr))
print("len(Xte): %s" % len(Xte))

# tf Graph Input
xtr = tf.placeholder("float", [None, 4])
xte = tf.placeholder("float", [4])

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.neg(xte))), reduction_indices=1)
# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    # for i in range(len(Xte)):
    # Get nearest neighbor
    i = 1
    print("Xte[i]: %s" % Xte[i])
    nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i]})
    print("nn_index: %s" % nn_index)
    print("found this prediction: %s" % Xtr[nn_index])
    # Get nearest neighbor class label and compare it to its true label
    print("Test", i, "Prediction:", np.argmax(Xtr[nn_index]), "True Class:", np.argmax(Xte[i]))
    # Calculate accuracy
    if np.argmax(Xtr[nn_index]) == np.argmax(Xte[i]):
        accuracy += 1./len(Xte)

    print("Done!")
    print("Accuracy:", accuracy)