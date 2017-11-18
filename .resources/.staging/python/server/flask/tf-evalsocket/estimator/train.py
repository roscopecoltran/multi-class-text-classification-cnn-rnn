import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)


# Load datasets.
training_set = mnist.train.images
test_set = mnist.test.images

# Specify that all features have real-value data
feature_name = "digit_features"
feature_columns = [tf.feature_column.numeric_column(feature_name, 
                                                    shape=[784])]

classifier = tf.estimator.LinearClassifie(
    feature_columns=feature_columns,
    n_classes=10,
    model_dir="/tmp/mnist_model",
    hidden_units=[100, 70, 50, 25])

def input_fn(dataset):
    def _fn():
        features = {feature_name: tf.constant(dataset.data)}
        label = tf.constant(dataset.target)
        return features, label
    return _fn

# Fit model.
classifier.train(input_fn=input_fn(training_set),
               steps=1)

print('fit done')

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=input_fn(test_set), 
                                     steps=1)["accuracy"]
print('\nAccuracy: {0:f}'.format(accuracy_score))

# Export the model for serving
feature_spec = {'digit_features': tf.FixedLenFeature(shape=[784], dtype=np.float32)}

serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

classifier.export_savedmodel(export_dir_base='/tmp/mnist_model' + '/export', 
                            serving_input_receiver_fn=serving_fn)