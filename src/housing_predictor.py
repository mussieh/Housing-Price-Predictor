# CS 480

# Artificial Intelligence Project

# This program is used to predict the Adair county housing prices
# using multi-layer perceptrons (deep learning). It is basically
# a regression model to predict the selling price of properties
# based on past data, which is used to train this software agent.
# 80% of the data set is used for training purposes, while the
# remaining 20% is used for testing.

# Group Members : Ronit Das, Zorig Magnaituvshin, Mussie Habtemichael,
# Jerry Lin

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import tensorflow as tf

# Constants for normalizing features
AGE_MEAN = 56.752808988764
AGE_STD_DEV = 33.6793978362266
FLOOR_AREA_MEAN = 2275.94382022472
FLOOR_AREA_STD_DEV = 1987.30658498307
LOT_SIZE_MEAN = 66188.3370786517
LOT_SIZE_STD_DEV = 158780.121251338
PRICE_PER_AREA_MEAN = 70.8013483146068
PRICE_PER_AREA_STD_DEV = 29.5289033749143
BATHS_TO_BEDS_MEAN = 0.707078651685394
BATHS_TO_BEDS_STD_DEV = 0.262081711092305

# Data sets for training and testing. The .csv files contain the data that we collected
# over different sites including Zillow, Trulia, etc.
HOUSING_TRAINING = "housing_prediction_training.csv"
HOUSING_TEST = "housing_prediction_test.csv"

# Class for predicting housing prices in Adair County
class PricePredictor:

    # Constructs the PricePredictor object with given parameters
    def __init__(self, beds=0, baths=0, age=0, floorarea=0.0, lot_size=0.0, price_sqft=0.0):
        self.age = age
        self.floorarea = floorarea
        self.lot_size = lot_size
        self.baths = baths
        self.beds = beds
        self.price_sqft = price_sqft

        # Load datasets into Tensorflow's input pipeline. All our features are continuous.
        # Our target value (selling price) is also continuous.
        training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=HOUSING_TRAINING,
            target_dtype=np.int,
            features_dtype=np.float32)

        test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=HOUSING_TEST,
            target_dtype=np.int,
            features_dtype=np.float32)

        # Specify that all features have real-value data. We are considering 6 different
        # features for every property which are : ratio of #bedrooms to #bathrooms,
        # area in sq.ft, lot size in sq.ft, price per sq.ft, and age of the property.
        feature_columns = [tf.feature_column.numeric_column("x", shape=[5])]

        # Build 4 layer DNN with 8, 10, 10, and 10 units in each layer respectively.
        self.regressor_model = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                                     hidden_units=[8, 10, 10, 8])

        # Defining the training input pipeline
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(training_set.data)},
            y=np.array(training_set.target),
            num_epochs=None,
            shuffle=True)

        # Define the test input pipeline
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(test_set.data)},
            y=np.array(test_set.target),
            num_epochs=1,
            shuffle=False)

        # Training the model
        self.regressor_model.train(input_fn=train_input_fn, steps=25000)

        # evaluate_result = regressor_model.evaluate(input_fn=test_input_fn, steps=1)


    # Set the housing price parameters
    def setParameters(self, beds, baths, age, floorarea, lot_size, price_sqft):
        self.age = float(age.get())
        self.floorarea = float(floorarea.get())
        self.lot_size = float(lot_size.get())
        self.baths = float(baths.get())
        self.beds = float(beds.get())
        self.price_sqft = float(price_sqft.get())

    # Function that employs machine learning for outputting the predicted price
    def predict(self):

      # Normalizing the features
      norm_age = (self.age - AGE_MEAN) / AGE_STD_DEV
      norm_floor_area = (self.floorarea - FLOOR_AREA_MEAN) / FLOOR_AREA_STD_DEV
      norm_lot_size = (self.lot_size - LOT_SIZE_MEAN) / LOT_SIZE_STD_DEV
      norm_bath_bed = ((self.baths/self.beds) - BATHS_TO_BEDS_MEAN) / BATHS_TO_BEDS_STD_DEV
      norm_price_per_area = (self.price_sqft - PRICE_PER_AREA_MEAN) / PRICE_PER_AREA_STD_DEV

      new_samples = np.array(
        [[norm_age, norm_floor_area, norm_lot_size, norm_bath_bed, norm_price_per_area]], dtype=np.float32)

      predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_samples},
        num_epochs=1,
        shuffle=False)

      # Print out the predictions
      y = self.regressor_model.predict(input_fn=predict_input_fn)

      # Convert to a list and return the prediction extracted from the dictionary
      predictions = list(itertools.islice(y, 6))
      for i in predictions:
        return str(i["predictions"][0])
