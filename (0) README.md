# Musk-Classification-of-Chemical-Compounds-with-Deep-Neural-Network-in-Keras-on-GPU

This is a Classifier model for Musk classification of Organic Chemical Compounds dataset of 6598 datapoints. Deep Neural Network is built with Keras Library and Evaluation of model is also done after training the model on GPU.

## Dataset Overview

he given dataset contains details about organic chemical compounds including their chemical features, isomeric conformation, names and the classes in which they are classified. The compounds are classified as either ‘Musk’ or ‘Non-Musk’ compounds.  There are 166 Chemical features are presented for each compound.

## Model Overview

This is the code of Deep Neural Network Softmax Classifier which is implemented on Organic Chemical Compounds dataset.

Train set is 80% and Test(validations) set is 20% of the entire dataset, which is 5278 and 1320 respectively.

Sequential Layer model of 4 Hidden layers [100,50,50,50] is used in Deep Neural Network. Categorical Cross Entropy is used to obtain loss of the model and Adam optimizer is used to optimize the model. 60 epochs are set for as model's hyperparameter.

## Conclusion

After 48 epochs Training Accuracy and Validation Accuracy became constant at 1.00 and 0.9992, respectively. 

Following Metrices are achieved by training the model:
1. f1 score: 0.9976019184652278
2. precision: 1.0
3. recall: 0.9952153110047847
4. Validation Loss: 0.001435200567357242
5. Validation Accuracy: 0.9992424249649048

## Graphs
### Model Accuracy
![Model Accuracy](https://github.com/ajaychouhan-nitbhopal/Musk-Classification-of-Chemical-Compounds-with-Deep-Neural-Network-in-Keras-on-GPU/blob/main/(4)_Model_Accuracy.jpg?raw=true)

### Model Loss
![Model Loss](https://github.com/ajaychouhan-nitbhopal/Musk-Classification-of-Chemical-Compounds-with-Deep-Neural-Network-in-Keras-on-GPU/blob/main/(5)_Model_Loss.jpg?raw=true)


## Dependencies

[numpy](https://numpy.org/)

[pandas](https://pandas.pydata.org/)

[TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras)

[scikt-learn](https://scikit-learn.org/stable/)

[matplotlib](https://matplotlib.org/)

[keras](https://keras.io/)

Install missing dependencies with [pip](https://pip.pypa.io/en/stable/)

## Usage
1. (1) Musk Classification of Chemical Compounds with Deep Neural Network in Keras on GPU.ipynb is Jupyter Notebook which contains classifier model.
2. (2) musk_csv.csv is csv file of abovementioned dataset.
3. (3) model.h5 is a HDF5 file which contains parameters of the DNN classifier.
4. (4) Model Accuracy.JPG is a jpg file which contains Graph of training and validation accuracies vs epoch.
5. (5) Model Loss.JPG is a jpg file which contains Graph of training and validation loss vs epochs.

Install jupyter [here](https://jupyter.org/install).
