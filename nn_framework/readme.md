# Neural Network Framework
This framework consists of three different main modules:
* NN_Layer
* Cost_Functions
* Neural_Network

Theses three modules are explained in the following.

## NN_Layer
This module contains the class _Layer_ which receives the following parameters:
* **size_pred**: this is the size of the predecessor layer
* **size**: this is the size of this layer
* **activation_function**: the standard option is "relu", the other option is "sigmoid"
* **W, b** (_optional_): Weights and bias for this layer. If no values are given, they are determined randomly. 

The following methods are available:
* **_intit_param()**: sets weights and bias randomly
* **activation_function(x)**: returns result of selected activation function
* **activation_function_derivative(x)**: returns result for the derived selected activation function
* **forward(input)**: performs forward propergation step for this layer

## Cost_Functions
This module contains the abstract class _Cost_Function_ with the following abstract methods:
* **cost_function(pred,gt)**: returns the cost for the given predicted and ground truth values
* **cost_function_deriv(pred,gt,X)**: returns _dW_, _db_ which are the gradients for the weights and the bias for layer _X_.

_Cost_Function_ is implemented by two different classes:
* **MSE**: 
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
, where​ $y_i$ represents the actual values and $\hat{y}_i$ represents the predicted values.

* **Cross_Entropy**:

This class receives the parameter _binary_ which differentiates between the usage of the _binary cross entropy_:
$$
L(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
$$
, where $ y_i$ represents the actual labels and $\hat{y}_i$ represents the predicted probabilities.

and _categorical cross entropy_:
$$
L(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij}) 
$$
, where $y_{ij}$ represents the actual probabilities of class $j$ for example $i$, and $\hat{y}_{ij}$ represents the predicted probabilities of class $j$ for example $i$.

## Neural_Network
This module combines the previous within the class _NeuralNetwork_. This class receives the following paramters:
* **layers**: list of layers
* **cost_function**: object of CostFunction-Class (MSE or CE)
* **softmax** (_True_): whether to use softmax on the output layer
* **adam_optimizer** (_False_): whether to use the Adam optimizer

This class contains various methods, but the _gradient_decent_ is the most important.
* **gradient_decent(X, Y, X_val, Y_val, alpha, iterations, batch_size)**: returns the trained weights and biases.

To perfom the gradient decent, the other methods of the NeuralNetwork class are used:
* **forward_propagation(X_batch)**: returns the cache which includes values for Z and A of the different layers
* **_backward_propagation(cache, X_batch, Y_batch)**: returns the gradients for the different weight and biases.
* **_update_params(grads, alpha)**: returns the updated weights and biases. If activated, it utilizes the Adam optimizer.

Some smaller methods:
* **get_predictions(AL)**: transforms output of last layer into numeric class label. (Used for comparison with ground truth) 
* **softmax_func(self,x)**: returns softmax for input x. If activated, used to convert last layer to softmax-layer. 

# Feature Scaling
The Feature Scaling file contains the Class StandardScaler.
The StandardScaler standardized the data by centering it around mean and normalizing it by dividing all values by the std deviation of the data.

The StandardScaler can, for example, be used like this:
```python
from nn_framework.feature_scaling import StandardScaler
scaler = StandardScaler()
# fit the StandardScaler on your training data
scaler.fit(X_train)
# scale the data that you want to scale
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Data can be transformed back to its original with inverse_transform
X_test = scaler.inverse_transform(X_test_)
```


# PCA
The PCA class is used for Principle Component Analysis.

You can use the class similariy to the StandardScaler:

```python
from nn_framework.PCA import PCA
pca = PCA(num_components = 50) # number of components you want to reduce the dimensionality to
# fit the PCA on your training data
pca.fit(X_train)
# scale the data that you want to scale
X_train_scaled = pca.transform(X_train)
X_test_scaled = pca.transform(X_test)
```

You can use the plot_explained_variance_ratio() method to plot the ratio of explained variance and number of dimensions. This allows you to apply the elbow method to find an optimum.


# Buffer Predictor 
The BufferPredictor Class can be used for smoothing continuous predictions.
It works by applying the argmax to the product of the n previous predictions.
The previous predictions each get an softmax applied to them and are then multiplied.

You can use it like this:
```python
# Initialize the BufferPredictor with the desired size of the buffer and the number of classes your model can predict
buffer_predictor = BufferPredictor(buffer_size, num_classes)

# give the output of your forward propagation to the buffer_predictor to get a prediction
AL, cache = nn.forward_propagation(X.T)
prediction = buffer_predictor.predict(AL)
```