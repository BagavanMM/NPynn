# NPyn - Build Neural Networks in Python
This repository shows the code to a project caleld NPynn <i>(En-pine)</i>, a deep learning framework that allows you to perform classifciation and regression

# Table of Contents
1. [What this framework includes](#what-this-framework-includes)
2. [How to build model](#how-to-build-model)


## What this framework includes 
Currently(v.1.0), this frameowrk is pretty basic. 
#### Layers:
 - Dense/Fully Connected Layers
 - Dropout Layers
 - Input Layers (For first layer of model)
#### Activations: 
- Rectified Linear Unit (ReLU)
- Softmax
- Sigmoid
- Linear
#### Optimizers:
 - SGD
 - Adagrad
 - RMSprop
 - Adam
#### Losses:
 - Categorical Crossentropy
 - Binary Crossentropy
 - Mean Squared Error (MSE)
 - Mean Absolute Error (MAE)
#### Accuracy:
 - Categorical
 - Regression


## How to build model
Let's go over how to build a model. This section won't go over how to load models, but just how to build a model. You can check out the example codes in this repository to see how you can load models.
<br> We first need to begin by calling our `Net` function.
```
model = Net()
```
 After calling our Neural Network function and storing it in the `model` variable. We will then go and add layers to our model using the `model.add()` function. 
 <br> Here is one example of adding layers.
 ```
 model.add(Dense(X.shape[1], 128))
model.add(ReLU())
model.add(Dense(128, 128))
model.add(ReLU())
model.add(Dense(128, 10))
model.add(Softmax())
```
#### Compile model
After building the actuall model, you woudl then need to compile the model using `model.compile()` with the accuracy, loss, and optimizer.
```
model.compile(
              loss = CategoricalCrossentropy(),
              accuracy = Categorical,
              optimizer = Adam(learning_rate=0.01)
              )
```
