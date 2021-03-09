# NPynn - Build Neural Networks in Python
This repository shows the code to a project caleld NPynn <i>(En-pine)</i>, a deep learning framework that allows you to perform classifciation and regression

# Table of Contents
[What this framework includes](#what-this-framework-includes)<br>
[How to build model](#how-to-build-model)


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
Before training, we need to finalize our model using the `model.finalize()` function. This function will *finalize* the model so that the its ready to train.
```
model.finalize()
```

#### Train our model
Finally, after finalizing our model we will now begin to train it. This will be done with the `model.train()` function. This function has 6 parameters. The first two parameters are where you add your labels and features(X, and y). `model.train(X, y)` <br>
The third parameter is the `validation_data` parameter. If you have validation data, you can use this parameter to add your data. `model.train(..., validation_data=(X_val, y_val)` <br>
You will then need to choose the number of `epochs` as well as `batch_size`. `model.train(..., epochs=5, batch_size=128)` <br>
Finally, you can specify when you want your model to print the summary while training with the `print_every` function. `model.train(..., print_every=100)`. Keep in mind that the this funciton will be 100 by default.
<br> Putting this code all together you woudl train your model like this:
```
model.train(X, y, validation_data=(X_val, y_val), epochs=5, batch_size=128, print_every=100)
```
This is how you can build and train your own neural network wtih NPynn. <br>
[Here are some more examples of using NPynn](https://github.com/BagavanMM/Npyn/tree/main/Code%20Examples)
