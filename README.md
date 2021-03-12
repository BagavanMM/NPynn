# NPynn - Build Neural Networks in Python (v.1.1)
This repository shows the code to a project called NPynn <i>(En-pine)</i>, a deep learning framework that allows you to perform classification and regression

![NPynn Image](https://github.com/BagavanMM/NPynn/blob/a2b7854fd0808f5160408bf7ed58c805eddbbd27/Images/NPynn.PNG)

## Table of Contents
[Contents of Framework](#contents-of-framework)<br>
[How to Use NPynn](#how-to-use-npynn)<br>
[Intention](#intention)<br>
[Process](#how-did-i-do-it?)


## Contents of Framework
Currently(v.1.0), this framework is pretty basic. 
#### Layers:
*Dense/Fully Connected Layers, Dropout Layers, Input Layers (For first layer of model)*
#### Activations: 
*Rectified Linear Unit (ReLU), Softmax, Sigmoid, Linear*
#### Optimizers:
*SGD, Adagrad, RMSprop, Adam*
#### Losses:
*Categorical Cross Entropy, Binary Crossentropy, Mean Squared Error (MSE), Mean Absolute Error (MAE)*
#### Accuracy:
*Categorical and Regression*


## How to use NPynn
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
After building the actual model, you would then need to compile the model using `model.compile()` with the accuracy, loss, and optimizer.
```
model.compile(loss = CategoricalCrossentropy(),
              accuracy = Categorical,
              optimizer = Adam(learning_rate=0.01))
```
Before training, we need to finalize our model using the `model.finalize()` function. This function will *finalize* the model so that it's ready to train.
```
model.finalize()
```

#### Train our model
Finally, after finalizing our model we will now begin to train it. This will be done with the `model.train()` function. This function has 6 parameters. The first two parameters are where you add your labels and features(X, and y). `model.train(X, y)` <br>
The third parameter is the `validation_data` parameter. If you have validation data, you can use this parameter to add your data. `model.train(..., validation_data=(X_val, y_val)` <br>
You will then need to choose the number of `epochs` as well as `batch_size`. `model.train(..., epochs=5, batch_size=128)` <br>
Finally, you can specify when you want your model to print the summary while training with the `print_every` function. `model.train(..., print_every=100)`. Keep in mind that this function will be 100 by default.
<br> Putting this code all together you would train your model like this:
```
model.train(X, y, validation_data=(X_val, y_val), epochs=5, batch_size=128, print_every=100)
```
This is how you can build and train your own neural network wtih NPynn. <br>
[Here are some more examples of using NPynn](https://github.com/BagavanMM/Npyn/tree/main/Code%20Examples)


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## Intention
Artificial Intelligence has been revolutionizing the world! Whether that's from self-driving cars, or fighting the Coronavirus pandemic! However, there is still a ton of complex math behind neural networks! Take this for example:
![Forward Pass Math](https://github.com/BagavanMM/NPynn/blob/2a43734578c7e14c4b1d9dac4f019bd8d680227c/Images/NeuralNetworkMath.PNG)
And this is supposed to be the 'simple' part of a neural network! Yea...

Thankfully, we have libraries like Tensorflow, PyTorch and Keras that do all of this math for us! Meaning that you don't need to understand what this means in order to build your own AI models!
<br>
So then I thought: "*hmmm, what if I just build my own deep learning library for people to use?*"
<br>
This would mean that I would need to learn all of that math in order to build one on my own! 
This might seem crazy, but this is actually really helpful. Cause not only am I going to be learning all of that fun math, but I'm also going to be able to put that into practice as well!

## How did I do it?
Well, we're here right? So how did I do it?
<br>
Well, I did this by using the top-down method.
### The Top-Down Method
The top down method is when you have an end goal in mind and work your way down on how to complete that goal. This might seem a bit confusing so I'll be using my example to teach you.
<br> First of all, my end goal was to build my own deep learning framework. 
<br> In order to do that, I first need to understand the math behind neural networks
<br> In order to understand the math behind neural networks, you need to learn calculast
<br> And if you want to learn calculus, you need to have a solid understanding of algebra and trigonometry
<br> *You basically keep going down this process until you stop at something that you already know*
<br> ...
<br> In order to understand *X*, I need to know how to multiply. Oh wait, I know how to multiply!
<br>
So then you just start there, and once you learn the first thing, you will be able to learn the next until you get to your goal.
<br> This was essentially what I did in order to learn the math behind neural networks. However, I'm going to be honest. I still don't fully understand some parts of it. 

### Programming Process
Before I start building my framework, I still need to have to know how to use that math in practice. This is why I decided to first start off by learning how to build neural networks from scratch with Numpy. I've learned how to build these neural networks, and train them. All I would need to do is put that into a framework!
<br> Simple right? I wish it was...
After spending days looking at other deep learning frameworks and practicing building my own, I finally finished! Well... there were still a ton of bugs.
<br> After fixing those bugs, I finally finished building it, and started building my own models from my own framework!

<br>
<br>
<br>
<br>
And that is how I basically built my own framework!
