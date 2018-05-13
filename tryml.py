from uwimg import *

def softmax_model(inputs, outputs):
    l = [make_layer(inputs, outputs, SOFTMAX)]
    return make_model(l)

def neural_net(inputs, outputs):
    print inputs

    # TWO LAYERS WITH DEFAULT PARAMS
    # l = [make_layer(inputs, 32, LOGISTIC), make_layer(32, outputs, SOFTMAX)]                            # training accuracy: 0.889283333333  test accuracy:     0.8949
    # l = [make_layer(inputs, 32, LINEAR), make_layer(32, outputs, SOFTMAX)]                              # training accuracy: 0.913316666667  test accuracy:     0.9162
    l = [make_layer(inputs, 32, RELU), make_layer(32, outputs, SOFTMAX)]                                # training accuracy: 0.92605         test accuracy:     0.9281
    # l = [make_layer(inputs, 32, LRELU), make_layer(32, outputs, SOFTMAX)]                               # training accuracy: 0.926166666667  test accuracy:     0.9281
    # l = [make_layer(inputs, 32, SOFTMAX), make_layer(32, outputs, SOFTMAX)]                             # training accuracy: 0.09915         test accuracy:     0.1009

    # THREE LAYERS WITH RATE .1 AND ITERS 3000 ALL ELSE DEFAULT
    # l = [make_layer(inputs, 64, LRELU), make_layer(64, 32, LRELU), make_layer(32, outputs, SOFTMAX)]    # training accuracy: 0.979616666667  test accuracy:     0.9663
    # l = [make_layer(inputs, 64, LRELU), make_layer(64, 32, RELU), make_layer(32, outputs, SOFTMAX)]     # training accuracy: 0.984           test accuracy:     0.9697
    # l = [make_layer(inputs, 64, LRELU), make_layer(64, 32, LINEAR), make_layer(32, outputs, SOFTMAX)]   # training accuracy: 0.9804          test accuracy:     0.9663 
    # l = [make_layer(inputs, 64, LRELU), make_layer(64, 32, LOGISTIC), make_layer(32, outputs, SOFTMAX)] # training accuracy: 0.982166666667  test accuracy:     0.968
    # l = [make_layer(inputs, 64, LRELU), make_layer(64, 32, LRELU), make_layer(32, outputs, LRELU)]      # training accuracy: 0.98445         test accuracy:     0.9739
    # but all loss are nan
    # l = [make_layer(inputs, 128, LRELU), make_layer(128, 64, LRELU), make_layer(64, 32, LRELU), make_layer(32, outputs, LRELU)] 
    # l = [make_layer(inputs, 256, LRELU), make_layer(256, 128, LRELU), make_layer(128, outputs, LRELU)]
    return make_model(l)

print("loading data...")
train = load_classification_data("mnist.train", "mnist.labels", 1)
test  = load_classification_data("mnist.test", "mnist.labels", 1)
# train = load_classification_data("cifar.train", "cifar/labels.txt", 1)
# test = load_classification_data("cifar.test", "cifar/labels.txt", 1)
print("done")
print

print("training model...")
batch = 128
iters = 1000
rate = .01
momentum = .9
decay = .0 

# SOFTMAX
# m = softmax_model(train.X.cols, train.y.cols)
# rate = 10      # training accuracy: 0.09915        test accuracy:     0.1009
# rate = 1       # training accuracy: 0.850616666667 test accuracy:     0.8464
# rate = .1      # training accuracy: 0.920716666667 test accuracy:     0.9171
# rate = .01     # training accuracy: 0.903433333333 test accuracy:     0.9091
# rate = .001    # training accuracy: 0.859033333333 test accuracy:     0.8669

# decay = 1      # training accuracy: 0.899033333333 test accuracy:     0.9049
# decay = .1     # training accuracy: 0.9031         test accuracy:     0.9089
# decay = .01    # training accuracy: 0.90345        test accuracy:     0.9092
# decay = .001   # training accuracy: 0.90345        test accuracy:     0.9091
# decay = .0001  # training accuracy: 0.903433333333 test accuracy:     0.9091
# decay = .00001 # training accuracy: 0.903433333333 test accuracy:     0.9091



# LRELU + SOFTMAX
m = neural_net(train.X.cols, train.y.cols)
# rate = 10      # training accuracy: 0.09915        test accuracy:     0.1009
# rate = 1       # training accuracy: 0.09915        test accuracy:     0.1009
rate = .1      # training accuracy: 0.947583333333 test accuracy:     0.9431
# rate = .01     # training accuracy: 0.926166666667 test accuracy:     0.9281
# rate = .001    # training accuracy: 0.86715        test accuracy:     0.8766


# training accuracy: 0.971616666667
# test accuracy:     0.969
# training accuracy: 0.970933333333
# test accuracy:     0.9678
# training accuracy: 0.972083333333
# test accuracy:     0.9701
# with rate = .1
# decay = 1      # training accuracy: 0.923566666667 test accuracy:     0.9257
# decay = .1     # training accuracy: 0.949633333333 test accuracy:     0.9447
# decay = .01    # training accuracy: 0.95745        test accuracy:     0.9531
# decay = .001   # training accuracy: 0.95705        test accuracy:     0.9509
# decay = .0001  # training accuracy: 0.95595        test accuracy:     0.9513
# decay = .00001 # training accuracy: 0.950666666667 test accuracy:     0.9452

# LRELU + RELU + SOFTMAX 
# iters = 3000
# decay = .0001 # training accuracy: 0.982283333333  test accuracy:     0.971
# decay = .001  # training accuracy: 0.981166666667  test accuracy:     0.9692
# decay = .01   # training accuracy: 0.983533333333  test accuracy:     0.9724
# decay = .1    # training accuracy: 0.97495         test accuracy:     0.9686
# decay = 1     # training accuracy: 0.9208          test accuracy:     0.924

# CIFAR
# l = [make_layer(inputs, 64, LRELU), make_layer(64, 32, LRELU), make_layer(32, outputs, LRELU)]
# rate =  .01  decay = .0      training accuracy: 0.3961  test accuracy:     0.3952
# rate =  .1   decay = .0      training accuracy: 0.26368 test accuracy:     0.2689
# rate =  .001 decay = .0      training accuracy: 0.32366 test accuracy:     0.32
# rate = 1     decay = .0      training accuracy: 0.1     test accuracy:     0.1
# rate =  .01  decay = .01     training accuracy: 0.39446 test accuracy:     0.3938
# rate =  .01  decay = .00001  training accuracy: 0.4032  test accuracy:     0.3981
# rate =  .01  decay = .000001 training accuracy: 0.3961  test accuracy:     0.3952
# rate =  .01  decay = .0001   training accuracy: 0.3837  test accuracy:     0.3815 


# l = [make_layer(inputs, 128, LRELU), make_layer(128, 64, LRELU), make_layer(64, outputs, LRELU)]
# rate = .01 decay = .00001  training accuracy: 0.41442 test accuracy:     0.4089
# l = [make_layer(inputs, 256, LRELU), make_layer(256, 128, LRELU), make_layer(128, outputs, LRELU)]
# rate = .01 decay = .00001  training accuracy: 0.42512 test accuracy:     0.4181


train_model(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: " + str(accuracy_model(m, train)))
print("test accuracy:     " + str(accuracy_model(m, test)))


## Questions ##

# 2.2.1 Why might we be interested in both training error and testing error? What do these two numbers tell us about our current model?
# Looking at both is helpful in detecting overfitting (high training accuracy, low testing accuracy) vs an inappropriate model (low training and testing accuracy). Training accuracy gives a good baseline on how appropriate our model is and testing accuracy tells us how well our model generalizes.


# 2.2.2 Try varying the model parameter for learning rate to different powers of 10 (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the model. What patterns do you see and how does the choice of learning rate affect both the loss during training and the final model accuracy?
# Learning rates larger than 1 result in nan loss and higher loss overall and terrible accuracies. Larger learning rates decays the loss/makes the loss converge faster, but can get stuck at worse values resulting in worse accuracies (for example when the learning rate was 1). The smaller learning rates took more time for the loss to converge, but were getting down to lower losses toward the end of the iterations and might have just needed a few more iterations to converge properly that would result in better accuracies. It felt that if we tested even smaller learning rates, the learning rate would be so slow that the loss would start decreasing linearly instead of kind of exponentially. For 1000 iterations, .01 was the sweet spot.  


# 2.2.3 Try varying the parameter for weight decay to different powers of 10: (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5). How does weight decay affect the final model training and test accuracy?
# High decay (here high means decay greater or equal to 1) actually decreases both accuracies compared to having a decay of 0. The accuracies improve and peak at when decay is .01. The accuracies then get worse as decay decreases and seems to go towards accuracies of when decay is 0. 


# 2.3.1 Currently the model uses a logistic activation for the first layer. Try using a the different activation functions we programmed. How well do they perform? What's best?
# LOGISTIC  >training accuracy: 0.889283333333     >test accuracy:     0.8949
# LINEAR    >training accuracy: 0.913316666667     >test accuracy:     0.9162
# RELU      >training accuracy: 0.92605            >test accuracy:     0.9281
# LRELU     >training accuracy: 0.926166666667     >test accuracy:     0.9281
# SOFTMAX   >training accuracy: 0.09915            >test accuracy:     0.1009   # because we didn't actually calculate SOFTMAX's gradient and used a loophole that allowed SOFTMAX to work if it was the last layer.
# LRELU was the best, only slightly better than RELU by having a better training accuracy.


# 2.3.2 Using the same activation, find the best (power of 10) learning rate for your model. What is the training error and testing error?
# rate = .1 = 10^-1  >training accuracy: 0.947583333333  >test accuracy:     0.9431

# 2.3.3 Right now the regularization parameter `decay` is set to 0. Try adding some decay to your model. What happens, does it help? Why or why not may this be?
# Adding a decay greater than or equal to 1 worsened both accuracies probably because when decay is 1 or larger, we penalize a bit too heavily. Adding a decay less than 1 and greater than 0 improved both accuracies and peaked at when decay is .01 probably because .01 got the right balance of limiting how much weights can fluxuate but still giving it enough freedom to reach effective weights and going lower allows the weights to bounce around/fluxuate a little bit too much. 

# 2.3.4 Modify your model so it has 3 layers instead of two. The layers should be `inputs -> 64`, `64 -> 32`, and `32 -> outputs`. Also modify your model to train for 3000 iterations instead of 1000. Look at the training and testing error for different values of decay (powers of 10, 10^-4 -> 10^0). Which is best? Why?
# decay = .01  >training accuracy: 0.983533333333   >test accuracy:     0.9724   because it resulted in the best accuracies with the model I ended up using which was LRELU->RELU->SOFTMAX. One thing that was weird and idk how to interpret is that decay of .0001 performed better than .001, but then decay of .01 spiked up to be the best before having .1 be worse than .001 and better than 1.

# 3.2.1 How well does your network perform on the CIFAR dataset?
# My best network: l = [make_layer(inputs, 256, LRELU), make_layer(256, 128, LRELU), make_layer(128, outputs, LRELU)] with 1000 iterations, a learning rate of .01, decay of .00001, batch = 128, and momentum of .9. It resulted in a training accuracy of 0.42512 and test accuracy of 0.4181. It took goddamn forever to run and there's no extra credit attached to this, so I'm gonna call it good. I accidentally found that three layers of LRELU did the best out of all the three layered neural networks that I tried. I then found the best learning rate for the three LRELU layers of inputs->64, 64->32, 32->outputs, then the best decay for it as well. I deleted the code, but I played around with momentum and I .9 ended up being the best (I tried .9, .5, .85 (I think), and .95). I then tried different input/output sizes for each layer. 

