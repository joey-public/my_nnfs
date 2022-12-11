# Neural Network Notes

## High Level Overview

At a high level a neural network can be viewed as a black box that takes some vector as an input and produces another vector at the output. The length of the input and output vector do not need to match and in general will be different lengths. 

The output vector is the result of several matrix vector calculations performed by the hidden layers of the network.

Typically the output vector represents a probability distribution where the element with the highest value represents the "answer" that the neural network has reached. 

Ultimately a neural network can be fully represented by a set of Weight matrices, bias vectors, and activation functions. 

## Forward Propagation

We can represent the input to the network as a column $(1\times N_X)$ vector:
$$\vec x = \begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ :\\ x_{N_x} \end{bmatrix}$$

The first hidden layer has $N_{1}$ neurons, so it takes $\vec x$ as an input an produced $N_{1}$ outputs (one for each neuron in the layer). We can represned hidden Layer 1 as a weight Matrix $W_{1}$, a bias vector $\vec b_1$, and an activation function $f_1(x)$. Then the output of the layer $\vec y_1$ is defined as:
$$y_1 = f(\vec xW+\vec b)$$

Looking at the shapes of each vector and matrix we see:
$$(1\times N_1)=(1\times N_x)(N_x\times N_1)+(1\times N_1)$$
So that we must have a Weight matrix with shape of $(N_x \times N_1)$ or (# inputs$\times$# neurons in hidden layer). And then the bias vector mush have shape $(1 \times N_1)$ or (1$\times$# neurons in hidden layer)

Theoretically a neural network can hae any number of hidden layers with each layer haveing any number of neurons assosiated with it.

To find the toal number of tunable parameters (weights and biases) for a network we need to know the number of layers in the network and how many neurons are in each layer. For example assume a network with $M$ total layers where the vector $\vec k$ is a $(1 \times M+1)$ vector holding the number of neurons for each layer. The total tunable parameters $P_{tot}$ can be found as:
$$P_{tot} = Biases + Weights$$
The biases can be found by summing the elements of $\vec k$ and the total Weights can be found as a dotproduct of the $\vec$ with a shifted by one version of itself. 

For example consider a network with $M=2$ hiddend layers then we have a vector $\vec k = [2,4,8]$ This can be thought of as a network with 2 input samples, 2 hidden layers and 8 output samples. The forward propigation shapes are as follows: 

Input $\to$ Layer 1 output
$$(1\times 4) = (1\times2)(2\times 4)+(1\times4)$$

Input $\to$ Layer 2 output (output of NN)
$$(1\times 4) = (1\times4)(4\times 8)+(1\times8)$$

So for this example we needed $8+4=12$ biases and $(4)(8)+(2)(4)=40$ weights for a total. If we allow for the input layer to apply initial weights then we get a total of $12+2=14$ biases. 

In gereral is we take $\vec k = [k_0, k_1,k_2,...,k_M]$ a shifted version of $\vec k_s=[k_1,k_2,k_3...,k_{M},0]$ we can find the total biases as $B_{tot}=sum(\vec k)$ if we allow for input biases, or $sum(\vec k_s)$ if there are no input biases. Then the total number of weights in the network becomes  the dot product $W_{tot}=\vec k \cdot \vec k_s$. Finally the total parameters is just $P_{tot}=B_{tot}+W_{tot}$

So looking back at out simple exampl to we would get:
$$\vec k = [2,4,8]$$
$$\vec k_s=[4,8,0]$$
$$P_{tot}=sum(\vec k)+\vec k \cdot \vec k_s$$
$$P_{tot}= 14+(8+32)=54$$

Which we can see matches what we expected. 
