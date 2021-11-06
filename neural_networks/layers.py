"""
Author: Tassilo Schwarz, Sophia Sanborn, Sagnik Bhattacharya
Institution: UC Berkeley
Date: Spring 2021
Course: CS189/289A
Website: github.com/sophiaas, github.com/sagnibak
"""

import numpy as np
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
# from neural_networks.utils import pad2d
from collections import OrderedDict

from typing import Callable, List, Literal, Tuple, Union


class Layer(ABC):
    """Abstract class defining the `Layer` interface."""

    def __init__(self):
        self.activation = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(
        self, param_name: str, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]


def initialize_layer(
    name: str,
    activation: str = None,
    weight_init: str = None,
    n_out: int = None,
    kernel_shape: Tuple[int, int] = None,
    stride: int = None,
    pad: int = None,
    mode: str = None,
    keep_dim: str = "first",
) -> Layer:
    """Factory function for layers."""
    if name == "fully_connected":
        return FullyConnected(
            n_out=n_out, activation=activation, weight_init=weight_init,
        )

    elif name == "conv2d":
        return Conv2D(
            n_out=n_out,
            activation=activation,
            kernel_shape=kernel_shape,
            stride=stride,
            pad=pad,
            weight_init=weight_init,
        )

    elif name == "pool2d":
        return Pool2D(kernel_shape=kernel_shape, mode=mode, stride=stride, pad=pad)

    elif name == "flatten":
        return Flatten(keep_dim=keep_dim)

    else:
        raise NotImplementedError("Layer type {} is not implemented".format(name))


class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        W = self.init_weights((self.n_in,)+(self.n_out,))
        b = np.zeros((1,self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache = OrderedDict({"X":[],"Z":[]})  # cache for backprop
        self.gradients = OrderedDict({"W": np.zeros_like(W),"b":np.zeros_like(b)})  # parameter gradients initialized to zero # np.random.randn(*W.shape)
                                           # MUST HAVE THE SAME KEYS AS `self.parameters`


    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        # Weight acceessed by self.parameters[W]
        
        # perform an affine transformation and activation
        one_vec = np.ones(X.shape[0]).reshape(-1,1)
        Z = X@self.parameters["W"]+ one_vec@self.parameters["b"]
        out = self.activation.forward(Z)
        self.cache["X"] = X
        self.cache["Z"] = Z
        
        # store information necessary for backprop in `self.cache`


        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  derivative of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        
        # unpack the cache
        X = self.cache["X"]
        Z = self.cache["Z"]
        W=self.parameters["W"]
        
        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer
        dLdZ = self.activation.backward(Z,dLdY)

        # for chain rule:
        dZdX = W.T
        dZdW = X.T # strictly this is before the matrix.....
        
        dW = np.matmul(dZdW,dLdZ)
        dX = np.matmul(dLdZ, dZdX)
        db = np.sum(dLdZ, axis = 0)
        # print(self.parameters["b"].shape)
        # print(db.shape)

        self.gradients["W"] = dW
        self.gradients["b"] = db



        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.


        return dX


class Conv2D(Layer):
    """Convolutional layer for inputs with 2 spatial dimensions."""

    def __init__(
        self,
        n_out: int,
        kernel_shape: Tuple[int, int],
        activation: str,
        stride: int = 1,
        pad: str = "same",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int, int, int]) -> None:
        """Initialize all layer parameters and determine padding."""
        self.n_in = X_shape[3]

        W_shape = self.kernel_shape + (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache = OrderedDict({"Z": [], "X": []})
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)})

        if self.pad == "same":
            self.pad = ((W_shape[0] - 1) // 2, (W_shape[1] - 1) // 2)
        elif self.pad == "valid":
            self.pad = (0, 0)
        elif isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        # caching any values required for backprop

        if self.n_in is None:
            self._init_parameters(X.shape)

        npad=((0,0),(self.pad[0],self.pad[0]),(self.pad[1],self.pad[1]),(0,0))
        X_pad = np.pad(X,pad_width=npad,mode='constant',constant_values=0.0)

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        kernel_half_height = kernel_height//2
        kernel_half_width= kernel_width//2
        assert kernel_height%2 == 1
        assert kernel_width%2 == 1

        
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)

       

        # assert in_rows%self.stride == 0
        # assert in_cols%self.stride == 0

        s=self.stride

        out_rows = int((in_rows + 2*self.pad[0] - kernel_height) / self.stride + 1)

        out_cols = int((in_cols + 2*self.pad[1] - kernel_width) / self.stride + 1)
        out=np.zeros((n_examples, out_rows,out_cols,out_channels))


        for x in range(out_rows):
            for y in range(out_cols):
                out[:,x,y,:] = np.apply_over_axes(np.sum, W[None]*X_pad[:,x*s:x*s+kernel_height,y*s:y*s+kernel_width,:][...,None], [1,2,3])[:,0,0,0,:]
        
        out+=b

        self.cache["X_pad"] = X_pad
        self.cache["X"] = X
        self.cache["Z"] = out

        out=self.activation(out)


        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        Parameters
        ----------
        dLdY  derivative of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
        # unpack the cache
        X = self.cache["X"]
        X_pad = self.cache["X_pad"]
        Z = self.cache["Z"]
        W=self.parameters["W"]  
        s=self.stride      
        
        n_samples,x_height,x_width,chans = X.shape
        kernel_height,kernel_width,neurons_before,neurons_after = W.shape

        assert chans == neurons_before

        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer
        dLdZ = self.activation.backward(Z,dLdY)

        # Pad X
        npad=((0,0),(self.pad[0],self.pad[0]),(self.pad[1],self.pad[1]),(0,0))
        X_pad = np.pad(X,pad_width=npad,mode='constant',constant_values=0.0)

        # Create dLdX padded, which has same size as X_pad
        dLdX_pad = np.zeros(X_pad.shape)
        W_flip = np.flip(W,axis=[0,1])

        
        for r1 in range(X_pad.shape[1]):
            for r2 in range(X_pad.shape[2]):
                for c in range(chans):
                    accumul = np.zeros(n_samples)
                    for d1 in range( dLdY.shape[1]):
                        for d2 in range( dLdY.shape[2]):
                            if (0 <= r1 - self.stride * d1 < kernel_height) and (0 <= r2 - self.stride * d2 < kernel_width):
                                b = W[r1 - self.stride * d1, r2 - self.stride * d2 , c, :]
                                accumul += np.matmul(dLdZ[:, d1, d2, :], b)
                    dLdX_pad[:,r1,r2,c] = accumul
        
        # slice
        dLdX = dLdX_pad[:,self.pad[0]:-self.pad[0],self.pad[1]:-self.pad[1],:]

        dLdW=np.zeros(W.shape) 

        dLdW=np.zeros(W.shape)
        for i in range(kernel_height):
            for j in range(kernel_width):
                for c in range(neurons_before):
                    for n in range(neurons_after):
                        accumul = np.zeros(X.shape[0])
                        for d1 in range( dLdY.shape[1]):
                            for d2 in range( dLdY.shape[2]):
                                if(d1*s+i<X.shape[1] and d2*s+j<X.shape[2]):
                                    accumul += dLdZ[:,d1,d2,n]*X[:,d1*s+i,d2*s+j,c]
                        dLdW[i,j,c,n] = np.sum(accumul)


        dLdB=np.apply_over_axes(np.sum, dLdZ, axes=[0,1,2])
        

        self.gradients["W"] = dLdW
        self.gradients["b"] = dLdB


        return dLdX


class Pool2D(Layer):
    """Pooling layer, implements max and average pooling."""

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        mode: str = "max",
        stride: int = 1,
        pad: Union[int, Literal["same"], Literal["valid"]] = 0,
    ) -> None:

        if type(kernel_shape) == int:
            kernel_shape = (kernel_shape, kernel_shape)

        self.kernel_shape = kernel_shape
        self.stride = stride

        if pad == "same":
            self.pad = ((kernel_shape[0] - 1) // 2, (kernel_shape[1] - 1) // 2)
        elif pad == "valid":
            self.pad = (0, 0)
        elif isinstance(pad, int):
            self.pad = (pad, pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

        self.mode = mode

        if mode == "max":
            self.pool_fn = np.max
            self.arg_pool_fn = np.argmax
        elif mode == "average":
            self.pool_fn = np.mean

        self.cache = {
            "out_rows": [],
            "out_cols": [],
            "X_pad": [],
            "p": [],
            "pool_shape": [],
        }
        self.parameters = {}
        self.gradients = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: use the pooling function to aggregate local information
        in the input. This layer typically reduces the spatial dimensionality of
        the input while keeping the number of feature maps the same.

        As with all other layers, please make sure to cache the appropriate
        information for the backward pass.

        Parameters
        ----------
        X  input array of shape (batch_size, in_rows, in_cols, channels)

        Returns
        -------
        pooled array of shape (batch_size, out_rows, out_cols, channels)
        """
        

        # implement the forward pass
         # get dimensions
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_height,kernel_width = self.kernel_shape[0], self.kernel_shape[1]
        s = self.stride

        # generate the pooled tensor for the forward pass
        out_rows = int((in_rows + 2*self.pad[0] - kernel_height) / self.stride + 1)

        out_cols = int((in_cols + 2*self.pad[1] - kernel_width) / self.stride + 1)

        npad=((0,0),(self.pad[0],self.pad[0]),(self.pad[1],self.pad[1]),(0,0))
        X_pad = np.pad(X,pad_width=npad,mode='constant',constant_values=0.0)

        X_pool = np.zeros((n_examples, out_rows, out_cols, in_channels))


        # implement pooling forward pass
        for r1 in range(out_rows):
            for r2 in range(out_cols):
                window = X_pad[:, r1*s:r1*s+kernel_height, r2*s:r2*s+kernel_width,:]
                X_pool[:, r1, r2, :] = self.pool_fn(window, axis=(1,2)) # [:,0,0,:] #  np.apply_over_axes(self.pool_fn, window, [1, 2])[:, 0, 0, :]
        
        
        self.cache["X"] = X
        self.cache["X_pad"] = X_pad
        self.cache["Z"] = X_pool

        return X_pool

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for pooling layer.

        Parameters
        ----------
        dLdY  gradient of loss with respect to the output of this layer
              shape (batch_size, out_rows, out_cols, channels)

        Returns
        -------
        gradient of loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, channels)
        """

    

        X = self.cache["X"]
        s = self.stride
        X_pad = self.cache["X_pad"] 

        # dimensions
        n_samples,x_height,x_width,chans = X_pad.shape
        kernel_height,kernel_width = self.kernel_shape
        

        # instantiate output
        dLdX_pad = np.zeros_like(X_pad)

        
        for r1 in range(dLdY.shape[1]):
            for r2 in range(dLdY.shape[2]):
                # limits (low, high)
                height_low = r1 * s
                height_high = height_low + kernel_height
                widht_low = r2 * s
                width_higih = widht_low + kernel_width
                x = X_pad[:, height_low:height_high, widht_low:width_higih, :]
                # case max
                if self.mode == "max":
                    n, h, w, c = x.shape
                    x = x.reshape(n, h * w, c)
                    idx = np.argmax(x, axis=1)
                    n_idx, c_idx = np.indices((n, c))

                    # masking
                    bit_mask = np.zeros((n,h,w,c))
                    bit_mask.reshape(n, h * w, c)[n_idx, idx, c_idx] = 1
                    dLdX_pad[:, height_low:height_high, widht_low:width_higih, :] += dLdY[:, r1:r1 + 1, r2:r2 + 1, :] * bit_mask

                # avg
                else:
                    wnd = dLdY[:, r1:r1+1, r2:r2+1, :] / (kernel_height*kernel_width) *np.ones([n_samples,kernel_height,kernel_width,chans])
                    # add current window
                    dLdX_pad[:, height_low:height_high, widht_low:width_higih, :] += wnd

        # remove padding
        dLdX = dLdX_pad[:,self.pad[0]:dLdX_pad.shape[1]-self.pad[0],self.pad[1]:dLdX_pad.shape[2]-self.pad[1],:]

        return dLdX

class Flatten(Layer):
    """Flatten the input array."""

    def __init__(self, keep_dim: str = "first") -> None:
        super().__init__()

        self.keep_dim = keep_dim
        self._init_params()

    def _init_params(self):
        self.X = []
        self.gradients = {}
        self.parameters = {}
        self.cache = {"in_dims": []}

    def forward(self, X: np.ndarray, retain_derived: bool = True) -> np.ndarray:
        self.cache["in_dims"] = X.shape

        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)

        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        in_dims = self.cache["in_dims"]
        dX = dLdY.reshape(in_dims)
        return dX
