# CNN from scratch

The interesting part is in the folder `neural_networks/layers.py`: Code for a convolutional neural network, based on only numpy (no PyTorch or TensorFlow). It is therefore very foundational and illustrates how CNNs work mathematically. 

The CNNs is compatible with color images (3-channel rgb), includes pooling layers (`class Pool2D`) and works with any given (valid) stride.

`neural_networks/activations.py` contains basic activation functions, like ReLu or SoftMax with the appropriate forward / backward implementations calculating the jacobian, etc. as needed for backpropagation.

Many functions make heavy use of slicing, to speed up the training process significantly. See e.g. `Conv2D.forward`:

```python
for x in range(out_rows):
    for y in range(out_cols):
        out[:,x,y,:] = np.apply_over_axes(np.sum, W[None]*X_pad[:,x*s:x*s+kernel_height,y*s:y*s+kernel_width,:][...,None], [1,2,3])[:,0,0,0,:]
```

which is the sliced version of a depth-6 nested for loop -- and thus allows for significant speedup (on my computer, more than 20x speedup for the given training data).

In `losses.py`, `CrossEntropy` is the most important function. To allow for speed-up, we simplified mathematically as much as possible, yielding

```python
loss = -1.0/m *np.trace(np.matmul(Y,np.log(Y_hat.T)))
```
for the forward pass and 
```python
-1/m*(np.divide(Y,Y_hat))
```
for the backward pass.


This is based on a project for CS289 at UC Berkeley. 
