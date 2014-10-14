import convnet as cn
import numpy as np
import sys

pbtxt_file = "/home/fs/ylu/Code/convnet/examples/imagenet/CLS_net_20140801232522.pbtxt"
params_file = "/home/fs/ylu/Code/convnet/examples/imagenet/CLS_net_20140801232522.h5"
means_file = "/home/fs/ylu/Code/convnet/examples/imagenet/pixel_mean.h5"

model = cn.ConvNet(pbtxt_file)  # Load the model architecture.
model.Load(params_file)  # Set the weights and biases.
model.SetNormalizer(means_file, 224)  # Set the mean and std for input normalization.

data = np.random.randn(128, 224* 224* 3)  # 128 images of size 224x224 as a numpy array.
model.Fprop(data)  # Fprop through the model.

              # Returns the state of the requested layer as a numpy array.
last_hidden_layer = model.GetState('hidden7')
output = model.GetState('output')

print output.shape, last_hidden_layer.shape  # (128, 1000) (128, 4096).
