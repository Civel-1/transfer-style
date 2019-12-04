#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install ..')


# In[14]:


from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.applications import VGG19
from PIL import Image
from tqdm import tqdm

import transferutils as tu

STYLE_PATH = "../data/style"
CONTENT_PATH = "../data/content"

IMAGE_SHAPE = (512, 512, 3)


# In[2]:


style_img = tu.import_transform(f"{STYLE_PATH}/vangogh.png")
content_img = tu.import_transform(f"{CONTENT_PATH}/ensc.png")


# In[3]:


fig = plt.figure(figsize=(15, 10))

axes = fig.add_subplot(1, 2, 1)
axes.imshow(np.squeeze(style_img))
axes.set_title("Style image")

axes = fig.add_subplot(1, 2, 2)
axes.imshow(np.squeeze(content_img))
axes.set_title("Content image")

plt.show(fig)


# In[4]:


vgg_max = VGG19(include_top=False,  weights='imagenet', input_shape=IMAGE_SHAPE)
vgg_max.trainable = False
vgg = tu.replace_max_by_average_pooling(vgg_max)
vgg.summary()


# In[11]:


content_blocks = [tu.get_vgg_layer(vgg, f"block{i}_conv1", f"block{i}") for i in range(1, 6)]
style_blocks = [[content_blocks[j] for j in range(i)] for i in range(1, len(content_blocks) + 1)]


# In[ ]:


def compute_loss(style_layers: Sequence[keras.models.Model],
                 content_layer: keras.model.Model,
                 init_noise: tf.Variable,
                 content_features: tf.Tensor,
                 style_features: tf.Tensor) -> tf.Tensor:
    

