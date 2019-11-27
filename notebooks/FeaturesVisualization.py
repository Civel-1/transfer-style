#!/usr/bin/env python
# coding: utf-8

# # Feature Visualization based on VGG19 convolutionnal network

# In[58]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.applications import VGG19
from PIL import Image

STYLE_PATH = "../data/style"
CONTENT_PATH = "../data/content"

IMAGE_SHAPE = (512, 512, 3)


# ## Import and resize input images

# In[62]:


raw_style = Image.open(f"{STYLE_PATH}/vangogh.png")
raw_content = Image.open(f"{CONTENT_PATH}/ensc.png")


# In[63]:


content_img = raw_content.resize(IMAGE_SHAPE[0:2], Image.ANTIALIAS)
style_img = raw_style.resize(IMAGE_SHAPE[0:2], Image.ANTIALIAS)


# In[64]:


fig = plt.figure(figsize=(15, 10))

axes = fig.add_subplot(1, 2, 1)
axes.imshow(style_img)
axes.set_title("Style image")

axes = fig.add_subplot(1, 2, 2)
axes.imshow(content_img)
axes.set_title("Content image")

plt.show(fig)


# ## Preprocess images
# 
# Images are preprocessed to respect the training input policy of VGG19.

# In[65]:


content_img = np.array(content_img)
style_img = np.array(style_img)


# In[66]:


content_img = np.expand_dims(content_img, axis=0)
style_img = np.expand_dims(style_img, axis=0)


# In[71]:


print(f"Content shape : {content_img.shape}")
print(f"Style shape : {style_img.shape}")


# In[ ]:





# ## Load model

# In[41]:


vgg = VGG19(include_top=False,  weights='imagenet', input_shape=IMAGE_SHAPE)
vgg.trainable = False


# In[42]:


vgg.summary()


# In[48]:


def get_vgg_layer(vgg, layer_name: str, model_name: str=None) -> keras.models.Model:
    output = vgg.get_layer(layer_name).output
    return keras.models.Model(vgg.input, output, name=model_name)


# In[53]:


block1 = get_vgg_layer(vgg, "block1_conv2", "block1")
block2 = get_vgg_layer(vgg, "block2_conv2", "block2")


# In[ ]:


def build_feature_model(block: keras.models.Model):
    

