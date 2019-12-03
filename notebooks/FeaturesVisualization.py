#!/usr/bin/env python
# coding: utf-8

# # Feature Visualization based on VGG19 convolutionnal network

# In[1]:


from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.applications import VGG19
from PIL import Image
from tqdm import tqdm

STYLE_PATH = "../data/style"
CONTENT_PATH = "../data/content"

IMAGE_SHAPE = (512, 512, 3)


# ## Import and resize input images

# In[2]:


raw_style = Image.open(f"{STYLE_PATH}/vangogh.png")
raw_content = Image.open(f"{CONTENT_PATH}/ensc.png")


# In[3]:


content_img = raw_content.resize(IMAGE_SHAPE[0:2], Image.ANTIALIAS)
style_img = raw_style.resize(IMAGE_SHAPE[0:2], Image.ANTIALIAS)


# In[4]:


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

# In[5]:


content_img = np.array(content_img)
style_img = np.array(style_img)


# In[6]:


content_img = np.expand_dims(content_img, axis=0)
style_img = np.expand_dims(style_img, axis=0)


# In[7]:


print(f"Content shape : {content_img.shape}")
print(f"Style shape : {style_img.shape}")


# In[8]:


content_img = keras.applications.vgg19.preprocess_input(content_img, mode="tf")
style_img = keras.applications.vgg19.preprocess_input(style_img, mode="tf")


# In[9]:


fig = plt.figure(figsize=(15, 10))

axes = fig.add_subplot(1, 2, 1)
axes.imshow(np.squeeze(style_img))
axes.set_title("Style image")

axes = fig.add_subplot(1, 2, 2)
axes.imshow(np.squeeze(content_img))
axes.set_title("Content image")

plt.show(fig)


# # Features visualization for content

# ## Load model

# In[10]:


vgg_max = VGG19(include_top=False,  weights='imagenet', input_shape=IMAGE_SHAPE)
vgg_max.trainable = False


# In[2]:


def replace_max_by_average_pooling(model):

    input_layer, *other_layers = model.layers
    assert isinstance(input_layer, keras.layers.InputLayer)

    x = input_layer.output
    for layer in other_layers:
        if isinstance(layer, keras.layers.MaxPooling2D):
            layer = keras.layers.AveragePooling2D(
                pool_size=layer.pool_size,
                strides=layer.strides,
                padding=layer.padding,
                data_format=layer.data_format,
                name=layer.name,
            )
        x = layer(x)

    return keras.models.Model(inputs=input_layer.input, outputs=x)


# In[12]:


vgg = replace_max_by_average_pooling(vgg_max)
vgg.summary()


# In[13]:


def get_vgg_layer(model, layer_name: str, model_name: str=None) -> keras.models.Model:
    layer = model.get_layer(layer_name)
    output = layer.get_output_at(1)
    return keras.models.Model(model.layers[0].input, output, name=model_name)


# In[14]:


block1 = get_vgg_layer(vgg, "block1_conv1", "block1")
block2 = get_vgg_layer(vgg, "block2_conv1", "block2")
block3 = get_vgg_layer(vgg, "block3_conv1", "block3")
block4 = get_vgg_layer(vgg, "block4_conv1", "block4")
block5 = get_vgg_layer(vgg, "block5_conv1", "block5")

blocks = [block1, block2, block3, block4, block5]


# In[15]:


block3.summary()


# ## Create loss and gradient function
# 
# We want the latent space representation of the input noise to fit the latent space representation of the input image.

# ### Loss
# Let $c_m$ be the features extracted by the model $M$ from the content image $C$, and $x_m$ the features extracted by that same layer of the input noise $X$ :
# 
# $L(x_m, c_m) = \frac{1}{2} \sum_i {(x_{m, i} - c_{m, i)}2$

# Let $\tilde{X}$ be the reconstructed image based on the features exctracted by $M$ :
# 
# $\tilde{X} = \underset{X}{\mathrm{argmin}} ~~ L(M(X), M(C)) =  \underset{X}{\mathrm{argmin}} ~~ L(x_m, c_m)$

# In[16]:


def get_features_loss(noise_features, features_target):
    return tf.reduce_mean(tf.square(noise_features - features_target))


# In[17]:


def compute_content_loss(model, init_noise, features):
    model_outputs = model(init_noise)
    return get_features_loss(model_outputs, features) 


# In[18]:


def compute_content_grads(model, init_noise, features):
    with tf.GradientTape() as tape: 
        tape.watch(init_noise) # TOUJOURS REGARDER CE QUI VIENT D'AILLEURS 
        loss = compute_content_loss(model, init_noise, features)
    return tape.gradient(loss, init_noise), loss # dL(c, x)/dx


# ## Training function

# In[19]:


def train_content_features(block: tf.keras.models.Model, 
                           img: np.ndarray, iterations=500, 
                           opt=tf.keras.optimizers.Adam(5, decay=1e-3)) -> Tuple[List, np.ndarray]:

    with tf.device("GPU:0"):
        features = block(img)
        init_noise = tf.Variable(tf.random.normal([1, *IMAGE_SHAPE])) #WARNING: toujours mettre les trucs utilisés par le
                                                                      #GradientTape en tf.Variable ("mutable tensor") sinon c'est 
                                                                      #la hez
        min_vals = -1
        max_vals = 1

        history = []
        built_imgs = []
        best_loss = float("inf")


        for i in tqdm(range(iterations), f"Building img for {block.name}"):
            grads, loss = compute_content_grads(block, init_noise, features)
            opt.apply_gradients([(grads, init_noise)])
            clipped = tf.clip_by_value(init_noise, min_vals, max_vals)
            init_noise.assign(clipped) 

            history.append(loss.numpy())

            if loss < best_loss:
                best_loss = loss
                built_imgs.append(np.squeeze(init_noise.numpy().copy()))
                
    return history, built_imgs


# ## Train on noise image

# In[20]:


results = {}
for block in blocks:
    results[block.name] = train_content_features(block, content_img)


# Deprocessing based on :
# 
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py

# In[21]:


def deprocess_image(x):
    y = x.copy()
    y += 1.
    y *= 127.5
    return y.astype("uint8")


# In[22]:


fig = plt.figure(figsize=(30,10))
fig.suptitle("Content reconstruction")
for i, block in enumerate(blocks):
    axes = fig.add_subplot(2, len(blocks), i+1)
    axes.imshow(deprocess_image(results[block.name][1][-1]))
    axes.set_title(block.name)
    
    axes2 = fig.add_subplot(2, len(blocks), i+len(blocks)+1)
    axes2.plot(results[block.name][0])
    axes2.set_title("Loss over iterations")


# In[23]:


fig.savefig("../reports/content-features-visualization.png")


# ## Inversion

# In[33]:


results = {}
for block in blocks:
    results[block.name] = train_content_features(block, style_img)


# In[34]:


fig = plt.figure(figsize=(30,10))
fig.suptitle("Content reconstruction")
for i, block in enumerate(blocks):
    axes = fig.add_subplot(2, len(blocks), i+1)
    axes.imshow(deprocess_image(results[block.name][1][-1]))
    axes.set_title(block.name)
    
    axes2 = fig.add_subplot(2, len(blocks), i+len(blocks)+1)
    axes2.plot(results[block.name][0])
    axes2.set_title("Loss over iterations")


# In[35]:


fig.savefig("../reports/content-features-visualization-forstyle.png")


# # Feature visualization for style

# ## Loss and gradient functions
# 
# We take as latent space for the style the local correlations existing in the features representations of a VGG layer.
# These correlations are extracted by computing the Gram matrices $G^l$ 

# In[24]:


style_blocks = {
    "blocks1": [block1],
    "blocks2": [block1, block2],
    "blocks3": [block1, block2, block3],
    "blocks4": [block1, block2, block3, block4],
    "blocks5": [block1, block2, block3, block4, block5]
}


# In[25]:


def gram_matrix(tensor):
    tensor = tf.reshape(tensor, [-1, int(tensor.shape[-1])])
    filters = int(tf.shape(tensor)[0])
    gram = tf.matmul(tensor, tensor, transpose_a=True)
    return gram / tf.cast(filters, tf.float32)


# In[26]:


def get_style_loss(noise_features, target_features):
    noise_gram = gram_matrix(noise_features)
    target_gram = gram_matrix(target_features)
    return tf.reduce_sum(tf.square(noise_gram - target_gram)) / tf.cast(4, tf.float32)


# In[27]:


def compute_style_loss(models, init_noise, features_list):
    loss = 0.0
    for features, model in zip(features_list, models):
        model_outputs = model(init_noise)
        loss+=get_style_loss(model_outputs, features) 
    return loss / len(models)


# In[28]:


def compute_style_grads(models, init_noise, features_list):
    with tf.GradientTape() as tape: 
        tape.watch(init_noise) # TOUJOURS REGARDER CE QUI VIENT D'AILLEURS
        loss = compute_style_loss(models, init_noise, features_list)
    return tape.gradient(loss, init_noise), loss # dL(c, x)/dx


# In[29]:


def train_style_features(blocks: tf.keras.models.Model, 
                         img: np.ndarray, iterations=500, 
                         opt=tf.keras.optimizers.Adam(5, decay=1e-3)) -> Tuple[List, np.ndarray]:

    with tf.device("GPU:0"):
        features_list = []
        for block in blocks:
            features_list.append(block(img))
        
        init_noise = tf.Variable(tf.random.normal([1, *IMAGE_SHAPE])) #WARNING: toujours mettre les trucs utilisés par le
                                                                      #GradientTape en tf.Variable ("mutable tensor") sinon c'est 
                                                                      #la hez
        min_vals = -1
        max_vals = 1

        history = []
        built_imgs = []
        best_loss = float("inf")
        
        for i in tqdm(range(iterations), f"Building img for {block.name}"):
            grads, loss = compute_style_grads(blocks, init_noise, features_list) # grads is None for some reason, let's go back up a bit      
            opt.apply_gradients([(grads, init_noise)]) # pb is here, arrive dès la première iter: peut-être qu'il peut pas calculer de gradients lorsque y'a qu'une seule valeur ?
                                                        # Nope, ça arriverait pour les autres calculs aussi           
            clipped = tf.clip_by_value(init_noise, min_vals, max_vals)
            init_noise.assign(clipped) 
            history.append(loss.numpy())

            if loss < best_loss:
                best_loss = loss
                built_imgs.append(np.squeeze(init_noise.numpy().copy()))
                
    return history, built_imgs


# In[30]:


results = {}
for name, block in style_blocks.items():
    results[name] = train_style_features(block, style_img)


# In[31]:


fig = plt.figure(figsize=(30,10))
fig.suptitle("Image ")
i=1
for name, block in style_blocks.items():
    axes = fig.add_subplot(2, len(style_blocks), i)
    axes.imshow(deprocess_image(results[name][1][-1]))
    axes.set_title(name)
    
    axes2 = fig.add_subplot(2, len(style_blocks), i+len(style_blocks))
    axes2.plot(results[name][0])
    axes2.set_title("Loss over iterations")
    i+=1


# In[32]:


fig.savefig("../reports/style-features-visualization.png")


# ## Inversion

# In[36]:


results = {}
for name, block in style_blocks.items():
    results[name] = train_style_features(block, content_img)


# In[37]:


fig = plt.figure(figsize=(30,10))
fig.suptitle("Image ")
i=1
for name, block in style_blocks.items():
    axes = fig.add_subplot(2, len(style_blocks), i)
    axes.imshow(deprocess_image(results[name][1][-1]))
    axes.set_title(name)
    
    axes2 = fig.add_subplot(2, len(style_blocks), i+len(style_blocks))
    axes2.plot(results[name][0])
    axes2.set_title("Loss over iterations")
    i+=1


# In[38]:


fig.savefig("../reports/style-features-visualization-forcontent.png")

