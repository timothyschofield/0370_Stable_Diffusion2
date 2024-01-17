"""
High-performance image generation using Stable Diffusion in KerasCV

Exploration of the theme: Unlike most tutorials, where we first explain a topic then show how to implement it, 
with text-to-image generation it is easier to show instead of tell.

https://www.tensorflow.org/tutorials/generative/generate_images_with_stable_diffusion

02 January 2024

KerasCV is a library of modular Computer Vision components that work natively with TensorFlow, JAX, or PyTorch
Stable Diffusion is a latent text-to-image diffusion model from stability.ai https://stability.ai/

Here, we will show how to generate novel images based on a text prompt using 
the KerasCV implementation of stability.ai's text-to-image model, Stable Diffusion.

XLA (Accelerated Linear Algebra) is an open-source compiler for machine learning. 

"mixed precision support": float16 and bfloat16 as well as float32 dtype support

Note: In this guide, the term "numeric stability" refers to how a model's quality is affected by 
the use of a lower-precision dtype instead of a higher precision dtype. An operation is "numerically unstable" 
in float16 or bfloat16 if running it in one of those dtypes causes the model to have worse evaluation accuracy or 
other metrics compared to running the operation in float32.

CUDA version:
cd /usr/local/cuda 
nano version.json - everything is version 12.3.1


"""
# pip install tensorflow keras_cv --upgrade
# pip install tensorflow[and-cuda]
# pip install matplotlib
import time
import keras_cv
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# We need to make sure we have access to a GPU.
print("TensorFlow has access to the following devices:")
for device in tf.config.list_physical_devices():
    print(f" - {device}")

keras.mixed_precision.set_global_policy("mixed_float16")

start = time.time()
# First, we construct a model:
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512, jit_compile=True)

print("Compute dtype:", model.diffusion_model.compute_dtype)    
print("Variable dtype:",model.diffusion_model.variable_dtype,) 
"""
Mixed precision" consists of performing computation using float16 precision, while storing weights in the float32 format. 
This is done to take advantage of the fact that float16 operations are backed by significantly faster kernels than their float32 counterparts on modern NVIDIA GPUs.
"""
# Conclusion - JIT not working
# Standard: 152.64 seconds - with JIT: 153.64 seconds
# Compute dtype: float32
# Variable dtype: float32
# ----------------------------------
# mixed_float16: 126.86 seconds - with JIT: 123.75 seconds
# Compute dtype: float16
# Variable dtype: float32
# ----------------------------------

images = model.text_to_image("photograph of an astronaut riding a horse", batch_size=3)
# The possibilities are literally endless (or at least extend to the boundaries of Stable Diffusion's latent manifold).
# https://en.wikipedia.org/wiki/Latent_space

# This is a statistical method for visualizing high-dimensional data by giving each datapoint a location in a two or three-dimensional map.
# t-distributed stochastic neighbor embedding
# https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding

def plot_images(images):
    
    plt.figure(figsize=(12, 10))
    
    for i in range(len(images)):
        
           # plt.subplot(to y=1 , to x=3, (1, 2, 3))
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off") # off, square, equal, scaled?

end = time.time()
print(f"Time {(end - start):.2f} seconds")

plot_images(images) # A few minutes
plt.show()


###############################################################################
# Wait, how does this even work?
###############################################################################
"""
Unlike what you might expect at this point, StableDiffusion doesn't actually run on magic. 
It's a kind of "latent diffusion model". Let's dig into what that means.

You may be familiar with the idea of super-resolution: it's possible to train a deep learning model to denoise an input image 
-- and thereby turn it into a higher-resolution version. 
The deep learning model doesn't do this by magically recovering the information that's missing from the noisy, low-resolution input 
-- rather, the model uses its training data distribution to hallucinate the visual details that would be most likely given the input. 
To learn more about super-resolution, you can check out the following Keras.io tutorials:

Image Super-Resolution using an Efficient Sub-Pixel CNN
https://keras.io/examples/vision/super_resolution_sub_pixel/

Enhanced Deep Residual Networks for single-image super-resolution
https://keras.io/examples/vision/edsr/

When you push this idea to the limit, you may start asking -- what if we just run such a model on pure noise? 
The model would then "denoise the noise" and start hallucinating a brand new image.
By repeating the process multiple times, 
you can get turn a small patch of noise into an increasingly clear and high-resolution artificial picture.

This is the key idea of latent diffusion, proposed in [High-Resolution Image Synthesis with Latent Diffusion Models] in 2020. 
https://arxiv.org/abs/2112.10752
To understand diffusion in depth, you can check the Keras.io tutorial [Denoising Diffusion Implicit Models].
https://keras.io/examples/generative/ddim/

Now, to go from latent diffusion to a text-to-image system, 
you still need to add one key feature: the ability to control the generated visual contents via prompt keywords. 
This is done via "conditioning", a classic deep learning technique which consists of concatenating to the noise patch a vector that represents a bit of text, 
then training the model on a dataset of {image: caption} pairs.

This gives rise to the Stable Diffusion architecture. Stable Diffusion consists of three parts:
1) A text encoder, which turns your prompt into a latent vector.
2) A diffusion model, which repeatedly "denoises" a 64x64 latent image patch.
3) A decoder, which turns the final 64x64 latent patch into a higher-resolution 512x512 image.

First, your text prompt gets projected into a latent vector space by the text encoder, 
    which is simply a pretrained, frozen language model. 
Then that prompt vector is concatenated to a randomly generated noise patch, 
    which is repeatedly "denoised" by the diffusion model over a series of "steps" (the more steps you run the clearer and nicer your image will be 
    -- the default value is 50 steps).

Diagram here (Ctrl click)
https://i.imgur.com/2uC8rYJ.png

All-in-all, it's a pretty simple system -- the Keras implementation fits in four files that represent less than 500 lines of code in total:

    https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/stable_diffusion
    text_encoder.py: 87 LOC
    diffusion_model.py: 181 LOC
    decoder.py: 86 LOC
    stable_diffusion.py: 106 LOC

But this relatively simple system starts looking like magic once you train on billions of pictures and their captions. 
As Feynman said about the universe: "It's not complicated, it's just a lot of it!"
"""
###############################################################################
# The Perks of KerasCV
###############################################################################































