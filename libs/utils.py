import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import resize
from scipy.misc import imsave

# Crops an image to square.
# Determines the longer side and makes it equal to the shorter side
# by taking away equal amounts from either end of the longer side.
def crop_to_square(img):
    # img height is greater than the width
    if img.shape[0] > img.shape[1]:
        diff = img.shape[0] - img.shape[1]
        if diff % 2 == 0:
            crop = img[diff // 2:-diff // 2, :]
        else:
            crop = img[max(0, diff // 2 + 1):min(-diff // 2, -1,), :]
    # image width is greater than height
    elif img.shape[1] > img.shape[0]:
        diff = img.shape[1] - img.shape[0]
        if diff % 2 == 0:
            crop = img[:, diff // 2:-diff // 2]
        else:
            crop = img[:, max(0, diff // 2 + 1):min(-diff // 2, -1)]
    else:
        crop = img
    return crop

# Crops the image, keeping the same aspect ratio.
# Takes in a param between 0 and 1 that specifies the crop percentage 
def crop_keep_aspect(img, crop_pct):
    if crop_pct <= 0 or crop_pct >= 1:
        return img
    r = int(img.shape[0] * crop_pct) // 2
    c = int(img.shape[1] * crop_pct) // 2
    return img[r:-r, c:-c]

# Preprocess all the images in the specified directory.
def preprocess_images(imgdir):
    imgs = []
    files = os.listdir(imgdir)
    files = [os.path.join(imgdir, f) for f in files if '.jpg' in f]
    for f in files:
        img = plt.imread(f)
        # crop to square
        sq = crop_to_square(img)
        # crop a certain percent of the image
        crop = crop_keep_aspect(sq, 0.2)
        rsz = resize(crop, (100, 100))
        imgs.append(rsz)
    return imgs

# create a montage - copied from https://github.com/pkmital/CADL/blob/master/session-1/libs/utils.py
def montage(images, saveto='montage.png'):
    """Draw all images as a montage separated by 1 pixel borders.
    Also saves the file to the destination specified by `saveto`.
    Parameters
    ----------
    images : numpy.ndarray
        Input array to create montage of.  Array should be:
        batch x height x width x channels.
    saveto : str
        Location to save the resulting montage image.
    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    elif len(images.shape) == 4 and images.shape[3] == 1:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 1)) * 0.5
    elif len(images.shape) == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    else:
        raise ValueError('Could not parse image shape of {}'.format(
            images.shape))
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    imsave(arr=np.squeeze(m), name=saveto)
    return m

# Return a tensorflow op for a 2d gaussian kernel
def gauss_2d(mean, sigma, kersize):
    x = tf.linspace(-3.0, 3.0, kersize)
    z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
                            (2.0 * tf.pow(sigma, 2.0)))) *
         (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))

    z_2d = tf.matmul(
        tf.reshape(z, (kersize, 1)),
        tf.reshape(z, (1, kersize))
    )
    return z_2d

# Return a tensorflow op for a 2d gabor kernel
def gabor_2d(mean, sigma, kersize):
    x = tf.linspace(-3.0, 3.0, kersize)
    ys = tf.reshape(tf.sin(x), (kersize, 1))
    ones = tf.ones((1, kersize))
    gabor = tf.multiply(gauss_2d(mean, sigma, kersize), tf.matmul(ys, ones))
    return gabor
