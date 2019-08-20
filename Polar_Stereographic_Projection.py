import cv2
import numpy as np
import time
import random
import matplotlib.pyplot as plt

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def bordercorrect(r, s, k, center, R, bordertype='fit'):
    if k < 0:
        if bordertype == 'fit':
            x = r[0] / s[0]
        elif bordertype == 'crop':
            x = 1 / (1 + k * (min(center) / R) ** 2)
    elif k > 0:
        if bordertype == 'fit':
            x = 1 / (1 + k * (min(center) / R) ** 2)
        elif bordertype == 'crop':
            x = r[0] / s[0]

    return x

def distortfun(r, k, dt):
    if dt == 1:
        s = r * (1 / (1 + k * r))
    elif dt == 2:
        s = r * (1 / (1 + k * (r ** 2)))
    elif dt == 3:
        s = r * (1 + k * r)
    elif dt == 4:
        s = r * (1 + k * (r ** 2))
    elif dt == 5:
        s = r * (1 - k * (r ** 2))
    return s

def im_dist_correct(I, k, dt=4):
    # Determine the size of the image to be distorted
    M, N, _ = I.shape
    center = [round(N/2), round(M/2)]

    # Creates N x M (#pixels) x-y points
    xi, yi = np.meshgrid(range(N), range(M))

    # Creates converst the mesh into a colum vector of coordiantes relative to the center
    xt = np.matrix.flatten(xi) - center[0]
    yt = np.matrix.flatten(yi) - center[1]

    # Convert the x-y coordinates to polar coordinates
    r, theta = cart2pol(xt, yt)

    # Calculate the maximum vector (image center to image corner) to be used for normalization
    R = np.sqrt(center[0]**2 + center[1]**2)

    # Normalize the polar coordinate r to range between 0 and 1
    r = r / R

    # Aply the r-based transformation
    # s = np.multiply(r, (1 + np.multiply(k, np.square(r))));
    s = distortfun(r, k, dt)

    # un-normalize s
    s2 = s * R

    # Find a scaling parameter based on selected border type
    brcor = bordercorrect(r, s, k, center, R, bordertype='crop')
    # brcor = bordercorrect(r, s, k, center, R)
    s2 = s2 * brcor

    # Convert back to cartesian coordinates
    ut, vt = pol2cart(s2, theta)

    u = np.float32(np.reshape(ut, xi.shape) + center[0])
    v = np.float32(np.reshape(vt, yi.shape) + center[1])

    # Remap
    return cv2.remap(I, u, v, interpolation=cv2.INTER_LINEAR)

# image_path = '112_5482818905_2d9ff70e31_o.jpg'
image_path = '97_IMG_20180430_183433_input.jpg'
# image_path = '97_TPE_2018_0314_IMG_20180101_114412_P000.jpg'

# image_path = 'house.jpg'
# undistorted_image_path = '112_5482818905_2d9ff70e31_o(2).jpg'

im = cv2.imread(image_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
h, w, _ = im.shape
# center = [round(w/2), round(h/2)]
# im = cv2.resize(im, (480, 320))

t0 = time.time()

im_undistorted = im_dist_correct(im, 0.2, dt=3)

# im = cv2.resize(im, (480, 320))
# im_undistorted = cv2.resize(im_undistorted, (480, 320))

# out_im = np.zeros([320, 960, 3], dtype=np.uint8)
out_im = np.zeros([h, w*2, 3], dtype=np.uint8)

out_im[:, :w, :] = im
out_im[:, w:, :] = im_undistorted

plt.imshow(out_im)
plt.show()

print('> proc time: %.5f sec' % (time.time() - t0))