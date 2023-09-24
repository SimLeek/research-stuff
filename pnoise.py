from displayarray import display
import numpy as np
import random

def lerp(a,b,x):
    "linear interpolation"
    return a + x * (b-a)

def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h,x,y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y

def perlin(x, y, seed=0):
    # permutation table
    np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi, yi = x.astype(int), y.astype(int)
    # internal coordinates
    xf, yf = x - xi, y - yi
    # fade factors
    u, v = fade(xf), fade(yf)
    # noise components
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
    return lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here

i=0
def arr_pnoise(arr):
    global i
    shape = arr.shape[0:2]
    freq = 75
    lin = [np.linspace(0, freq, s, endpoint=False) for s in shape]
    x, y = np.meshgrid(lin[0], lin[1])
    n = perlin(x, y, seed=i)/2 + arr
    n_min = np.min(n)
    n_max = np.max(n)
    n_r = n_max-n_min
    n_out = (n-n_min)/n_r
    i+=1
    return n_out

arr = np.random.normal(0.5, 0.1, (100, 100))
arr = np.zeros((100,100))

def custom_l1_regularization(arr, noise, regularization_strength=0.1):
    # Flatten the noise array
    #flattened_noise = noise.flatten()

    # Sort the flattened noise in descending order
    #sorted_noise = np.sort(np.abs(flattened_noise))[::-1]

    # Calculate the threshold to retain a specific total activation
    #threshold = sorted_noise[int(regularization_strength * len(sorted_noise))]

    # Set values below the threshold to 0
    regularized_noise = np.where(np.abs(arr*.5+noise) >= 1.0-regularization_strength, 1.0, 0.0)

    return regularized_noise

arr2 = np.copy(arr)
with display(arr) as displayer:
    while displayer:
        arr2[:] = arr_pnoise(arr2)
        arr[:] = custom_l1_regularization(arr, arr2)
