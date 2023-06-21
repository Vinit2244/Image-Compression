import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image = Image.open("grayscale.jpeg").convert("L")
initial_size = image.size  # Store the initial size of the image
desired_size = (512, 512)
image = image.resize(desired_size, Image.ANTIALIAS)
image_array = np.array(image)
H = np.array([[0.125, 0.125, 0.25, 0, 0.5, 0, 0, 0],
              [0.125, 0.125, 0.25, 0, -0.5, 0, 0, 0],
              [0.125, 0.125, -0.25, 0, 0, 0.5, 0, 0],
              [0.125, 0.125, -0.25, 0, 0, -0.5, 0, 0],
              [0.125, -0.125, 0, 0.25, 0, 0, 0.5, 0],
              [0.125, -0.125, 0, 0.25, 0, 0, -0.5, 0],
              [0.125, -0.125, 0, 0.25, 0.5, 0, 0, 0.5],
              [0.125, -0.125, 0, -0.25, 0, 0, 0, -0.5]])
H_transpose = H.T
H_transpose_inverse = np.linalg.inv(H_transpose)
H_inverse = np.linalg.inv(H)
Threshold_value = float(input("Enter the Threshold Value: "))
block_size = 8

# Compression
compressed_image_array = np.copy(image_array)
for i in range(0, image_array.shape[0], block_size):
    for j in range(0, image_array.shape[1], block_size):
        A = image_array[i:i+block_size, j:j+block_size]
        B_Matrix = np.dot(H_transpose, A)
        B = np.dot(B_Matrix, H)
        for k in range(8):
            for l in range(8):
                if np.abs(B[k][l]) <= Threshold_value:
                    B[k][l] = 0
        final = np.dot(H_transpose_inverse, B)
        FinalA = np.dot(final, H_inverse)
        compressed_image_array[i:i+block_size, j:j+block_size] = FinalA

compressed_image = Image.fromarray(compressed_image_array)
reconstructed_image = compressed_image.resize(initial_size, Image.ANTIALIAS)
# Reverting to original size
reconstructed_image.save("RECONSTRUCTED.jpeg")
