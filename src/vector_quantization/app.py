import numpy as np
import PySimpleGUI as sg
from PIL import Image, ImageDraw
from sklearn.cluster import MiniBatchKMeans
import io
import threading

# Global variables
codebook_size = 256
block_size = 8

def generate_codebook(blocks, codebook_size):
    # Flatten the blocks into vectors
    vectors = np.reshape(blocks, (-1, block_size * block_size * blocks.shape[-1]))

    # Use Mini-Batch K-means clustering to generate the codebook
    kmeans = MiniBatchKMeans(n_clusters=codebook_size, random_state=0, batch_size=codebook_size * 20)
    kmeans.fit(vectors)
    codebook = kmeans.cluster_centers_

    # Reshape the codebook to match the shape of the blocks
    codebook = np.reshape(codebook, (codebook_size, block_size, block_size, blocks.shape[-1]))

    return codebook


def encode_image(blocks, codebook):
    indices = []
    for block in blocks:
        # Calculate the Euclidean distance between the block and each codeword
        distances = np.sum(np.square(block - codebook), axis=(1, 2, 3))

        # Find the index of the closest codeword
        index = np.argmin(distances)

        # Record the index of the assigned codeword
        indices.append(index)

    return indices


def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image)

    return image


def save_image(image, save_path):
    # Convert the image array back to PIL Image object
    image = Image.fromarray(np.uint8(image))

    # Save the image
    image.save(save_path)


def compress_image(image, codebook_size):
    # Preprocess the image into blocks
    blocks = preprocess_image(image)

    # Generate the initial codebook
    codebook = generate_codebook(blocks, codebook_size)

    # Encode the image using vector quantization
    indices = encode_image(blocks, codebook)

    # Decode the indices to reconstruct the compressed image
    reconstructed_image = decode_image(indices, codebook, image.shape[2], image.shape)

    # Remove singleton dimensions and convert to uint8 data type
    compressed_image = np.squeeze(reconstructed_image).astype(np.uint8)

    return compressed_image


def preprocess_image(image):
    # Split the image into blocks
    blocks = []
    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            block = image[i:i+block_size, j:j+block_size, :]
            blocks.append(block)

    # Convert the blocks to a numpy array
    blocks = np.array(blocks)

    return blocks


def decode_image(indices, codebook, num_channels, image_shape):
    reconstructed_blocks = []
    for index in indices:
        # Retrieve the corresponding codeword from the codebook
        codeword = codebook[index]

        # Reshape the codeword into a block shape
        block = np.reshape(codeword, (block_size, block_size, num_channels))

        # Append the reconstructed block
        reconstructed_blocks.append(block)

    # Convert the reconstructed blocks into an array
    reconstructed_image = np.array(reconstructed_blocks)

    # Reshape the reconstructed image to the original shape
    reconstructed_image = np.reshape(reconstructed_image, image_shape)

    return reconstructed_image


# Create the GUI layout
layout = [
    [sg.Text("Image Compression with Vector Quantization", font=("Arial", 16))],
    [sg.Image(key="-ORIGINAL-", size=(300, 300)), sg.Image(key="-COMPRESSED-", size=(300, 300))],
    [sg.Input(key="-FILE-", enable_events=True, visible=False), sg.FileBrowse("Select Image", key="-BROWSE-", size=(15, 1))],
    [sg.Button("Compress", key="-COMPRESS-", size=(15, 1)), sg.Button("Save Compressed Image", key="-SAVE-", size=(20, 1), disabled=True)],
]

# Create the window
window = sg.Window("Image Compression", layout)


def draw_uniform_box(image):
    uniform_box_color = (0, 0, 255)  # Blue color
    uniform_box_thickness = 2

    # Convert the image to PIL Image object
    pil_image = Image.fromarray(image)

    # Create a drawing object
    draw = ImageDraw.Draw(pil_image)

    # Get the image size
    image_width, image_height = pil_image.size

    # Draw a uniform box around the image
    draw.rectangle([(0, 0), (image_width - 1, image_height - 1)], outline=uniform_box_color, width=uniform_box_thickness)

    # Convert the PIL Image object back to numpy array
    image_with_uniform_box = np.array(pil_image)

    return image_with_uniform_box


def compress_image_gui(image):
    # Compress the image
    compressed_image = compress_image(image, codebook_size)

    # Draw a uniform box around the compressed image
    compressed_image_with_box = draw_uniform_box(compressed_image)

    # Convert the compressed image to PIL Image object
    compressed_img_data = Image.fromarray(compressed_image_with_box)
    compressed_bio = io.BytesIO()
    compressed_img_data.save(compressed_bio, format="PNG")

    # Update the GUI with the compressed image
    window["-COMPRESSED-"].update(data=compressed_bio.getvalue())

    # Enable the save button
    window["-SAVE-"].update(disabled=False)

def compress_image_gui(image):
    # Compress the image
    compressed_image = compress_image(image, codebook_size)

    # Convert the compressed image to PIL Image object
    compressed_img_data = Image.fromarray(compressed_image)
    compressed_bio = io.BytesIO()
    compressed_img_data.save(compressed_bio, format="PNG")

    # Update the GUI with the compressed image
    window["-COMPRESSED-"].update(data=compressed_bio.getvalue())

    # Enable the save button
    window["-SAVE-"].update(disabled=False)


while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED:
        break

    if event == "-FILE-":
        # Update the selected image path
        image_path = values["-FILE-"]

        if image_path:
            # Load the original image
            original_image = load_image(image_path)

            # Display the original image
            img_data = Image.fromarray(original_image)
            bio = io.BytesIO()
            img_data.save(bio, format="PNG")
            window["-ORIGINAL-"].update(data=bio.getvalue())

            # Enable the compress button
            window["-COMPRESS-"].update(disabled=False)
            window["-SAVE-"].update(disabled=True)

    if event == "-COMPRESS-":
        # Disable the compress button
        window["-COMPRESS-"].update(disabled=True)

        # Run the image compression in a separate thread
        thread = threading.Thread(target=compress_image_gui, args=(original_image,))
        thread.start()

    if event == "-SAVE-":
        # Specify the save path for the compressed image
        save_path = sg.popup_get_file("Save Compressed Image", save_as=True, file_types=(("PNG Files", "*.png"),))

        if save_path:
            # Compress the image again to ensure the latest compressed version is saved
            compressed_image = compress_image(original_image, codebook_size)

            # Save the compressed image
            save_image(compressed_image, save_path)
            sg.popup("Image saved successfully!")

window.close()
