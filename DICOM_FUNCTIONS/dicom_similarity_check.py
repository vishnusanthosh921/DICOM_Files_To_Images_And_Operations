import os
import pydicom
import base64
from flask import Flask, render_template
from skimage.metrics import structural_similarity as ssim
import cv2

app = Flask(__name__)

# Function to convert a DICOM file into an image
def convert_dicom_to_image(dicom_path):
    dicom_data = pydicom.dcmread(dicom_path)
    pixel_array = dicom_data.pixel_array
    return pixel_array

# Function to calculate similarity between two images using SSIM
def calculate_ssim(image1, image2):
    return ssim(image1, image2, multichannel=True)

# Function to traverse the folder structure, convert DICOM files into images,
# and calculate their similarities with the uploaded DICOM image
def find_similar_images(upload_image, main_folder_path):
    def traverse_folder(folder_path):
        similar_images = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.dic'):
                    dicom_path = os.path.join(root, file)
                    dicom_image = convert_dicom_to_image(dicom_path)
                    if upload_image.shape != dicom_image.shape:
                        continue  # Skip if dimensions are different
                    similarity = calculate_ssim(upload_image, dicom_image)
                    similar_images.append((dicom_path, similarity))
        return similar_images

    # Traverse the main folder and its subfolders
    similar_images = traverse_folder(main_folder_path)
    return similar_images



@app.route('/')
def index():
    # Specify the paths
    upload_image_path = r"Pacs_Test_Folder/dic_folder/1.2.840.113619.2.5.4230407869.26071.1715913201.453.dic"
    main_folder_path = r"Pacs_Test_Folder/dic_folder"

    # Load the upload image
    upload_image = convert_dicom_to_image(upload_image_path)

    # Find similar images
    similar_images = find_similar_images(upload_image, main_folder_path)

    # Define the threshold for high similarity
    threshold = 0.9  # Adjust as needed

    # Filter similar images based on the threshold
    high_similarity_images = [(path, similarity) for path, similarity in similar_images if similarity > threshold]

    # Sort similar images based on similarity score in descending order
    high_similarity_images.sort(key=lambda x: x[1], reverse=True)

    # Encode the upload image as base64
    _, upload_image_buffer = cv2.imencode('.png', upload_image)
    upload_image_base64 = base64.b64encode(upload_image_buffer).decode('utf-8')

    # Encode similar images as base64 and pass data to HTML template
    similar_images_base64 = []
    for dicom_path, similarity in high_similarity_images[:4]:
        dicom_image = convert_dicom_to_image(dicom_path)
        _, dicom_image_buffer = cv2.imencode('.png', dicom_image)
        dicom_image_base64 = base64.b64encode(dicom_image_buffer).decode('utf-8')
        similar_images_base64.append((dicom_image_base64, similarity))

    return render_template('treeindex.html', input_image=upload_image_base64, similar_images=similar_images_base64)

if __name__ == '__main__':
    app.run(debug=True)
