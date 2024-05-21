from flask import Flask, render_template
import os
import cv2
import numpy as np
import pydicom
from PIL import Image
import base64

app = Flask(__name__)

def convert_dicom_to_image(dicom_path):
    # Read the DICOM file
    dicom_data = pydicom.dcmread(dicom_path)

    # Get pixel data as numpy array
    pixel_array = dicom_data.pixel_array

    # Convert numpy array to PIL image
    image = Image.fromarray(pixel_array)

    # Convert to RGB if image has only one channel
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

def calculate_image_similarity(image1, image2):
    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    keypoints1, descriptors1 = orb.detectAndCompute(gray_image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_image2, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Compute similarity score
    similarity_score = len(matches)

    return similarity_score

def find_similar_images(image_path, dicom_folder, threshold=300):
    image = cv2.imread(image_path)
    similar_images = []
    
    dicom_files = [file for file in os.listdir(dicom_folder) if file.endswith('.dic')]
    for dicom_file in dicom_files:
        dicom_path = os.path.join(dicom_folder, dicom_file)
        dicom_image = convert_dicom_to_image(dicom_path)

        similarity = calculate_image_similarity(image, np.array(dicom_image))
        print(f"Similarity between {image_path} and {dicom_file}: {similarity}")
        
        # If similarity is above the threshold, add the DICOM image and its filename to the list
        if similarity > threshold:
            similar_images.append((dicom_image, dicom_file))

    return similar_images


@app.route('/')
def index():
    image_path = r"Dicom_Conversion\output.png"
    dicom_folder = r"Dicom_Conversion\dic_file_folder"
    similar_images_with_filenames = find_similar_images(image_path, dicom_folder, threshold=300)
    
    input_image = cv2.imread(image_path)
    _, buffer = cv2.imencode('.png', input_image)
    input_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    similar_image_data = []
    if similar_images_with_filenames:
        for dicom_image, filename in similar_images_with_filenames:
            _, buffer = cv2.imencode('.png', np.array(dicom_image))
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            similar_image_data.append((image_base64, filename))  # Append both the base64 data and the filename
    else:
        return render_template('index.html', input_image=input_image_base64, similar_images=None)
    
    return render_template('index.html', input_image=input_image_base64, similar_images=similar_image_data)



if __name__ == '__main__':
    app.run(debug=True)
