from flask import Flask, render_template
import pydicom
import cv2
import numpy as np
import base64

app = Flask(__name__)

def load_dicom_image(file_path):
    dicom = pydicom.dcmread(file_path)
    image = dicom.pixel_array
    return image

def compare_images_with_reference(image, reference_image):
    diff = np.abs(image - reference_image)
    return diff

def mark_high_differences(image, differences, num_marks):
    # Convert image to 8-bit depth if needed
    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image)

    marked_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Threshold the differences to create a binary image
    _, thresh = cv2.threshold(differences, 100, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and select the largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_marks]

    # Draw rectangles around the largest contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return marked_image

@app.route('/')
def index():
    # dicom_file1 = r"DICOM_FUNCTIONS\Pacs_Test\1.2.840.113619.2.25.4.2147483647.1715913201.368\1.2.840.113619.2.25.4.2147483647.1715913201.486\1.2.840.113619.2.5.4230407869.26071.1715913201.452.dic"
    # dicom_file2 = r"DICOM_FUNCTIONS\Pacs_Test\1.2.840.113619.2.25.4.2147483647.1715913201.368\1.2.840.113619.2.25.4.2147483647.1715913201.486\1.2.840.113619.2.5.4230407869.26071.1715913201.457.dic"

    dicom_file1 = r"DICOM_FUNCTIONS\Pacs_Test\1.2.840.113619.2.25.4.2147483647.1715918027.544\1.2.840.113619.2.25.4.2147483647.1715918028.333\1.2.840.113619.2.5.4230407869.9019.1715918027.926.dic"
    dicom_file2 = r"DICOM_FUNCTIONS\Pacs_Test\1.2.840.113619.2.25.4.2147483647.1715918027.544\1.2.840.113619.2.25.4.2147483647.1715918028.333\1.2.840.113619.2.5.4230407869.9019.1715918027.928.dic"


    image1 = load_dicom_image(dicom_file1)
    image2 = load_dicom_image(dicom_file2)

    image1_normalized = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image2_normalized = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    window_center = 127
    window_width = 255
    image1_windowed = cv2.convertScaleAbs(image1_normalized, alpha=(255.0/window_width))
    image2_windowed = cv2.convertScaleAbs(image2_normalized, alpha=(255.0/window_width))

    differences = compare_images_with_reference(image2_windowed, image1_windowed)

    # Mark only the top N high differences
    num_marks = 8  # Adjust number of marks as needed
    marked_image = mark_high_differences(image1_windowed, differences, num_marks)

    # Convert images to PNG format
    retval1, buffer1 = cv2.imencode('.png', image1_windowed)
    img_str1 = base64.b64encode(buffer1).decode()

    retval2, buffer2 = cv2.imencode('.png', image2_windowed)
    img_str2 = base64.b64encode(buffer2).decode()

    retval_marked, buffer_marked = cv2.imencode('.png', marked_image)
    img_str_marked = base64.b64encode(buffer_marked).decode()

    # dicom file 1 folder name remove
    split_path = dicom_file1.rsplit('\\', 1)
    folder_path = split_path[0]
    file_name1 = split_path[1]

    # dicom file 2 folder name remove
    split_path = dicom_file2.rsplit('\\', 1)
    folder_path = split_path[0]
    file_name2 = split_path[1]
    return render_template('dicom_pixel_difference.html', img_str1=img_str1, img_str2=img_str2, img_str_marked=img_str_marked, file_name1=file_name1, file_name2=file_name2)

if __name__ == "__main__":
    app.run(debug=True)
