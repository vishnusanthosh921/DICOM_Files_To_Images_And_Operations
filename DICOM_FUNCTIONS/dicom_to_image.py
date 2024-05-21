import pydicom
from PIL import Image

def convert_dicom_to_image(dicom_file):
    ds = pydicom.dcmread(dicom_file)
    pixel_array = ds.pixel_array
    
    # Convert pixel array to PIL image
    image = Image.fromarray(pixel_array)
    
    # Save image
    image.save('output_image6.png')
    print("Image saved as 'output_image6.png'")

if __name__ == "__main__":
    dicom_file = input("Enter the path to the .dic file: ")
    
    if not dicom_file.endswith('.dic'):
        print("Invalid DICOM file.")
    else:
        convert_dicom_to_image(dicom_file)
