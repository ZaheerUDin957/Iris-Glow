import streamlit as st
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from styles import overall_css
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
import streamlit as st
import matplotlib.pyplot as plt

def set_background_image(image_path):
    """
    Set a background image for the Streamlit app.
    :param image_path: str, path to the background image
    """
    # Read the image file
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    
    # Encode the image in base64
    import base64
    encoded_image = base64.b64encode(image_bytes).decode()
    
    # Define the CSS to set the background image
    background_image_style = f"""
    <style>
    .stApp {{
        background: url(data:image/jpg;base64,{encoded_image});
        background-size: cover;
    }}
    </style>
    """
    
    # Add the CSS to the Streamlit app
    st.markdown(background_image_style, unsafe_allow_html=True)


# Function to load and display images from a folder
def load_and_display_info(folder_path='./iris'):
    # Load Images
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.bmp')]  # Filter BMP files in the folder

    images = []  # Initialize an empty list to store loaded images
    for filename in image_files:
        img = cv2.imread(os.path.join(folder_path, filename))  # Load image using OpenCV
        images.append(img)  # Append loaded image to the list

    # Displaying the number of images loaded
    num_images = len(images)  # Count the number of loaded images
    st.write(f"<h3>Number of images loaded: {num_images}</h3>", unsafe_allow_html=True)  # Print the number of images loaded

    # Displaying dimensions of the first image
    if num_images > 0:
        height, width, channels = images[0].shape
        st.write(f"<h3>Dimensions of the sample image (height x width x channels): {height} x {width} x {channels}</h3>", unsafe_allow_html=True)
    return images, num_images

# Function to display a sample of images without iris detection
def display_sample_images(images, num_samples=6):
    # Randomly select indices
    num_images = len(images)
    if num_images > 0:
        sample_indices = np.random.choice(num_images, num_samples, replace=False)

        cols = st.columns(3)
        for i, idx in enumerate(sample_indices[:3]):
            with cols[i]:
                st.image(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB), channels="RGB")

        cols = st.columns(3)
        for i, idx in enumerate(sample_indices[3:6]):
            with cols[i]:
                st.image(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB), channels="RGB")

# Function to detect iris in an image
def detect_iris(image):
    """
    Detects the iris in the input image and returns the image with detected iris and its diameter.

    Parameters:
    - image: Input image containing the iris.

    Returns:
    - image_with_iris: Image with detected iris highlighted.
    - diameter: Diameter of the detected iris ellipse.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, 30, 150)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour as the iris
    if len(contours) > 0:
        iris_contour = max(contours, key=cv2.contourArea)

        # Fit an ellipse to the contour
        if len(iris_contour) >= 5:
            ellipse = cv2.fitEllipse(iris_contour)
            cv2.ellipse(image, ellipse, (0, 255, 0), 2)  # Draw ellipse on original image

            # Calculate diameter from the ellipse
            diameter = max(ellipse[1])  # Diameter is the maximum of width and height of the ellipse
            return image, diameter

    return None, None  # Return None if iris is not detected



def display_Labeled_Sample_Images(images, num_samples=6):
    # Randomly select indices
    num_images = len(images)
    if num_images > 0:
        sample_indices = np.random.choice(num_images, num_samples, replace=False)

        # Display images in two rows, each with three columns
        for row in range(2):
            cols = st.columns(3)
            for i in range(3):
                idx = sample_indices[row * 3 + i]
                detected_img, is_detected = detect_iris(images[idx])
                if is_detected:
                    cols[i].image(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB), channels="RGB", caption='With Iris')
                else:
                    cols[i].image(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB), channels="RGB", caption='Without Iris')

def display_detected_sample_images(detected_images_folder, num_samples=6):
    detected_image_files = [file for file in os.listdir(detected_images_folder) if file.endswith('.jpg')]

    # Randomly select indices
    num_images = len(detected_image_files)
    if num_images > 0:
        sample_indices = np.random.choice(num_images, num_samples, replace=False)
        
        for row in range(2):
            cols = st.columns(3)
            for i, idx in enumerate(sample_indices[row*3:(row+1)*3]):
                with cols[i]:
                    detected_img = cv2.imread(os.path.join(detected_images_folder, detected_image_files[idx]))
                    st.image(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB), channels="RGB", caption='Detected Iris Image')

def display_undetected_sample_images(undetected_images_folder, num_samples=6):
    undetected_image_files = [file for file in os.listdir(undetected_images_folder) if file.endswith('.jpg')]

    # Randomly select indices
    num_images = len(undetected_image_files)
    if num_images > 0:
        sample_indices = np.random.choice(num_images, num_samples, replace=False)
        
        for row in range(2):
            cols = st.columns(3)
            for i, idx in enumerate(sample_indices[row*3:(row+1)*3]):
                with cols[i]:
                    undetected_img = cv2.imread(os.path.join(undetected_images_folder, undetected_image_files[idx]))
                    st.image(cv2.cvtColor(undetected_img, cv2.COLOR_BGR2RGB), channels="RGB", caption='Undetected Iris Image')


def display_sample_with_diameter(input_folder='./detected_iris_images', num_samples=6):
    # Get image files
    image_files = [file for file in os.listdir(input_folder) if file.endswith('.jpg')]

    # Randomly select indices
    num_images = len(image_files)
    if num_images > 0:
        sample_indices = np.random.choice(num_images, num_samples, replace=False)

        for i in range(0, num_samples, 3):
            cols = st.columns(3)
            for j, idx in enumerate(sample_indices[i:i + 3]):
                with cols[j]:
                    img = cv2.imread(os.path.join(input_folder, image_files[idx]))  # Read image
                    detected_img, diameter = detect_iris(img)  # Detect iris and get diameter
                    if detected_img is not None and diameter is not None:
                        st.image(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB), channels="RGB",
                                 caption=f'Diameter: {diameter:.4f}px')  # Display image with caption



def extract_metadata_from_xml(xml_file):
    def get_text_or_default(element, default=""):
        """ Helper function to get text from an XML element or return a default value if the element is None. """
        return element.text if element is not None else default

    def extract_metadata(root):
        return {
            'Cine File Header Information': {
                'Cine Type': get_text_or_default(root.find("./CineFileHeader/Type")),
                'Compression': get_text_or_default(root.find("./CineFileHeader/Compression")),
                'Version': get_text_or_default(root.find("./CineFileHeader/Version")),
                'Total Image Count': get_text_or_default(root.find("./CineFileHeader/TotalImageCount")),
                'First Image No': get_text_or_default(root.find("./CineFileHeader/FirstImageNo")),
            },
            'Bitmap Info Header Information': {
                'Bitmap Width': get_text_or_default(root.find("./BitmapInfoHeader/biWidth")),
                'Bitmap Height': get_text_or_default(root.find("./BitmapInfoHeader/biHeight")),
                'Bit Depth': get_text_or_default(root.find("./BitmapInfoHeader/biBitCount")),
                'Compression': get_text_or_default(root.find("./BitmapInfoHeader/biCompression")),
                'Size Image': get_text_or_default(root.find("./BitmapInfoHeader/biSizeImage")),
                'Color Planes': get_text_or_default(root.find("./BitmapInfoHeader/biPlanes")),
                'X Pixels per Meter': get_text_or_default(root.find("./BitmapInfoHeader/biXPelsPerMeter")),
                'Y Pixels per Meter': get_text_or_default(root.find("./BitmapInfoHeader/biYPelsPerMeter")),
            },
            'Camera Setup Information': {
                'Frame Rate': get_text_or_default(root.find("./CameraSetup/FrameRate")),
                'Gamma': get_text_or_default(root.find("./CameraSetup/Gamma")),
                'Saturation': get_text_or_default(root.find("./CameraSetup/Saturation")),
            }
        }

    def display_metadata(metadata):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<h2>Cine File Header Information</h2>", unsafe_allow_html=True)
            cine_info = metadata.get('Cine File Header Information', {})
            for key, value in cine_info.items():
                st.write(f"{key}: {value}")

        with col2:
            st.markdown("<h2>Bitmap Info Header Information</h2>", unsafe_allow_html=True)
            bitmap_info = metadata.get('Bitmap Info Header Information', {})
            for key, value in bitmap_info.items():
                st.write(f"{key}: {value}")

        with col3:
            st.markdown("<h2>Camera Setup Information</h2>", unsafe_allow_html=True)
            camera_info = metadata.get('Camera Setup Information', {})
            for key, value in camera_info.items():
                st.write(f"{key}: {value}")

    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract and display metadata
    metadata = extract_metadata(root)
    display_metadata(metadata)


def detect_flash(img1, img2, threshold=50, roi=None):
    """
    Detect if a flash occurred between two images. Optionally consider only a region of interest (ROI).
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    
    if roi:
        x, y, w, h = roi
        diff = diff[y:y+h, x:x+w]  # Crop to ROI
    
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    non_zero_count = np.count_nonzero(thresholded)
    return non_zero_count > threshold

def load_flash_times(xml_file_path):
    """
    Load flash times from an XML file.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    flash_times = []
    for time_block in root.findall('.//TIMEBLOCK'):
        for time_elem in time_block.findall('.//Time'):
            flash_times.append(float(time_elem.text.split()[1]))
    
    return flash_times

def draw_roi(image, roi):
    """
    Draw a red, bold rectangle around the ROI on the image.
    """
    x, y, w, h = roi
    return cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red rectangle

def display_images_with_flash_status(images_folder, threshold=50, max_display_count=6):
    """
    Load images from the specified directory, detect flashes, and display the results with flash status.
    """
    images = [cv2.imread(os.path.join(images_folder, f)) for f in sorted(os.listdir(images_folder)) if f.endswith(('.jpg', '.jpeg'))]

    if len(images) < 2:
        st.write("Not enough images to detect flashes.")
        return
    
    flash_status_images = []
    
    for i in range(0, len(images) - 1):
        img1, img2 = images[i], images[i + 1]
        flash_detected = detect_flash(img1, img2, threshold)
        
        if flash_detected:
            flash_status_images.append((img2, True))
        else:
            flash_status_images.append((img2, False))
        
        if len(flash_status_images) >= max_display_count:
            break
    
    # Display images in Streamlit with flash status
    cols = st.columns(3)
    for i, (img, flash_detected) in enumerate(flash_status_images):
        with cols[i % 3]:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

def display_images_with_roi(images_folder, roi, threshold=50, max_display_count=6):
    """
    Load images from the specified directory, detect flashes, and display the results with ROI highlighted.
    """
    images = [cv2.imread(os.path.join(images_folder, f)) for f in sorted(os.listdir(images_folder)) if f.endswith(('.jpg', '.jpeg'))]

    if len(images) < 2:
        st.write("Not enough images to detect flashes.")
        return
    
    flash_images_with_roi = []

    for i in range(0, len(images) - 1):
        img1, img2 = images[i], images[i + 1]
        flash_detected = detect_flash(img1, img2, threshold, roi)
        
        if flash_detected:
            img_with_roi = draw_roi(img2, roi)
            flash_images_with_roi.append(img_with_roi)
        
        if len(flash_images_with_roi) >= max_display_count:
            break
    
    # Display images in Streamlit with ROI
    cols = st.columns(3)
    for i, img in enumerate(flash_images_with_roi):
        with cols[i % 3]:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", caption='Flash Detected')

def display_images_with_flash_times(images_folder, xml_file_path, threshold=50, roi=None, max_display_count=6):
    """
    Load images from the specified directory, detect flashes, display the results with ROI and flash times.
    """
    images = [cv2.imread(os.path.join(images_folder, f)) for f in sorted(os.listdir(images_folder)) if f.endswith(('.jpg', '.jpeg'))]

    if len(images) < 2:
        st.write("Not enough images to detect flashes.")
        return
    
    flash_images_with_times = []
    flash_times = load_flash_times(xml_file_path)

    for i in range(0, len(images) - 1):
        img1, img2 = images[i], images[i + 1]
        flash_detected = detect_flash(img1, img2, threshold, roi)
        
        if flash_detected:
            img_with_roi = draw_roi(img2, roi) if roi else img2
            flash_images_with_times.append((img_with_roi, flash_times[i]))
        
        if len(flash_images_with_times) >= max_display_count:
            break
    
    # Display images in Streamlit with ROI and flash times
    cols = st.columns(3)
    for i, (img, flash_time) in enumerate(flash_images_with_times):
        with cols[i % 3]:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", caption=f'Flash Detected at {flash_time:.2f}s')



def main():

    st.set_page_config(page_title="LuminoIris Tracker", layout="wide")  # Set page title and layout
    st.markdown(overall_css, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>LuminoIris Tracker</h1>", unsafe_allow_html=True)

    image_path = "1.jpg"
    set_background_image(image_path)

    images, num_images = load_and_display_info()
    st.markdown("<h2><center>Sample Images</center></h2>", unsafe_allow_html=True)
    display_sample_images(images)
    st.markdown("<h2><center>Iris detected and Undetected Image Status</center></h2>", unsafe_allow_html=True)
    display_Labeled_Sample_Images(images, num_samples=6)

    # Step 5: Display a sample of undetected iris images
    st.markdown("<h2><center>Iris Undetected Images</center></h2>", unsafe_allow_html=True)
    display_undetected_sample_images('./undetected_iris_images')

    # Step 6: Display a sample of detected iris images
    st.markdown("<h2><center>Iris detected Images</center></h2>", unsafe_allow_html=True)
    display_detected_sample_images('./detected_iris_images')

    # Call the function to display sample images with their diameters
    st.markdown("<h2><center>Iris detected Images with diameter</center></h2>", unsafe_allow_html=True)
    display_sample_with_diameter('./detected_iris_images', 6)

    extract_metadata_from_xml('./iris/Infra000000.xml')

    # Example usage
    st.markdown("<h2><center>Iris Flash detected Images</center></h2>", unsafe_allow_html=True)
    display_images_with_flash_status('./detected_iris_images')
    st.markdown("<h2><center>Selection of Region of Interest (ROI)</center></h2>", unsafe_allow_html=True)
    display_images_with_roi('./detected_iris_images', roi=(0, 600, 300, 200))
    st.markdown("<h2><center>Flash Duration</center></h2>", unsafe_allow_html=True)
    display_images_with_flash_times('./detected_iris_images', './iris/Infra000000.xml', roi=(0, 600, 300, 200))

# Entry point of the application
if __name__ == "__main__":
    main()