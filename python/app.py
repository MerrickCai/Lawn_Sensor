import sys
import time
import os
import numpy as np
from PIL import Image
import cv2
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from ultralytics import YOLO
from collections import Counter
from pathlib import Path
import shutil
import json


# ----------------- Global Variables -----------------
#This creates separate images for each plant (test)
script_path = os.path.abspath(__file__)    # c:\Users\Merrick\Desktop\lawn_sensor\python\app.py
script_dir = os.path.dirname(script_path)  # c:\Users\Merrick\Desktop\lawn_sensor\python
project_root = os.path.dirname(script_dir)  # c:\Users\Merrick\Desktop\lawn_sensor
# Create directories for saving images
tgi_dir = os.path.join(project_root, "frontend", "TGIimage")
yolo_dir = os.path.join(project_root, "frontend", "YOLOimages")
os.makedirs(tgi_dir, exist_ok=True)
os.makedirs(yolo_dir, exist_ok=True)
model_path = os.path.join(script_dir, "best1800v6.pt") #put YOLO model name here
if not os.path.exists(model_path):
    print(f"Model file not found at: {model_path}")
    sys.exit(1)
model = YOLO(model_path)
class_names = ['-', 'Blue Violets', 'Broadleaf Plantains', 'Common Ivy','Common Purslane', 'Eastern Poison Ivy', 'Fallen Leaves','Japanese Honeysuckle', 'Oxeye Daisy', 'Roundleaf greenbrier', 'Virginia Creeper', 'Chickweed', 'Crabgrass', 'dandelions']
confidences = [1,0.4,0.3,0.5,0.65, 0.275,0.275,0.7,0.225,0.225, 0.25,0.3,0.375,0.225]
grid_size_1 = 1#put first grid size here
grid_size_2 = 1 #put second grid size here


# --------------- generate output frames ---------------
def generateOutputFrames(FILENAME, outputDirectory, fraction):
    # Interval between saved frames
    n = 10
    print(n)
    # Create output folder if it doesn't exist
    if not os.path.exists(outputDirectory):
        print("Out: "+outputDirectory)
        os.makedirs(outputDirectory)
    else:
        print("exists")
    # Open the video file
    video = cv2.VideoCapture(FILENAME)
    print(FILENAME)
    print("frontend/uploadVideo/gopro1.mp4")
    print("File exists:", os.path.exists("frontend/uploadVideo/gopro1.mp4"))
    print(FILENAME)
    print(video)
    # Check if video opened successfully
    if not video.isOpened():
        print("Error: Could not open video file")
    # Get video properties
    print(f"Total frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"FPS: {video.get(cv2.CAP_PROP_FPS)}")
    # Initialize frame counter
    count = 1
    # Read all video frames
    while True:
        # Read frame
        success, frame = video.read()
        # Break the loop if we reach the end of the video
        if not success:
            break
        # Generate output filename
        output_path = os.path.join(outputDirectory, f"frame_{count:06d}.jpg")
        # Increment counter
        count += 1
        # Save every n frames
        if count % n == 1:
            # Resize frame to fraction of original size
            height, width = frame.shape[:2]
            new_dimensions = (int(width * fraction), int(height * fraction))
            resized_frame = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_AREA)
            # Save resized frame as JPG file
            cv2.imwrite(output_path, resized_frame)
            print(f"Processed {count} frames")
    # Release video capture object
    video.release()
    print(f"Extraction completed. Total frames extracted: {count/n}")


# --------------- filter image ---------------
colors = {
    'red': ([0, 50, 50], [10, 255, 255]),  # Dirt
    'brown': ([10, 35, 50], [45, 255, 255]),  # Dead leaves/grass
    'green': ([35, 80, 20], [55, 255, 255]),  # Grass/greenery
    'purple': ([120, 50, 30], [160, 255, 255]),  # Purple flowers
    'yellow': ([20, 100, 100], [30, 255, 255]),  # Yellow flowers
    'white': ([0, 0, 150], [180, 50, 255])  # Snow, White flowers, or other white images.
}
def color_detection(image_path, overlap=True):
    img = cv2.imread(image_path)  #Read the image from file path when this function is called
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Converts image from RGB to HSV, to improve color-based segmentation
    #hsv uses hue, saturation, and brightness
    color_masks = {}  # Initializes a dictonary for the color masks
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding. Often used to spearate objects
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Find contours of white regions. This is used in detecting snow and other white things in images.
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on the original image (or process as needed)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 2)
    # Iterates through a colors dictionary
    # Creates a mask for current color by checking pixels that fall within each color range.
    # The created mask is added to the color_masks dictionary
    for color_name, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        color_masks[color_name] = mask
        # Overlapping colors: simply display the masks on the original image
    for color_name, mask in color_masks.items():
        img[mask > 0] = colors[color_name][0]  # Color the detected regions
    return img


# --------------- generate TGI image ---------------
def generateTGIimage(string):
    print("starting")
    image = cv2.imread(string)
    PILimage = Image.open(string)
    img_array = np.array(PILimage)
    PILpixels = PILimage.load()
    height, width, channels = image.shape
    average_rgb = np.mean(img_array, axis=(0, 1))
    average_tgi = -0.05*((190)*(average_rgb[0]-average_rgb[1])-(120)*(average_rgb[0]-average_rgb[2]))/2
    print(f"Image size: {width}x{height}")
    i = 0
    j = 0
    nm = 0
    #Loop through every pixel
    while (i < width):
        while (j < height):
            red,green,blue = image[j,i]
            if i > -1 and j > -1:
                px_val = image[j,i]
                redVal = np.int32(red)
                greenVal = np.int32(green)
                blueVal = np.int32(blue)
                #The triangular greenness index is calculated here
                tgi = -0.05*((190)*(redVal-greenVal)-(120)*(redVal-blueVal))/2
                if (tgi < 0):
                    red = (int)(min(abs(tgi),255))
                    green = 0
                    blue = 0
                else:
                    green = min(abs((int)(tgi)),255)
                    red = 0
                    blue = 0
                PILpixels[i,j] = red,green,blue
            j = j + 1
        i = i+ 1
        j = 0
    print(i)
    print(j)
    print(width)
    print(height)
    # Save the resulting image to TGI directory
    tgi_output_path = os.path.join(tgi_dir, "output_image_TGI3.jpg")
    PILimage.save(tgi_output_path)
    print(f"TGI image saved to {tgi_output_path}")
    return average_tgi

# --------------- divide image into multiple square parts ---------------
def divide_image(image, grid_size):
    h, w, _ = image.shape
    image_parts = []
    for i in range(grid_size):
        for j in range(grid_size):
            y_start = i * h // grid_size
            y_end = (i + 1) * h // grid_size
            x_start = j * w // grid_size
            x_end = (j + 1) * w // grid_size
            # Extract part of the image
            part = image[y_start:y_end, x_start:x_end]
            image_parts.append((part, (x_start, y_start)))
    return image_parts

# --------------- process each square with YOLO ---------------
def process_with_yolo(FILENAME,index):
    # Run YOLO model on the image
    detected_classes=[]
    result = model(source=FILENAME,conf=confidences[index],imgsz=2048,iou=0.5,classes=[index])
    print(str(confidences[index])+str("CONF"))
    print(str(confidences[index])+str([index]))
    image = cv2.imread(FILENAME)
    # Process detections
    for i, detection in enumerate(result[0].boxes):
        # Get bounding box coordinates
        x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])
        # Crop the NumPy array
        cropped_image = image[y_min:y_max, x_min:x_max]
        # Convert BGR (OpenCV format) to RGB if needed
        if len(cropped_image.shape) == 3 and cropped_image.shape[2] == 3:  # Check if image has 3 channels
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image and save
        pil_image = Image.fromarray(cropped_image)
        directory = f"boxes/box{index+1}/box_{i}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        pil_image.save(f"boxes/box{index+1}/box_{i}/image.jpg")
        file_path = os.path.join(directory, "info.txt")
        with open(file_path, "w") as file:
            print("_________________________________________"+str(result[0].boxes.conf[i].item()))
            strToPrint = str(result[0].boxes.conf[i].item())
            file.write("confidence: "+strToPrint[:5]+", Name: "+class_names[index])
            print("WRITTEN:"+"confidence: "+strToPrint[:5]+", Name: "+class_names[index])
            detected_classes.append(class_names[index])
    return result,detected_classes

# --------------- display results with semi-transparent ellipses ---------------
def display_results(image, image_parts_1, results_1, image_parts_2, results_2, index, detected_classes):
    confStr = ""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.axis('off')
    transparency = 0.5
    
    # Overlay the first grid results
    for i, ((part, (x_offset, y_offset)), result) in enumerate(zip(image_parts_1, results_1)):
        boxes = result[0].boxes.xyxy
        confidences = result[0].boxes.conf
        class_ids = result[0].boxes.cls
        if len(boxes) > 0:
            for j in range(len(boxes)):
                x1, y1, x2, y2 = boxes[j].tolist()
                class_id = int(class_ids[j])
                class_name = class_names[class_id]
                confidence = confidences[j].item()  # Get the confidence score
                confStr = confStr + f"Detected object: {class_name}\n"+f"Confidence Level: {confidence:.2f}" # Print confidence and detection
                detected_classes.append(class_name)
                x1 += x_offset
                y1 += y_offset
                x2 += x_offset
                y2 += y_offset
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                axes = (x2 - x1, y2 - y1)
                ax.add_patch(Ellipse(center, axes[0], axes[1], fill=True, color='r', alpha=transparency, linewidth=0))
    for i, ((part, (x_offset, y_offset)), result) in enumerate(zip(image_parts_2, results_2)):
            # Access the bounding boxes, confidences, and labels
            boxes = result[0].boxes.xyxy
            confidences = result[0].boxes.conf
            class_ids = result[0].boxes.cls
            if len(boxes) > 0:
                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[j].tolist()
                    class_id = int(class_ids[j])
                    class_name = class_names[class_id]
                    confidence = confidences[j].item()  # Get the confidence score
                    confStr = confStr + f"Detected object2: {class_name}\n"+f"Confidence Level: {confidence:.2f}" # Print confidence and detection
                    detected_classes.append(class_name)
                    # Adjust the coordinates based on the position of the grid part
                    x1 += x_offset
                    y1 += y_offset
                    x2 += x_offset
                    y2 += y_offset
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    axes = (x2 - x1, y2 - y1)  # Use the full width and height for axes
                    ax.add_patch(Ellipse(center, axes[0], axes[1], fill=True, color='g', alpha=transparency, linewidth=2))
                    plt.savefig('ellipses/figure'+str(index)+'.jpg', format='jpg',bbox_inches='tight', pad_inches=0, dpi=300)

    # Create ellipses directory inside YOLOimages
    ellipses_dir = os.path.join(yolo_dir, "ellipses")
    os.makedirs(ellipses_dir, exist_ok=True)

    # Save to the YOLO directory
    plt.savefig(os.path.join(ellipses_dir, f'figure{str(index)}.jpg'), format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.draw()
    plt.savefig(os.path.join(yolo_dir, 'plot.png'))
    plt.savefig(os.path.join(yolo_dir, 'ellipses_display.jpg'), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(yolo_dir, 'plot.jpg'), format='jpg')
    plt.close()
    return confStr

# --------------- Load the image ---------------
def generateYOLOimages(FILENAME):
    image = cv2.imread(FILENAME)
    image_resized = cv2.resize(image, (2048, 2048))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    i = 2
    confStrGroup = ""
    detected_classes = []
    (Image.fromarray(image_rgb)).save(f'frontend/YOLOimages/ellipses_display.jpg')
    while (i<15):
        image_parts_1 = divide_image(image_rgb, grid_size=grid_size_1)
        results_1,detected_classes_subset = process_with_yolo(FILENAME,i)
        results_1[0].save(f'frontend/YOLOimages/ellipses/figure{i}.jpg')
        i = i +1
        detected_classes.extend(detected_classes_subset)
    class_counts = Counter(detected_classes)
    print("\nCount of each detected object type:")
    print(confStrGroup)
    uniqueSpecies = 0
    total_count = 0
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")
        uniqueSpecies = uniqueSpecies + 1
        total_count += count
    stats = f"Total Plants: {total_count}\nAverage Triangular Greenness Index: ?\nUnique Species: {uniqueSpecies}"
    print(stats)
    return confStrGroup

def createImagesIndividualPlants(source, destination,avr_tgi,lat,long):
    # Check if source folder exists
    if not Path(source).is_dir():
        print(f"Source folder '{source}' does not exist or is not a directory.")
        return
    # Resolve paths to absolute paths for clarity
    source = str(Path(source).resolve())
    destination = str(Path(destination).resolve())
    # Create destination folder if it doesn't exist
    Path(destination).mkdir(parents=True, exist_ok=True)
    image_extensions = {'.jpg', '.jpeg', '.png'}
    # Counter for naming images
    counter = 1
    # List to store JSON data
    json_data = []
    # Walk through source folder and all subfolders recursively
    for root, _, files in os.walk(source):
        # Check if current folder is a 'box_' folder
        is_box_folder = os.path.basename(root).startswith('box_')
        # For box_ folders, process only one image and one text file
        if is_box_folder:
            image_processed = False
            text_processed = False
            last_image_path = None
            for file in files:
                source_path = os.path.join(root, file)
                # Process one image
                if not image_processed and Path(file).suffix.lower() in image_extensions:
                    # Generate new filename
                    new_filename = f"img_{counter:03d}{Path(file).suffix.lower()}"
                    destination_path = os.path.join(destination, new_filename)
                    try:
                        # Copy the file with metadata
                        shutil.copy2(source_path, destination_path)
                        #verify the file exists
                        if Path(destination_path).exists():
                            print(f"Successfully copied {source_path} to {destination_path}")
                            last_image_path = new_filename
                            print(f"New image name: {new_filename}")
                        else:
                            print(f"Failed to verify {destination_path} after copying")
                        counter += 1
                        image_processed = True
                    except (IOError, OSError) as e:
                        print(f"Error copying {source_path} to {destination_path}: {e}")
                        continue
                #Process text file
                if not text_processed and Path(file).suffix.lower() == '.txt' and last_image_path:
                    try:
                        with open(source_path, 'r', encoding='utf-8') as txt_file:
                            content = txt_file.read()
                            # Get confidence and name
                            first = content[12:17]
                            next_text = content[25:]
                            # Add to JSON data
                            print(content)
                            json_data.append({
                                "image_name": Path(last_image_path).stem,
                                "confidence": first,
                                "predicted_class": next_text,
                                "image_path": "images/" + last_image_path,
                                "gps": {"lat": lat, "long": long},
                                "average_TGI": avr_tgi
                            })
                            text_processed = True
                    except (IOError, OSError) as e:
                        print(f"Error reading {source_path}: {e}")
                        continue
                # Stop processing if both image and text are done
                if image_processed and text_processed:
                    break
            # Print "DONE" after processing a "box_" folder
            if image_processed or text_processed:
                print("DONE")
    # Write JSON data to data.json
    if json_data:
        json_path = "frontend/js/data/data.json"
        try:
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, indent=2)
            print(f"Successfully wrote JSON data to {json_path}")
        except (IOError, OSError) as e:
            print(f"Error writing to {json_path}: {e}")
# ------------- Get the GPS coordinates from a user-entered string ----------------
def extract_coordinates(coord_string):
    try:
        # Remove any extra whitespace and split by comma
        coords = coord_string.replace(" ", "").split(",")
        # Ensure exactly two values (lat, long)
        if len(coords) != 2:
            raise ValueError("Expected exactly two values separated by a comma")
        # Convert to floats
        latitude = float(coords[0])
        longitude = float(coords[1])
        # Validate latitude and longitude ranges
        if not (-90 <= latitude <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if not (-180 <= longitude <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        return (latitude, longitude)
    except (ValueError, AttributeError) as e:
        print(f"Error processing coordinates '{coord_string}': {e}")
        return None

# --------------- Main function to run the script ---------------
if __name__ == "__main__":
    print("Starting script...")
    if len(sys.argv) < 3:
        print("No input provided.")
        sys.exit(1)
    mode = sys.argv[1]
    print(f"Mode: {mode}")
    lat=40.5 # Default latitude and longitude
    long=-74.5
    # ----------- frame mode --------------
    if mode == "frames":
        video_filename = sys.argv[2]
        video_path = os.path.join("frontend", "uploadVideo", video_filename)
        output_dir = os.path.join("frontend", "framesImages")
        if not os.path.exists(output_dir):
            print(f"Folder '{output_dir}' does not exist.")
        else:
            # Iterate through all files in the folder
            for file_name in os.listdir(output_dir):
                # Check if the file ends with .jpg (case-insensitive)
                if file_name.lower().endswith('.jpg'):
                    file_path = os.path.join(output_dir, file_name)
                    try:
                        #delete the file
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
        fraction = 1.0 #the size of the extracted image (e.g. 0.5 means the image will be half the size of the video)
        print(f"Extracting frames from {video_path} to {output_dir} with fraction {fraction}")
        generateOutputFrames(video_path, output_dir, fraction)

    # ----------- analysis mode --------------
    elif mode == "analysis":
        data = sys.argv[2]
        print(f"Running script with input: {data}")
        index_int = int(data)
        if index_int < 10:
            inputString = f"frontend/framesImages/frame_0000{index_int}0.jpg"
        else:
            inputString = f"frontend/framesImages/frame_000{index_int}0.jpg"
        time.sleep(0.2)
        avr_tgi = generateTGIimage(inputString)
        print("Average_TGI: " + str(avr_tgi))
        yoloResults = generateYOLOimages(inputString)
        print("YOLO: " + yoloResults)
        res_color = color_detection(inputString)
        (Image.fromarray(res_color)).save(f'frontend/YOLOimages/environments.jpg')
        yoloAll= model(source=inputString,conf=0.2,imgsz=2048,iou=0.5,classes=[1,2,3,4,5,6,7,8,9,10,11,12,13,14])
        yoloAll[0].save(f'frontend/YOLOimages/all.jpg')
        with open('floats.txt', 'r') as file:
            lat = float(file.readline())
            long = float(file.readline())
        createImagesIndividualPlants("boxes","frontend/images",avr_tgi,lat,long)
    elif mode == "GPS":
        data = sys.argv[2]
        print(f"Running script with input: {data}")
        (lat, long) = extract_coordinates(str(data))
        with open('floats.txt', 'w') as file:
            pass
        with open('floats.txt', 'w') as file:
            file.write(f"{lat}\n")
            file.write(f"{long}\n")
    else:
        print("Unknown mode")
        sys.exit(1)
    print("Done")
    sys.exit(0)
