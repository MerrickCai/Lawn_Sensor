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

model_path = os.path.join(script_dir, "best1800v6.pt")

if not os.path.exists(model_path):
    print(f"Model file not found at: {model_path}")
    sys.exit(1)

model = YOLO(model_path) #put YOLO model name here

class_names = ['-', 'Blue Violets', 'Broadleaf Plantains', 'Common Ivy','Common Purslane', 'Eastern Poison Ivy', 'Fallen Leaves','Japanese Honeysuckle', 'Oxeye Daisy', 'Roundleaf greenbrier', 'Virginia Creeper', 'Chickweed', 'Crabgrass', 'dandelions']
confidences = [1,0.6,0.6,0.5,0.8,0.5,0.2,0.2,0.2,0.2,1,0.5,0.5,0.1,0.5]
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
    #cv2.imshow(img) # Use cv2_imshow instead of cv2.imshow

# --------------- generate TGI image ---------------
def generateTGIimage(string):
    print("starting")
    image = cv2.imread(string)
    PILimage = Image.open(string)
    img_array = np.array(PILimage)
    PILpixels = PILimage.load()
    height, width, channels = image.shape
    average_rgb = np.mean(img_array, axis=(0, 1))
    #print(average_rgb)
    average_tgi = -0.05*((190)*(average_rgb[0]-average_rgb[1])-(120)*(average_rgb[0]-average_rgb[2]))/2
    #print(average_tgi)
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
            strToPrint = str(result[0].boxes.conf[i].item())
            file.write("confidence: "+strToPrint[:5]+", Name: "+class_names[index])
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
                confStr = confStr + f"Detected object: {class_name}\n"+f"Confidence: {confidence:.2f}" # Print confidence and detection
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
                    confStr = confStr + f"Detected object2: {class_name}\n"+f"Confidence: {confidence:.2f}" # Print confidence and detection
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
    i = 10
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
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")
        uniqueSpecies = uniqueSpecies + 1
    stats = f"Total Plants: {count}\nAverage Triangular Greenness Index: ?\nUnique Species: {uniqueSpecies}"
    print(stats)
    return confStrGroup

# --------------- Main function to run the script ---------------
if __name__ == "__main__":

    print("Starting script...")

    if len(sys.argv) < 3:
        print("No input provided.")
        sys.exit(1)

    mode = sys.argv[1]
    print(f"Mode: {mode}")

    # ----------- frame mode --------------
    if mode == "frames":
        video_filename = sys.argv[2]
        video_path = os.path.join("frontend", "uploadVideo", video_filename)
        output_dir = os.path.join("frontend", "framesImages")
        fraction = 1.0
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
        print("Average TGI: " + str(avr_tgi))
        yoloResults = generateYOLOimages(inputString)
        print("YOLO: " + yoloResults)
        res_color = color_detection(inputString)
        (Image.fromarray(res_color)).save(f'frontend/YOLOimages/environments.jpg')
        yoloAll= model(source=inputString,conf=0.1,imgsz=2048,iou=0.5,classes=[1,2,3,4,5,6,7,8,9,10,11,12,13,14])
        yoloAll[0].save(f'frontend/YOLOimages/all.jpg')
    else:
        print("Unknown mode")
        sys.exit(1)

    print("Done")
    sys.exit(0)
