#This combines the TGI and mean filtering tests
import numpy as np
from PIL import Image
import cv2
from collections import Counter
from scipy import stats
import time
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from ultralytics import YOLO

#This creates separate images for each plant (test)
model = YOLO("best1700v1.pt") #put YOLO model name here
FILENAME = "Lawn_Sensor-main/images/img_002.jpg"
#['-', 'Blue Violets', 'Broadleaf Plantains', 'Common Ivy','Common Purslane',
# 'Eastern Poison Ivy', 'Japanese Honeysuckle', 'Oxeye Daisy', 'Roundleaf greenbrier', 'Virginia Creeper',
#'Wild Garlic and others - v1 2025-03-25 9-53am', 'chickweed', 'crabgrass-weed', 'dandelions']
confidences = [1,0.6,0.6,0.5,0.8,0.5,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5]
grid_size_1 = 1#put first grid size here
grid_size_2 = 1 #put second grid size here

# Input: ./video/gopro1.mp4, ./framesImage, 0.1
def generateOutputFrames(FILENAME, outputDirectory, fraction):
    # Interval between saved frames
    n = 10
    # Create output folder if it doesn't exist
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)
    # Open the video file
    video = cv2.VideoCapture(FILENAME)
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

def filterImage(size,string):
#read the image
    imageUnsized = cv2.imread(string)
    heightSize, widthSize, channels = imageUnsized.shape
    imageB = cv2.resize(imageUnsized, (int(widthSize*size), int(heightSize*size)))
    imageC = Image.fromarray(imageB.astype(np.uint8))
    imageC.save("resized_image.png")
    imageUnchanged = Image.fromarray(imageUnsized.astype(np.uint8))
    imageUnchanged.save("viewingImage.png")
    time.sleep(0.3)
    image = cv2.imread("resized_image.png")
    PILimage = Image.open("resized_image.png")
    PILimage.save("output_image_DELETE.png")
    height, width, channels = image.shape
    img_array = np.array(PILimage)
    PILpixels = PILimage.load()
    print(f"Image size: {width}x{height}")
    i = 0
    j = 0

    # Loop through all pixels for analysis
    while (i < (width)):
        print(i)
        while (j < (height)):
            px_val = image[j,i]
            #red, green, and blue values are set
            r = int(px_val[2])
            g = int(px_val[1])
            b = int(px_val[0])
            #An overlay color is set based on the image
            #dirt
            if (r > 50 and r < 520 and g > 50 and g < 160 and b < 200 and abs(r-g)>10):
                red,green,blue = 255,0,0
            # Grass (Greenery)
            if 40 < r and r < 200 and 80 < g and g < 250 and 20 < b and b < 160 and 30 < abs(r-g) and abs(r-g) < 100 and 0 < abs(g-b) and abs(g-b) < 150 and 0 < abs(r-b) and abs(r-b) < 100:
                red,green,blue = red,green,blue = 0,255,0#0,255,0
            #road
            elif r > 100 and g > 70 and b < 60 and abs(g - r) < 1:# was 50
                red,green,blue = 128,128,128
            #sidewalk
            elif r > 255 and g > 150 and b > 150 and abs(r - g) < 50 and abs(r - b) < 50:
                red,green,blue = 255,255,128
            #yellow flowers
            elif r > 200 and g > 200 and b < 100 and abs(r - g) > 50:
                red,green,blue = 255,255,0
            # Purple Flowers
            elif r > 100 and g < 100 and b > 150 and abs(r - b) < 50:
                red,green,blue = 255,128,255
            # Snow
            elif r > 255 and g > 200 and b > 200 and abs(r - g) < 50 and abs(r - b) < 50:
                red,green,blue = 255,255,255
            # Dead Grass/Leaves
            elif r > 100 and g > 70 and b < 60 and abs(g - r) < 50:
                red,green,blue = 128,128,0
            else:
                green = 0
                red = 0
                blue = 128
            PILpixels[i,j] = red,green,blue
            j = j + 1
        i = i+ 1
        j = 0
    print(i)
    print(width)
    PILimage.save("output_image.png")
    # Load the image
    resizeWidth = int(widthSize*size)
    resizeHeight = int(heightSize*size)
    print(resizeWidth)
    print(resizeHeight)
    img = PILimage
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Get the image dimensions
    height, width, channels = img_array.shape
    # Define the kernel size 
    kernel_size = 4
    half_kernel = kernel_size // 2
    # Create an empty array to store the new image
    filtered_img_array = np.zeros_like(img_array)
    # Apply the median filtering
    for i in range(half_kernel, height - half_kernel):
        for j in range(half_kernel, width - half_kernel):
            # Extract the grid of neighbors
            neighbors = img_array[i - half_kernel:i + half_kernel + 1, j - half_kernel:j + half_kernel + 1]
            # Compute the median of the neighbors
            non_black_mask =  np.any(neighbors != [0., 0., 0.], axis=-1)
            non_black_neighbors = neighbors[non_black_mask]
            if non_black_neighbors.size > 0:
                # Count the frequency of each color
                mode_color, _ = stats.mode(non_black_neighbors, axis=0)
            else:
                mode_color = np.array([0, 0, 256])
                print("ELSE")
            # Assign the average color to the current pixel
            filtered_img_array[i, j] = mode_color
    print("loops B")
    print(i)
    print(j)
    #Convert the filtered array back to an image
    filtered_img = Image.fromarray(filtered_img_array.astype(np.uint8))
    #Display the filtered image
    originalImage = Image.open("resized_image.png")
    originalResized = originalImage#.resize((resizeHeight,resizeWidth))
    newIm = filtered_img
    # Number original image is multiplied by
    ratio = 0.8
    # Number filtered image is divided by
    filter_ratio = 1.5
    print("loopsC")
    print(resizeWidth)
    print(resizeHeight)

    for i in range(int(resizeWidth-kernel_size)):
        for j in range(int(resizeHeight-kernel_size)):
            r1,g1,b1 = filtered_img.getpixel((i,j))
            r2,g2,b2 = originalResized.getpixel((i,j))
            r2=r2*ratio
            g2=g2*ratio
            b2=b2*ratio
            newIm.putpixel((i,j),((int)((r1/filter_ratio+r2)),(int)((g1/filter_ratio+g2)),(int)((b1/filter_ratio+b2))))
    newIm_array = np.array(newIm)

    # Get height and width (note: OpenCV uses (height, width), PIL uses (width, height))
    h, w = newIm_array.shape[:2]  # Only take first two dimensions (h, w), ignore channels if present

    # Resize to 2x bigger using cv2
    finalIm = cv2.resize(newIm_array, (w * 2, h * 2))  # Width first, then height in cv2

    # Convert back to PIL Image and save
    finalIm_pil = Image.fromarray(finalIm.astype(np.uint8))
    finalIm_pil.save("filtered_img_resized3.png")

def generateTGIimage(string):
    print("starting")
    image = cv2.imread(string)
    PILimage = Image.open(string)
    img_array = np.array(PILimage)
    PILpixels = PILimage.load()
    height, width, channels = image.shape
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
    # Save the resulting image
    PILimage.save("output_image_TGI3.jpg")

# Function to divide image into multiple squrae parts
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

# Function to process each square with YOLO
def process_with_yolo(image_parts,confidence,index):
    results = []
    for part, _ in image_parts:
        result = model(part, conf=confidences[index],classes=[index])
        results.append(result)
    return results

# Function to display results with semi-transparent ellipses
def display_results(image, image_parts_1, results_1, image_parts_2, results_2,index):
    # Create a plot to overlay both grid results
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.axis('off')  # Remove axes
    #The transparency of the ellipses, where 1 equals 100%
    transparency = 0.5
    # Overlay the first grid results
    for i, ((part, (x_offset, y_offset)), result) in enumerate(zip(image_parts_1, results_1)):
        # Access the bounding boxes, confidences, and labels
        boxes = result[0].boxes.xyxy
        confidences = result[0].boxes.conf
        class_ids = result[0].boxes.cls
        if len(boxes) > 0:
            for j in range(len(boxes)):
                x1, y1, x2, y2 = boxes[j].tolist()
                # Adjust the coordinates based on the position of the grid part
                x1 += x_offset
                y1 += y_offset
                x2 += x_offset
                y2 += y_offset
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                axes = (x2 - x1, y2 - y1)
                ax.add_patch(Ellipse(center, axes[0], axes[1], fill=True, color='r', alpha=transparency, linewidth=0))

    # Overlay the second grid results
    for i, ((part, (x_offset, y_offset)), result) in enumerate(zip(image_parts_2, results_2)):
        # Access the bounding boxes, confidences, and labels
        boxes = result[0].boxes.xyxy
        confidences = result[0].boxes.conf
        class_ids = result[0].boxes.cls
        if len(boxes) > 0:
            for j in range(len(boxes)):
                x1, y1, x2, y2 = boxes[j].tolist()
                # Adjust the coordinates based on the position of the grid part
                x1 += x_offset
                y1 += y_offset
                x2 += x_offset
                y2 += y_offset
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                axes = (x2 - x1, y2 - y1)  # Use the full width and height for axes
                ax.add_patch(Ellipse(center, axes[0], axes[1], fill=True, color='g', alpha=transparency, linewidth=2))
                #plt.savefig('ellipses/figure'+str(index)+'.jpg', format='jpg',bbox_inches='tight', pad_inches=0, dpi=300)
    plt.savefig('ellipses/figure'+str(index)+'.jpg', format='jpg',bbox_inches='tight', pad_inches=0, dpi=300)
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.draw()
    #plt.show()
    plt.savefig('plot.png')
    #plt.savefig('overlaid_results_ellipses.jpg', bbox_inches='tight', pad_inches=0)
    plt.savefig('ellipses_display.jpg', bbox_inches='tight', pad_inches=0)
    plt.savefig('plot.jpg', format='jpg')

# Load the image
def generateYOLOimages(FILENAME):
    image = cv2.imread(FILENAME)
    image_resized = cv2.resize(image, (256, 256))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    i = 1
    while (i<15):
        image_parts_1 = divide_image(image_rgb, grid_size=grid_size_1)
        results_1 = process_with_yolo(image_parts_1,0.1,i)
        i = i +1
        display_results(image_rgb, image_parts_1, results_1, image_parts_1, results_1,i)

def runTest5():
    data = request.json.get('input', 'NULL')  # Get input from the request
    print("data:"+str(data)+".")
    if (str(data)=="G" or str(data)=="g"):
        generateOutputFrames("Lawn_Sensor-main/gopro1.mp4","output_frames",0.1)
    else:
        if (int(data) < 10):
            inputString = "output_frames/frame_0000"+str(data)+"0.jpg"
        else:
            inputString = "output_frames/frame_000"+str(data)+"0.jpg"
        
        print("data:"+str(data)+".")
        time.sleep(0.2)
        generateTGIimage(inputString)
        # filterImage(0.5,inputString)
        generateYOLOimages(inputString)
    result = f"Processed: {data.upper()}"  # Example processing
    return jsonify({'result': result})