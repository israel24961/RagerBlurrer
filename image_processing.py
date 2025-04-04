# Person and car identification using YOLOv8
import glob
import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
from enum import Enum

# Load the YOLOv8 model, medium version
model = YOLO('yolov8x.pt')  # Load the YOLOv8 model

#Plate detectiono5
from fast_alpr import ALPR

# You can also initialize the ALPR with custom plate detection and OCR models.
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)

class Yoloclasses(Enum):
    PERSON = 0
    BICYCLE = 1
    CAR = 2
    MOTORCYCLE = 3
    AIRPLANE = 4
    BUS = 5
    TRAIN = 6
    TRUCK = 7
    BOAT = 8
    TRAFFIC_LIGHT = 9
    FIRE_HYDRANT = 10
    STOP_SIGN = 11
    PARKING_METER = 12
    BENCH = 13
    BIRD = 14
    CAT = 15
    DOG = 16
 


# YOLOv8 then fast_alpr on the detected vehicles
def blur_cars_and_persons(image, plate_rectangle_location):
    # Perform inference
    results = model(image,conf=0.2, classes=[Yoloclasses.PERSON.value, Yoloclasses.CAR.value, Yoloclasses.MOTORCYCLE.value,
                                    Yoloclasses.BUS.value, Yoloclasses.TRUCK.value] )  # Perform inference

    if plate_rectangle_location is not None:
        #Discard the car rectangle that contain the plate
        x1, y1, x2, y2 = plate_rectangle_location
        # Create a mask for the plate rectangle
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255  # Set the plate rectangle area to 255 (white)
        # Blur the image
        blurred_image = cv2.GaussianBlur(image, (15, 15), 0)  # Apply Gaussian blur to the entire image
        # Combine the blurred image with the original image using the mask
        image = cv2.bitwise_and(blurred_image, blurred_image, mask=mask)  # Apply the mask to the blurred image
        # Invert the mask to get the area outside the plate rectangle
        mask_inv = cv2.bitwise_not(mask)  # Invert the mask
        # Combine the original image with the blurred image using the inverted mask
        image = cv2.bitwise_and(image, image, mask=mask_inv)  # Apply the inverted mask to the original image
        # Draw the plate rectangle on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the plate rectangle on the image
        # Draw the plate text on the image
        cv2.putText(image, 'Plate', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Put the label on the image

    for result in results:
        boxes = result.boxes.xyxy  # Get the bounding boxes
        confidences = result.boxes.conf  # Get the confidence scores
        classes = result.boxes.cls  # Get the class IDs

        for box, conf, cls in zip(boxes, confidences, classes):
            if conf<0.3 and int(cls) is not 1:
                continue  
            x1, y1, x2, y2 = map(int, box)  # Get the coordinates of the bounding box
            label = f'{model.names[int(cls)]} {conf:.2f}'  # Create a label with class name and confidence
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a rectangle around the detected object
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Put the label on the image


    vehicleResults_classes=[2, 3, 5, 7]
    vehicleResults = []
    for result in results:
        boxes = result.boxes.xyxy  # Get the bounding boxes
        confidences = result.boxes.conf  # Get the confidence scores
        classes = result.boxes.cls  # Get the class IDs

        for box, conf, cls in zip(boxes, confidences, classes):
            if int(cls) in vehicleResults_classes:
                x1, y1, x2, y2 = map(int, box)  # Get the coordinates of the bounding box
                vehicleResults.append((x1, y1, x2, y2))

    # In eache vehicle, detect the plate
    for vehicleResult in vehicleResults:
        x1, y1, x2, y2 = vehicleResult  # Get the coordinates of the bounding box
        # Detect the plate in the vehicle
        plate_rectangle_locations = alpr.predict(image[y1:y2, x1:x2])
        print(f"Plate rectangle location: {plate_rectangle_location}")
        # In original image paint in red the rectangle ( translating the coordinates)
        for prl in plate_rectangle_locations:
            # Get the coordinates of the plate rectangle
            plate_rectangle_location = prl.detection.bounding_box
            plate_rectangle_location = (
                plate_rectangle_location.x1 + x1,
                plate_rectangle_location.y1 + y1,
                plate_rectangle_location.x2 + x1,
                plate_rectangle_location.y2 + y1,
                )
            # Draw the plate rectangle on the image in red 
            cv2.rectangle(image, (int(plate_rectangle_location[0]), int(plate_rectangle_location[1])),
                          (int(plate_rectangle_location[2]), int(plate_rectangle_location[3])), (0, 0, 255), 2)
            # Draw the plate text on the image
            cv2.putText(image, 'Plate', (int(plate_rectangle_location[0]), int(plate_rectangle_location[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    # print(f"Processed image {results}")
    return image

def only_fast_alpr(image):
    # Detect the plate in the image
    plate_rectangle_locations = alpr.predict(image)
    print(f"Plate rectangle location: {plate_rectangle_locations}")
    # In original image paint in red the rectangle
    for prl in plate_rectangle_locations:
        # Get the coordinates of the plate rectangle
        plate_rectangle_location = prl.detection.bounding_box
        # Draw the plate rectangle on the image in red 
        cv2.rectangle(image, (int(plate_rectangle_location.x1), int(plate_rectangle_location.y1)),
                      (int(plate_rectangle_location.x2), int(plate_rectangle_location.y2)), (0, 0, 255), 2)
        # Draw the plate text on the image
        cv2.putText(image, 'Plate', (int(plate_rectangle_location.x1), int(plate_rectangle_location.y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image

# Picture path

search_path = './input/mpv-shot*'
# Get the list of files matching the pattern
files = glob.glob(search_path)
# Check if any files were found
if not files:
    print("No files found matching the pattern.")
    exit(1)

print(f"Found {len(files)} files matching the pattern: {search_path}")

for picture_path in files:
# Load the image

    image = cv2.imread(picture_path)  # Read the image using OpenCV

    time_start = time.time()  # Start the timer

    processed_image = blur_cars_and_persons(image, None)  # Process the image
    # processed_image = only_fast_alpr(image)  # Process the image with only fast_alpr

    time_end = time.time()  # End the timer
    print(f"Processing time: {(time_end - time_start)*1000:.2f} ms")  # Print the processing time

    
    # Save the image with detections
    output_path = os.path.join('output', os.path.basename(picture_path))  # Create output path
    os.makedirs('output', exist_ok=True)  # Create output directory if it doesn't exist
    cv2.imwrite(output_path, processed_image)  # Save the image with detections
    print(f"Processed {picture_path}, saved to {output_path}")  # Print processing information
