import cv2
import numpy as np

from ultralytics import YOLO
import time
import os

def detect(image):
    model = YOLO('yolov8n.pt')
    results = model(image, conf=0.7)
    return results


import cv2
import numpy as np

def remove_object(image, boxes, class_names, object_name):
    # Convert image to RGBA (4 channels: Red, Green, Blue, Alpha)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Create a mask with the same dimensions as the image
    mask = np.zeros_like(image)

    for box in boxes:
        # class_id = box[0]
        # class_name = class_names[class_id]
        
        # if class_name == object_name:
        if class_names[int(box.cls.item())] ==object_name:
            x = int(box.xyxy[0][0].item())
            y = int(box.xyxy[0][1].item())
            w = int(box.xyxy[0][2].item())
            h = int(box.xyxy[0][3].item()) 

            # Draw a filled rectangle on the mask in white
            cv2.rectangle(mask, (x, y), (w, h), (255, 255, 255, 255), -1)

    # Use the mask to remove the object from the image
    image = cv2.bitwise_and(image, mask)

    return image

# Example usage
def main():
    unique_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

    image = cv2.imread("input\orig_20240422105235700.jpg")
    get_results = detect(image)

    # Object name to remove
    object_name = 'tie'

    # Remove the object and make the removed portion transparent
    result = remove_object(image, get_results[0].boxes, get_results[0].names, object_name)

    # Save the result
    cv2.imwrite(f"output/result_{unique_time}.png", result)

def mainlist():
    unique_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    input_dir = "input/"
    output_dir = "output/"

    # Object name to remove
    object_name = 'person'

    image_files = os.listdir(input_dir)
    for image_file in image_files:
        # Load the image
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)

        get_results = detect(image)


        # Remove the object and make the removed portion transparent
        result = remove_object(image, get_results[0].boxes, get_results[0].names, object_name)

        # Save the result
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        merged_image = np.concatenate((image, result), axis=1)

        # Save the merged image
        output_path = os.path.join(output_dir, f"merged_{unique_time}_{image_file}")
        cv2.imwrite(output_path, merged_image)


if __name__ == "__main__":
    main()
