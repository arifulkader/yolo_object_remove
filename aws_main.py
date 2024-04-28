from dotenv import load_dotenv
import boto3
import os
import cv2
import numpy as np
import time

import onnxruntime


load_dotenv(verbose=True)


aws_client = boto3.client('rekognition',os.getenv("AWS_REGION"),aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),aws_secret_access_key=os.getenv("AWS_SECRET_kEY"))

def aws_detect(image):
    success, byte_image = cv2.imencode('.jpg', image)
    
    response = aws_client.detect_labels(
            Image={
                'Bytes': byte_image.tobytes()
            },
            Features=[
                'GENERAL_LABELS'
            ]   
        )
    return response
    
def remove_object(image, object_info,object_name):
    # Convert image to RGBA (4 channels: Red, Green, Blue, Alpha)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Create a mask with the same dimensions as the image
    mask = np.zeros_like(image)

    # Extract object bounding box information
    for instance in object_info['Labels']:
        if instance['Name'] == object_name:
            for box in instance['Instances']:
                bbox = box['BoundingBox']
                x = int(bbox['Left'] * image.shape[1])
                y = int(bbox['Top'] * image.shape[0])
                w = int(bbox['Width'] * image.shape[1])
                h = int(bbox['Height'] * image.shape[0])

                # Draw a filled rectangle on the mask in white
                cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255, 255), -1)

    # Use the mask to remove the object from the image
    image = cv2.bitwise_and(image, mask)

    return image


def background_remove_process(image_path):
    model_path = "temp\modnet.pth"

    ref_size = 512

    # Get x_scale_factor & y_scale_factor to resize image
    def get_scale_factor(im_h, im_w, ref_size):

        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32

        x_scale_factor = im_rw / im_w
        y_scale_factor = im_rh / im_h

        return x_scale_factor, y_scale_factor

    ##############################################
    #  Main Inference part
    ##############################################

    # read image

    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # unify image channels to 3
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # normalize values to scale it between -1 to 1
    im = (im - 127.5) / 127.5

    im_h, im_w, im_c = im.shape
    x, y = get_scale_factor(im_h, im_w, ref_size)

    # resize image
    im = cv2.resize(im, None, fx=x, fy=y, interpolation=cv2.INTER_AREA)

    # prepare input shape
    im = np.transpose(im)
    im = np.swapaxes(im, 1, 2)
    im = np.expand_dims(im, axis=0).astype('float32')

    # Initialize session and get prediction
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: im})

    # refine matte
    matte = (np.squeeze(result[0]) * 255).astype('uint8')
    matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation=cv2.INTER_AREA)

    # cv2.imwrite(output_path, matte)

    ##############################################
    # Optional - save png image without background
    ##############################################
    return matte



def main():
    unique_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

    image = cv2.imread("input\orig_20240422020858139.jpg")
    get_results = aws_detect(image)

    # Object name to remove
    object_name = 'Person'

    result = remove_object(image, get_results, object_name)
    
    file_name =f"output/result_{unique_time}.png"
    cv2.imwrite(file_name, result)
    
    # result = background_remove_process(file_name)
    # cv2.imwrite(file_name, result)


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
    # mainlist()
