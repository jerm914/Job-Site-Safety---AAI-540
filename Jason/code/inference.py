import argparse
import json
import numpy as np
import os
import torch
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

    
def input_fn(request_body, request_content_type):
    if request_content_type == 'image/jpeg':
        
        # deserialize image data from raw file
        return np.array(Image.open(BytesIO(request_body)))

    
def model_fn(model_dir):
    
    # build model and load trained weights
    weights_path = os.path.join(model_dir, 'model.pt')
    model = YOLO(weights_path)
    
    return model
    

def predict_fn(input_data, model):
    
    # perform inference and return results list
    results_list = model(input_data)

    return results_list


def output_fn(prediction, response_content_type):
    # process prediction and return as json format
    if response_content_type == 'application/json':
        
        # list to store customized results in json format
        results_json = []

        # iterate through each sample image
        for sample in prediction:

            # iterate through each result
            for r in sample:
                boxes = r.boxes
                class_id = int(boxes.cls.item())
                class_name = r.names[class_id]
                conf = round(boxes.conf.item(), 4)
                boxes_xywh = [round(value.item(), 4) for value in boxes.xywh[0]]
                box_x = boxes_xywh[0]
                box_y = boxes_xywh[1]
                box_w = boxes_xywh[2]
                box_h = boxes_xywh[3]

                # define data dictionary
                data_dict = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "box_x": box_x,
                    "box_y": box_y,
                    "box_w": box_w,
                    "box_h": box_h
                }

                # append result to list in json format
                results_json.append(json.dumps(data_dict))

        return results_json
    

if __name__ =='__main__':
    
    parser = argparse.ArgumentParser()
    
    # model directory
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    # obtain arguments
    args, _ = parser.parse_known_args()
    
    # load model with saved weights
    model = model_fn(args.model_dir)