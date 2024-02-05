import argparse
import os
import torch
from ultralytics import YOLO

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--data', type=str, default='data.yaml') # yaml config file for custom dataset
    parser.add_argument('--epochs', type=int, default=3) # number of training epochs
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt') # pretrained base model
    parser.add_argument('--saved-model-name', type=str, default='model.pt') # name for model export
    
    # data and model directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    # obtain arguments
    args, _ = parser.parse_known_args()

    # build model
    model = YOLO(args.yolo_model)

    # train model
    results = model.train(data=args.data, epochs=args.epochs)
    
    # save model - parameters only in state_dict
    model_path = os.path.join(args.model_dir, args.saved_model_name)
    torch.save(model.state_dict(), model_path)