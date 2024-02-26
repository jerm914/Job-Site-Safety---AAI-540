import argparse
import os
import shutil
import torch
from ultralytics import settings
from ultralytics import YOLO

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--data', type=str, default='data.yaml') # yaml config file for custom dataset
    parser.add_argument('--epochs', type=int, default=3) # number of training epochs
    parser.add_argument('--batch', type=int, default=-1) # batch size, -1 for AutoBatch
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt') # pretrained base model
    parser.add_argument('--saved-model-weights', type=str, default='model.pt') # saved model weights
    
    # data and model directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    # obtain arguments
    args, _ = parser.parse_known_args()

    # build model
    model = YOLO(args.yolo_model)

    # train model
    results = model.train(data=args.data, epochs=args.epochs, batch=args.batch)
    
    # define soure/dest paths for best model weights
    path_best_model_source = f"{settings['runs_dir']}/detect/train/weights/best.pt"
    path_best_model_dest = os.path.join(args.model_dir, args.saved_model_weights)
    
    # copy best model weights file for packaging
    shutil.copy(path_best_model_source, path_best_model_dest)
    
    # evaluate model on test dataset
    print('EVALUATING MODEL ON TEST DATASET...')
    model_val = YOLO(path_best_model_dest)
    metrics = model_val.val(data=args.data, split='test')
    
    # print evaluation metric to compare performance between models
    print('-------------')
    print('-------------')
    print('-------------')
    print('MODEL EVALUATION METRIC:')
    print('mAP50:', round(metrics.box.map50, 4))
    print('-------------')
    print('-------------')
    print('-------------')