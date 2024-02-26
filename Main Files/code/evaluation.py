import json
import pathlib
import pickle
import tarfile
import argparse
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='data-eval.yaml') # yaml config file for custom dataset

    # obtain arguments
    args, _ = parser.parse_known_args()
    
    model_path = '/opt/ml/processing/model/model.tar.gz'
    with tarfile.open(model_path) as tar:
        tar.extractall(path="/opt/ml/processing/model/")
    
    # evaluate model on test dataset
    print('EVALUATING MODEL ON TEST DATASET...')
    model_val = YOLO('/opt/ml/processing/model/model.pt')
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
    
    
    map50 = round(metrics.box.map50, 4)

    report_dict = {
        "detection_metrics": {
            "mAP50": {"value": map50},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
