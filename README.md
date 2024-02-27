# Job Site Safety

![Basic-Personal-Protective-Equipment-PPE-for-Construction-Workers](https://github.com/jerm914/Job-Site-Safety---AAI-540/assets/68162866/905d93c6-2a94-4f8b-a663-32f912a5e517)

This project is a part of the AAI-540 Machine Learning Operations course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

### <b>Project Status: Completed 2/26/2024</b>

## ML System Architecture Diagram
This diagram is a visual representation of the ML system's architecture, illustrating how different elements of the system are connected and how data flows between them.

![image](https://github.com/jerm914/Job-Site-Safety---AAI-540/assets/68162866/05513901-ca83-416c-a212-855e2564e9e7)

## Pipeline Directed Acyclic Graph (DAG)
The graph below represents operations as nodes, with dependencies illustrated as edges. Each node corresponds to a specific operation within the pipeline, while each edge signifies the data flow between tasks.

![image](https://github.com/jerm914/Job-Site-Safety---AAI-540/assets/68162866/4cb0156c-9e8b-40d1-9c2f-225993094898)

## Installation
To create a copy of the repository on your Sagemaker. Navigate to the Terminal then use the following command:
```python
   git clone https://github.com/jerm914/Job-Site-Safety---AAI-540.git
```

## Design Document
[AAI-540 ML Design - Job Site Safety.docx](https://github.com/jerm914/Job-Site-Safety---AAI-540/blob/main/AAI-540%20ML%20Design%20-%20Job%20Site%20Safety.docx)
  - Project Overview
  - Project Background
  - Technical Background
  - Goals vs Non-Goals
  - Solution Overview
  - Data Sources
  - Data Engineering
  - Training Data
  - Feature Engineering
  - Model Training & Evaluation
  - Model Deployment
  - Model Monitoring
  - Model CI/CD
  - Security Checklist, Privacy, and Other Risks
  - Future Enhancements

## Project Intro
Our project is aimed at enhancing safety compliance by focusing on adherence to Personal Protective Equipment (PPE) usage and reducing serious injury occurrences on job sites. Our mission is to develop a model capable of detecting, classifying, and reporting the presence of 'Helmet', 'Vest', and 'Head' in given input images. The detection of 'Head' instances may indicate a safety violation where a required helmet isn't worn. This project tackles an object detection and classification problem in Machine Learning. The model's initial challenge is to sufficiently identify these three object types in images. Subsequently, it will need to accurately classify the detected objects into their correct categories.

In this course, however, our focus is primarily on developing an end-to-end ML system and not the modeling process itself. We build CI/CD pipelines to provide automation to the process in terms of preprocessing, model training, model evaluation, conditionally creating and registering a model based on our metric threshold being met, and performing a batch transform job to output results. The results are then combined, processed, and available for business queries to review trends in PPE compliance.

Instead of using an available algorithm in Amazon SageMaker, we chose to use Script Mode to support our intention of using an Ultralytics YOLOv8 model. This allowed us to use the PyTorch framework and containers while making use of customized scripts to support our needs in terms of preprocessing, training, evaluation, and inference. This also allowed us to use custom requirements and yaml files for our dataset format.

## Project Goals
- Reduced instances of serious injury on construction sites (Business Metric)
- Increased safety compliance (wearing required PPE) (Business Metric)
- Mean Average Precision 50 (mAP50) for model evaluation/selection (Model Metric)
  * Low box loss (specific amount to be determined w/ company) (Model Loss, training)
  * Low classification loss (focus on ‘Helmet’ and ‘Head’) (Model Loss, training)

## Contributors
- [Jason Raimondi](https://github.com/jeraimondi)
- [Jeremy Cryer](https://github.com/jerm914)
- [Maimuna Bashir](https://github.com/maymoonah-bash)

## Methods Used
- Computer Vision
- Machine Learning
- Deep Learning

## Technologies
- Amazon Web Services
    * Amazon SageMaker
    * S3
    * Athena
    * AWS Glue
    * CloudWatch
- Python
- PyTorch
- SQL
- Ultralytics (YOLOv8)

## Data Source 
[HardHat-Vest Dataset](https://www.kaggle.com/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3/data)

This dataset contains 23,673 images with bounding box annotations for these 3 classes:
- Helmet
- Vest
- Head

## Acknowledgments
Thank you to all the USD professors. Special thanks to Dr. Christenson for your continued dedication, guidance, and support throughout this course.





