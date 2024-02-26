# Job Site Safety

![Basic-Personal-Protective-Equipment-PPE-for-Construction-Workers](https://github.com/jerm914/Job-Site-Safety---AAI-540/assets/68162866/905d93c6-2a94-4f8b-a663-32f912a5e517)

This project is a part of the AAI-540 Machine Learning Operations course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

**Project Status: Completed 26/2/2024**

## Architecture Diagram
This diagram is a visual representation of the system's architecture, illustrating how different elements of the system are connected and how data flows between them.

![image](https://github.com/jerm914/Job-Site-Safety---AAI-540/assets/68162866/832d9834-1bad-4a90-9ebe-3e7fb1857744)

## Pipeline Directed Acyclic Graph (DAG)
The graph below represents operations as nodes, with dependencies illustrated as edges. Each node corresponds to a specific operation within the pipeline, while each edge signifies the data flow between tasks.

![image](https://github.com/jerm914/Job-Site-Safety---AAI-540/assets/68162866/4cb0156c-9e8b-40d1-9c2f-225993094898)

## Installation
To create a copy of the repository on your Sagemaker. Navigate to the Terminal then use the following command:
```python
   git clone https://github.com/jerm914/Job-Site-Safety---AAI-540.git
```

## Design Document
  **Project Overview ->** [Link to the document](https://docs.google.com/document/d/1obVlE8EgxbMFm6EnmXJaMwGWfN_TBcZQ/edit)
  - Project Background
  - Technical Background
  - Goals vs Non-Goals
  - Data Sources
  - Data Engineering
  - Training Data
  - Feature Engineering
  - Model Training & Evaluation:
  - Model Deployment
  - Model Monitoring
  - Model CI/CD
  - Security Checklist, Privacy, and Other Risks
  - Future Enhancements

## Project Into
Welcome to our project, aimed at enhancing safety compliance by focusing on adherence to Personal Protective Equipment (PPE) usage and reducing serious injury occurrences on job sites. Our mission is to develop a model capable of detecting, classifying, and reporting the presence of 'Helmet', 'Vest', and 'Head' in given input images. The detection of 'Head' instances may indicate a safety violation where a required helmet isn't worn. This project tackles an object detection and classification problem in Machine Learning. The model's initial challenge is to sufficiently identify these three object types in images. Subsequently, it will need to accurately classify the detected objects into their correct categories.

## Project Goals
Goals:
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
- Python
- SQL
- Amazone Web Servace
    * SageMaker
    * CloudWatch
    * S3
    * Athena


## Data Source 
[Safety Helmet Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)

This dataset contains 5000 images with bounding box annotations for these 3 classes:
- Helmet
- Person
- Head

## Acknowledgments
Thank you to all the USD professors. Special thanks to Dr. Christenson for your continued dedication, guidance, and support throughout this course.





