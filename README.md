# Vehicle Detection in Adverse Weather

**CSC490 (Machine Learning for Vision) - University of Toronto**

## Overview

This project investigates vehicle detection under adverse weather conditions using YOLOv8 and the DAWN (Detection in Adverse Weather Nature) dataset.

Our work focuses on addressing class imbalance in adverse-weather vehicle detection. Since some object categories appear significantly more frequently than others, a model may perform better on common classes while struggling to detect underrepresented categories.

To address this challenge, we investigate a class-weighted classification loss for YOLOv8, assigning greater importance to underrepresented classes during training.

## Team

* Madeeha Khan
* Mahak Mishra
* Samaah Abdullah

## Problem

Object detection becomes more challenging under adverse weather conditions such as fog, rain, snow, and sandstorms. Reduced visibility, poor image quality, and occlusion can make objects harder to detect.

In addition to these challenges, the DAWN dataset contains an imbalanced distribution of object categories. Common classes, particularly cars, account for a large proportion of the annotations, while categories such as bicycles and motorcycles are underrepresented.

Our project investigates whether modifying YOLOv8's classification loss can improve detection performance on these less frequent classes.

## Baseline

Our baseline model is YOLOv8, a one-stage object detection architecture that predicts object bounding boxes, class labels, and confidence scores. We use pretrained YOLOv8 weights and fine-tune the model on the DAWN dataset.

**Framework**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

**Documentation**: [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/)

## Dataset

We use the DAWN (Detection in Adverse Weather Nature) dataset, which contains real-world road images captured under different adverse weather conditions.

The dataset includes four weather conditions: fog, rain, snow, and sand.

It contains annotated object categories such as cars, trucks, buses, bicycles, motorcycles, and people.

**Dataset:** [DAWN — Detection in Adverse Weather Nature](https://www.kaggle.com/code/aaditya2002/dawn-dataset-yolov8)

## Proposed Improvement: Class-Weighted Classification Loss

Our main extension modifies YOLOv8's classification loss to account for the imbalance in object category frequencies.

### 1. Class Distribution Analysis
- Analyze the frequency of each object category in the training dataset.
- Identify underrepresented classes.
- Calculate class weights based on category frequencies.

### 2. Modified Classification Loss
- Incorporate class weights into YOLOv8's classification loss.
- Assign greater weights to underrepresented classes and lower weights to more frequent classes.
- Investigate whether this encourages the model to learn better representations of less frequent categories.

### 3. Model Training

We compare the original YOLOv8 baseline against the modified class-weighted model using consistent dataset splits and comparable training configurations.

## Evaluation

We evaluate model performance using standard object detection metrics:

- **Precision**: Proportion of predicted detections that are correct.
- **Recall**: Proportion of ground-truth objects successfully detected.
- **mAP@50**: Mean Average Precision at an IoU threshold of 0.50.
- **mAP@50–95**: Mean Average Precision across IoU thresholds from 0.50 to 0.95.
- **Per-class AP**: Detection performance for individual object categories.

Our comparisons focus on:
- Original YOLOv8 baseline
- YOLOv8 with class-weighted classification loss

We pay particular attention to performance changes in underrepresented categories to determine whether class weighting improves detection without significantly compromising performance on common classes.

## Repository Structure

The repository will contain the implementation of the baseline model, our proposed modifications, experiment configurations, and documentation.

```text
dawn-yolov8-adverse-weather-detection/
│
├── README.md
├── notebooks/
│   ├── yolov8_baseline_and_weighted_loss.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── loss.py
│   ├── train.py
│   └── evaluate.py
│
├── app/
│   └── app.py
│
├── configs/
│   └── data.yaml
│
├── results/
│   ├── metrics/
│   ├── plots/
│   └── predictions/
│
├── requirements.txt
└── .gitignore
```

