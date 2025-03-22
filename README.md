# Pneumonia Detection with Explainable AI: An Internship Task

This repository contains the code and documents produced as part of an internship project focused on pneumonia detection in chest X-ray images. The project explores a Convolutional Neural Network (CNN) model with explainable AI (XAI) techniques, including a Hybrid Explanation Fusion Network (HEFNet), for better understanding the model's decisions.

## Project Overview

The goal of this project was to develop a deep learning model capable of accurately detecting pneumonia from chest X-ray images, and to then implement and explore explainable AI methods to understand and visualize the model's decision-making process. The **HEFNet** component aims to combine the individual XAI insights into a single, more clinically relevant interpretation. This helps in building trust in AI-based diagnostic tools.

## Repository Contents

*   `cnn-xai.ipynb`:  Jupyter Notebook containing the CNN model, training, evaluation, standard XAI implementations (Grad-CAM, LIME, Occlusion Sensitivity), and the HEFNet fusion.
*   `Case Study Document.docx`: Document containing the Problem Statement, Objectives, Data Preprocessing details, Model Development, Visualizations, Insights, and Recommendations.
*   `Literature Review.docx`: Document providing a review of relevant literature and identification of research gaps.
*   `pneumonia_model_final.keras`: Saved Keras model file.
*   `README.md`: This file (project documentation).

## Model Details

The model is a CNN built using TensorFlow/Keras. The architecture leverages transfer learning from a pre-trained VGG16 model. This approach helps to overcome data scarcity in the medical imaging domain.

### Key Model Components:

*   **Base Model:** VGG16 (pre-trained on ImageNet) with frozen initial layers.
*   **Custom Layers:** Global Average Pooling, Dense layers, and Dropout for regularization.
*   **Loss Function:** Binary Crossentropy.
*   **Optimizer:** Adam.
*   **Metrics:** Accuracy, AUC.

## Explainable AI Techniques

The project implements the following XAI methods to interpret the model's predictions:

*   **Grad-CAM:** Highlights the regions of the X-ray that most influenced the prediction.
*   **LIME:** Identifies superpixels that contributed to the classification decision.
*   **Occlusion Sensitivity:** Measures the impact of occluding different parts of the image.
*   **HEFNet (Hybrid Explanation Fusion Network):** A novel fusion approach that combines the Grad-CAM, LIME, and Occlusion Sensitivity maps into a single, more comprehensive explanation. The HEFNet uses weighted averaging to prioritize regions deemed most clinically relevant.

## Code Details

### `cnn-xai.ipynb`

* **Model Training:** Implements a CNN model based on VGG16 architecture and trains the model
* **XAI Methods:** Implements the Grad-CAM, LIME, Occlusion Sensitivitiy and the HEFNet model.
* **Performance Metrics:** AUC, accuracy, sensitivity, specificity

## Results and Visualizations

The `cnn-xai.ipynb` notebook generates visualizations for the XAI techniques, including the HEFNet output. The results are also summarized in the `Case Study Document.docx`.

## Relevant Research Paper

This project draws upon concepts from the following research paper:

*   "Explainable Transfer Learning for Medical Image Analysis" (2023 International Conference on Computer Engineering and Distance Learning (CEDL), DOI: 10.1109/CEDL60560.2023.00029)


## Project Structure

This repository is a result of an internship. Feel free to explore and build upon this code. For any further query please make sure to connect me on social media.

![roc_curve](https://github.com/user-attachments/assets/5330a15a-6c7d-4df5-b081-c6cc176aee6e)
![occlusion_sensitivity_pneumonia](https://github.com/user-attachments/assets/360e6432-c35a-4980-9bac-e949e60307fb)
![occlusion_sensitivity_normal](https://github.com/user-attachments/assets/629c72e6-7d58-4693-91a0-59c3054636c1)
![lime_visualization](https://github.com/user-attachments/assets/03028932-17f9-4e6d-9bd9-c7a622bfb1fc)
![gradcam_visualization](https://github.com/user-attachments/assets/c5403bf8-a782-4737-afd8-ccabdf0e10c0)
![confusion_matrix](https://github.com/user-attachments/assets/6943bfff-66b2-43a1-8718-85726a8c54f1)
![training_history](https://github.com/user-attachments/assets/10b4b490-dec1-4bed-aa82-9078c7bf6fe3)
