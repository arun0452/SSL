**Self-Supervised Learning for Image Rotation**

This project demonstrates a self-supervised learning (SSL) approach for the pretext task of image rotation. The model is trained using the CIFAR-10 dataset and a ResNet-based architecture. The goal is to classify images that have been rotated by one of four possible rotations (0°, 90°, 180°, or 270°).

**Project Structure**

**The project is divided into several Python modules:**

**utils.py:** Contains utility functions for data preprocessing and image rotation.

**lr_schedule.py**: Defines a learning rate schedule used during training.

**resnet.py**: Contains the ResNet model and necessary helper functions to build it.

**main.py**: The main script that handles data loading, model training, and evaluation.
Setup

**Install Dependencies:** Before running the code, make sure to install the necessary Python libraries. You can install them using pip:
pip install numpy tensorflow keras seaborn matplotlib
Download CIFAR-10 Dataset: The CIFAR-10 dataset will be automatically downloaded by the script when needed. No manual downloading is required.
File Organization: Ensure that the project files are organized as follows:

/project_root
├── main.py
├── resnet.py
├── lr_schedule.py
├── utils.py
└── saved_models/

**How to Run**

To run the project, execute the main.py script. It will load the CIFAR-10 dataset, apply image rotation, train a ResNet model with self-supervised learning (SSL), and save the best model and training history.

**Run the script:**

python main.py

**Model Checkpoints:**

During training, the model checkpoints will be saved in the saved_models directory. The best model based on validation accuracy will be saved with the name Restnetv1_SSL_Rotation.keras.

**Training History:**

The training and validation accuracy and loss history will also be saved in a .pkl file (Restnetv1_SSL_Rotation.keras_history.pkl). You can use this file to visualize the model's performance using the provided seaborn and matplotlib visualizations.

**Results Visualization**

After training, the script will automatically generate accuracy and loss plots for both the training and validation datasets:

Accuracy Plot: Shows how the model's accuracy evolves throughout the training process.

Loss Plot: Displays the loss trends during training and validation.

Learning Rate Schedule

The learning rate schedule used in this project is based on the epoch number. The learning rate is adjusted as follows:

1e-3 for epochs 0-79
1e-2 for epochs 80-119
1e-1 for epochs 120-159
1e-3 for epochs 160-179
5e-4 for epochs 180 and beyond
Notes

**Data Augmentation:** The model uses a simple data augmentation strategy (width/height shift and horizontal flip) to improve generalization.
Self-Supervised Pretext Task: The pretext task is image rotation, where the network must predict the rotation angle (0°, 90°, 180°, or 270°) applied to the image.
Model: The model is a variant of ResNet, which includes residual connections and is specifically designed for image classification tasks.
Troubleshooting

If you encounter any issues with TensorFlow or Keras versions, make sure to use compatible versions. The code was tested with TensorFlow 2.x and Keras 2.x.
References

**ResNet: **He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 770-778.
Self-Supervised Learning: Doersch, C., Gupta, A., & Efros, A. A. (2015). Unsupervised Visual Representation Learning by Context Prediction. IEEE International Conference on Computer Vision (ICCV).
