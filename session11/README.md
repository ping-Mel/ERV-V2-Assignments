# CIFAR-10 Training with ResNet18 and GradCAM Visualization

This repository contains the implementation of training a ResNet18 model on the CIFAR-10 dataset using PyTorch, complete with GradCAM visualizations to highlight misclassified images. The project is structured to ensure that all core functionalities are modularized in separate Python scripts, allowing the Google Colab notebook to remain clean and only contain calls to these modules.

## Repository Structure

The project is organized into several directories, each serving a specific function:

- `models/`: Contains the implementation of the ResNet18 model.
- `utils/`: Includes utility functions for data loading, transformations, and training/testing procedures.
- `config/`: Contains configuration files or scripts setting global parameters for the project.
- `notebooks/`: This directory contains the Google Colab notebook that executes the training and visualization by importing functions from other directories.

### Key Components

- **ResNet18 Implementation**: Defined under `models/resnet.py`.
- **Data Loader and Transformations**: Setup is managed by `utils/dataloader.py`, which includes the specific transformations required by the assignment.
- **Training and Testing Logic**: Handled by `utils/train_test.py`.
- **GradCAM Implementation**: Located in `utils/gradcam.py`, tailored to handle visualization on convolutional layers of ResNet18.
- **Visualization Tools**: Script for displaying misclassified images and their GradCAM outputs is in `utils/visualizations.py`.



## Outputs

After running the notebook, you can expect the following outputs:

- **Training and Validation Loss Curves**: Plots showing the progression of loss during training and validation.
- **Gallery of Misclassified Images**: Visual representation of ten misclassified images from the test set.
- **GradCAM Visualizations**: Heatmaps overlayed on the same ten misclassified images to indicate areas of focus by the model.

## Contributing

Contributions to this project are welcome. Please ensure that any pull requests or changes adhere to the existing project structure and coding style.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
