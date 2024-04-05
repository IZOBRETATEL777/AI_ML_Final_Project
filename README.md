# AI & Machine Learning Fundamentals
The final project for the AI & Machine Learning Fundamentals is a simple image classifier using [Glasses Classification Dataset](https://www.kaggle.com/datasets/ashfakyeafi/glasses-classification-dataset) from Kaggle to classify images of people wearing glasses or not.

Project is using PyTocrh and TorchVision to create a simple Convolutional Neural Network (CNN) to classify images. TensorBoard is used to visualize the training process.

Project based on [Ai_ml_template](https://github.com/m1rakram/Ai_ml_template) by Mirakram Aghalarov aka [m1rakram](https://github.com/m1rakram).

Additionally, the `image_cropper.py` script is used to crop images to get rid of class labels and other unnecessary parts of the images (white frame). Output images has size 100x100 pixels.

## Model Configurations
| CNN Model | Optimizer | Learning Rate | Epochs |
|-----------|-----------|---------------|--------|
| ResNet18  | SGD       | 0.001         | 15     |
| ResNet18  | Adam      | 0.001         | 15     |
| VGG16     | SGD       | 0.01          | 12     |
| VGG16     | Adam      | 0.001         | 10     |

Loss function: *CrossEntropyLoss*. All models are not pretrained.


## Results
| Model    | Optimizer | Loss  | Accuracy | F1     | Classwise F1 Score | Classwise Accuracy Score |
|----------|-----------|-------|----------|--------|---------------------|---------------------------|
| ResNet18 | SGD       | 0.7088| 0.9501   | 0.6415 | Class 1: 0.2963 Class 2: 0.6415 | Class 1: 0.46 Class 2: 0.55 |
| ResNet18 | Adam      | 0.4116| 0.9613   | 0.9302 | Class 1: 0.9189 Class 2: 0.9302 | Class 1: 0.65 Class 2: 0.20 |
| VGG16    | SGD       | 0.6951| 0.9506   | 0.5961 | Class 1: 0.0772 Class 2: 0.5961 | Class 1: 0.90 Class 2: 0.20 |
| VGG16    | Adam      | 0.7455| 1.000    | 0.6667 | Class 1: 0.2746 Class 2: 0.3921 | Class 1: 0.72 Class 2: 0.27 |


## Installation

Just run:
```sh
    pip install -r requirements.txt
```

If you are using Python environments,  it is highly recommended to initiate the environment with system-site packages access to use GPU-related technologies without problems (e.g.: CUDA):

```sh
    python3 -m venv env --system-site-packages
```

In the command above, `env` is the name of the environment. You can change it to any name you want.

Then, activate the environment:

- **Linux/macOS**:
```sh
    source env/bin/activate
```

- **Windows**:
```cmd
    env\Scripts\activate
```

## Train the model
To train model, start train file:

```sh
    python3 train.py
```


## TensorBoard
To visualize the training process, run the following command:

```sh
    tensorboard --logdir=runs
```

Then, open the browser and go to `http://localhost:6006/` to see the training process.