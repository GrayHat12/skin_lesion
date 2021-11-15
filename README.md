# Skin Lesion

## Introduction

The 7 classes of skin cancer lesions included in this dataset are:
* Melanocytic nevi (nv)
* Melanoma (mel)
* Benign keratosis-like lesions (bkl)
* Basal cell carcinoma (bcc) 
* Actinic keratoses (akiec)
* Vascular lesions (vas)
* Dermatofibroma (df)

---

## Data Preparation

### Initial Data

[(code)](./analysis.py)

* Age

![age](./plots/before_balancing/age.png)

* Classes

![classes](./plots/before_balancing/cell_type.png)

* Localization

![localization](./plots/before_balancing/localization.png)

* Gender

![Gender](./plots/before_balancing/sex.png)

### Balancing

[(code)](./prepare_data.py)

We balanced data by keeping 1000 images for every class. Upscaled classes with less than 1000 images and Downscaled the ones with more.

---

## Models

### ANN - Type 1

[(code)](./ann_with_overfitting.py)

* Architecture:

![ANN](./plots/ann_with_overfitting_architecture.png)

* History:

![ANN](./plots/ann_with_overfitting_history.png)

* Confusion Matrix:

![ANN](./plots/ann_with_overfitting_confusion_matrix.png)

* Fractional Incorrect Misclassifications:

![ANN](./plots/ann_with_overfitting_fractional_incorrect_misclassifications.png)


### ANN - Type 2

[(code)](./ann_no_overfitting.py)

* Architecture:

![ANN](./plots/ann_no_overfitting_architecture.png)

* History:

![ANN](./plots/ann_no_overfitting_history.png)

* Confusion Matrix:

![ANN](./plots/ann_no_overfitting_confusion_matrix.png)

* Fractional Incorrect Misclassifications:

![ANN](./plots/ann_no_overfitting_fractional_incorrect_misclassifications.png)

### CNN

[(code)](./cnn.py)

* Architecture:

![CNN](./plots/cnn_architecture.png)

* History:

![CNN](./plots/cnn_history.png)

* Confusion Matrix:

![CNN](./plots/cnn_confusion_matrix.png)

* Fractional Incorrect Misclassifications:

![CNN](./plots/cnn_fractional_incorrect_misclassifications.png)


### Transfer Learning (Mobilenet)

[(code)](./transfer_learning.py)

* Architecture:

![TL](./plots/transfer_learning_architecture.png)

* History:

![TL](./plots/transfer_learning_history.png)

* Confusion Matrix:

![TL](./plots/transfer_learning_confusion_matrix.png)

* Fractional Incorrect Misclassifications:

![TL](./plots/transfer_learning_fractional_incorrect_misclassifications.png)