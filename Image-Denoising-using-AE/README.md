# Image Denoising using AE

## Mnist Dataset
### Original Dataset
![Dataset](mnist_dataset/clean_data.png)

### Noisy Dataset
![Noisy_Dataset](mnist_dataset/noise_data.png)

|Model|Loss Graph|
|----------|-----------|
|![Model](mnist_dataset/model.png)|![Loss Graph](mnist_dataset/loss.png)|

### Model Denoising Predictions
![Denoising](mnist_dataset/predictions.png)

## RGB Dataset with RGB Salt and Peper Noise (amount=0.3)
### Clean and Nosiy Dataset
![Dataset](RGB_dataset_salt_and_pepper_nosie/dataset.png)

|Autoencoder Model|CNN Model|
|------------|-------------|
|![Model](RGB_dataset_salt_and_pepper_nosie/AE_model.png)|![CNNModel](RGB_dataset_salt_and_pepper_nosie/CNN_model.png)|
|![AELoss Graph](RGB_dataset_salt_and_pepper_nosie/aeloss.png)|![CNNLoss Graph](RGB_dataset_salt_and_pepper_nosie/cnnloss.png)|
|![AEAcc Graph](RGB_dataset_salt_and_pepper_nosie/aeacc.png)|![CNNAcc Graph](RGB_dataset_salt_and_pepper_nosie/cnnacc.png)|
|![AELr Graph](RGB_dataset_salt_and_pepper_nosie/aelr.png)|![CNNLr Graph](RGB_dataset_salt_and_pepper_nosie/cnnlr.png)|

### AE and CNN Model Denoising Predictions
![RGBDenoising](RGB_dataset_salt_and_pepper_nosie/predictions.png)

## RGB Dataset with RGB Gaussian Noise (var=0.05)
### Clean and Nosiy Dataset
![Dataset](RGB_dataset_gaussian_noise/dataset.png)

|Autoencoder Model|CNN Model|
|------------|-------------|
|![AELoss Graph](RGB_dataset_gaussian_noise/aeloss.png)|![CNNLoss Graph](RGB_dataset_gaussian_noise/cnnloss.png)|
|![AEAcc Graph](RGB_dataset_gaussian_noise/aeacc.png)|![CNNAcc Graph](RGB_dataset_gaussian_noise/cnnacc.png)|
|![AELr Graph](RGB_dataset_gaussian_noise/aelr.png)|![CNNLr Graph](RGB_dataset_gaussian_noise/cnnlr.png)|

### AE and CNN Model Denoising Predictions
![RGBDenoising](RGB_dataset_gaussian_noise/predictions.png)

## RGB Dataset with Random Noise
### Clean and Nosiy Dataset
![Dataset](RGB_dataset_random_noise/dataset.png)

|Autoencoder Model|CNN Model|
|------------|-------------|
|![AELoss Graph](RGB_dataset_random_noise/aeloss.png)|![CNNLoss Graph](RGB_dataset_random_noise/cnnloss.png)|
|![AEAcc Graph](RGB_dataset_random_noise/aeacc.png)|![CNNAcc Graph](RGB_dataset_random_noise/cnnacc.png)|
|![AELr Graph](RGB_dataset_random_noise/aelr.png)|![CNNLr Graph](RGB_dataset_random_noise/cnnlr.png)|

### AE and CNN Model Denoising Predictions
![RGBDenoising](RGB_dataset_random_noise/predictions.png)
