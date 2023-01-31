# Urban-Rural-Classification
Streamlit App Link: [click here](https://hwaseem04-urban-rural-classification-app-urj25b.streamlit.app/)

## Introduction
Wanted to try out Streamlit. So I decided to implement a Rural-Urban Scene Classifier. Train and validation data downloaded from [Kaggle](https://www.kaggle.com/datasets/dansbecker/urban-and-rural-photos). Manually downloaded test data from [pexels.com](https://www.pexels.com/).

## How to use
* Refer to the demo [in youtube](https://youtu.be/vcRnDZyhBuY).

![demo video](demo.gif)

* You can use data that I have in this folder by cloning this repo, or you can manually use data from internet.

* **Test with data that are in similar grounds of training data**.
* **Upload only rural/urban scene images**.

## Implementation
* Picked up `Inception-v3` Architecture for transfer learning.
* Finetuned it with urban-rural scene images by modifying the final classifier layer.
* Impressively, even after few epoch, able to get both train and validation accuracy to be 100%. Quite overfitted, will have to add more noise in augmentation step. But so far it predicts well.
* You can refer to my [Jupyter file](Urban-Rural-Scene-classification.ipynb) for implementation of transfer learning.

## Improvements
* Incase certain images are misclassified please raise an issue along with the image.
