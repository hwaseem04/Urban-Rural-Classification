# Urban-Rural-Classification

## Why?
Wanted to try out Streamlit. So I decided to implement a Rural-Urban Scene Classifier. Train and validation data downloaded from [Kaggle](https://www.kaggle.com/datasets/dansbecker/urban-and-rural-photos). Manually downloaded test data from [pexels.com](https://www.pexels.com/).

## How to use
* Refer to this demo: 

![demo video](demo.gif)

* You can use data that I have in this folder by cloning this repo, or you can manually use data from internet.

* **Test with data that are in similar grounds of training data**.
* **Upload only rural/urban scene images**.

## Implementation
* Picked up `Inception-v3` Architecture.
* Finetuned it with urban-rural scene images by modifying the final classfifier layer.
* Impressively, even after few epoch, able to get both train and validation accuracy to be 100%. Quite overfitted, will have to add more noise in augmentation step. But so far it predicts well.


## Improvements
* Incase certain images are misclassified please raise an issue along with the image.
