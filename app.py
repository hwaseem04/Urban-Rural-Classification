import streamlit as st
import torch.nn as nn
import torch
from torchvision.models import inception_v3
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title('Transfer Learning')
st.write('## Urban or Rural Scene Classification')

st.write("Due to the unavailability of huge data, I used transfer learning with Inception-V3.")
st.write("For this problem even simpler pretrained networks would give pretty decent results.")

## Load pretrained model weights for inception_v3 
model = inception_v3(weights="IMAGENET1K_V1")

## Modify final unit and auxilary unit's output size to 1
num_ftrs = model.AuxLogits.fc.in_features
model.AuxLogits.fc = nn.Linear(num_ftrs, 1)

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1)


## Load Finetuned Model weights
checkpoint = torch.load('inception_finetuned_model.pt')

## Set finetuned model weights to model
model.load_state_dict(checkpoint['model_state_dict'])

image = st.file_uploader("Upload Image to classify as Rural/Urban Scene", 
type=['png', 'jpeg', 'jpg'])

## Required transformations for model
transform = transforms.Compose([
    transforms.Resize([342], interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop([299,299]),
    transforms.ToTensor() # # Rescaled to 0-1 aswell
])

## Enabling .eval this will avoid generating Auxilary output
model.eval()

if image != None:
    st.write("## Image you have uploaded")
    im = Image.open(image)
    st.image(im, width=None)
    im = transform(im)
    im = im.unsqueeze(0)
    
    out = torch.sigmoid(model(im)).item()
    label = ['Rural', 'Urban']
    st.write("## Prediction")
    if out >= 0.5:
        st.write(f"* **Rural** (Class-0): {1-out:.4f}") 
        st.write(f"* **Urban** (Class-1): {out:.4f}")
        Prob = [1-out, out]
    else:
        st.write(f"* **Rural** (Class-0): {1-out:.4f}")
        st.write(f"* **Urban** (Class-1): {out:.4f}")
        Prob = [1-out, out]

    chart_data = pd.DataFrame(
        np.array([1-out, out]),
        columns=['Probability']
    )
    st.bar_chart(chart_data)