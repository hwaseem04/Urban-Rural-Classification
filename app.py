import streamlit as st
import torch
from torchvision.models import inception_v3
from torch.optim import  Adam
import torch.nn as nn

st.title('Urban or Rural Scene Classification using Transfer learning')

st.write("Due to the unavailability of huge data, I used transfer learning with Inception-V3.")
st.write("For this problem even simpler pretrained networks would give pretty decent results")

checkpoint = torch.load('inception_finetuned_model.pt')

