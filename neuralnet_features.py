from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# Load the pretrained model
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))

# Set model to evaluation model
model.eval()

# Image transforms
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_feature_vec(img):
    if isinstance(img,(str,Path)):
        # 1. Load the image with Pillow library
        input_image = Image.open(str(img))
    else:
        input_image = Image.fromarray(img)
    # 2. Create a PyTorch Variable with the transformed image

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    return output.numpy().squeeze()



