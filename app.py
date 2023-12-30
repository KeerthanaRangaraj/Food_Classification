
from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision.models as models
import torch.nn as nn
import cv2

import os
app = Flask(__name__, static_url_path='/static')


# Load the pre-trained ResNet50 model with modifications
model = models.resnet50(pretrained=True)

# Freeze layers except the last few
for param in model.parameters():
    param.requires_grad = False

# Modify the last few layers for your classification task with a more complex structure
num_classes = 3  # Replace with the actual number of classes
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

# Load your pre-trained weights
model.load_state_dict(torch.load(r"basic_resnet2_food.pth", map_location=torch.device('cpu')))
model.eval()

# Define the transformation to be applied to the uploaded image
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (96, 96))
    img = img.astype(np.float32) / 255.0  # Normalize
    return img

def predict_image(image_path):
    # Preprocess the image
    preprocessed_img = preprocess_image(image_path)

    # Convert to PyTorch tensor and add batch dimension
    image_tensor = transform(Image.fromarray((preprocessed_img * 255).astype(np.uint8)))
    image_tensor = image_tensor.unsqueeze(0)

    # Make the prediction
    with torch.no_grad():
        output = model(image_tensor)

    # Process the output to get the class probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)

    # Return class labels and probabilities
    return probabilities

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    probabilities = None
    uploaded_image = None

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            error = 'No file part'
        else:
            file = request.files['file']

            # Check if the file has a name
            if file.filename == '':
                error = 'No selected file'
            else:
                # Check if the file is allowed
                if file:
                    # Save the uploaded file temporarily
                    file_path = os.path.join('\static', 'temp_image.jpg')
                    print(file_path)
                    file.save(file_path)

                    # Get predictions
                    probabilities = predict_image(file_path)

                    # Move the file to the 'static' folder
                    new_file_path = os.path.join('\static', 'temp_image.jpg')

                    os.rename(file_path, new_file_path)
                    print("New file_path:", new_file_path)


                    # Get predictions
                    probabilities = predict_image(file_path)

                    # Pass the uploaded image path for display
                    uploaded_image = file_path
                    print(uploaded_image)
                 
    return render_template(r"templates/index.html", error=error, probabilities=probabilities, uploaded_image=uploaded_image)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


