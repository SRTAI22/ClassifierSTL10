import io
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as transforms 
from PIL import Image


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        #full layer
        self.fc1 = nn.Linear(64 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

loaded_model = CNN()
loaded_model.load_state_dict(torch.load("cnn_model.pth")) # it takes the loaded dictionary, not the path file itself
loaded_model.eval()


#transform images

def transform_image(image_bytes):
    transform = transforms.Compose(
        [transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor()]
    )
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    outputs = loaded_model(image_tensor)
        # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted