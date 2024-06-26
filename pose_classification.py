import torch.nn as nn
import numpy as np
import os
import torch
from sklearn.preprocessing import StandardScaler

sign_class = ['Hello', 'Thank you', 'ambulance', 'call', 'doctor', 'hurt', 'road']

def load_checkpoint(model, filename='checkpoints/angle_model.pth'):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded: epoch {epoch}, loss {loss}")
        return model, epoch, loss
    else:
        print(f"No checkpoint found at '{filename}'")
        return model, 0, float('inf')
        
def infer_realtime(model, input):
    model.eval()  # Set the model to evaluation mode
        
    with torch.no_grad():
        output = model(input)
        _, predicted = torch.max(output.data, 1)
    return predicted.numpy()

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def build_model():
    input_size = 30 # X_train.shape[1] #
    num_classes = 7  
    model = MLP(input_size, num_classes)
    # scaler = StandardScaler()
    model, start_epoch, best_loss = load_checkpoint(model)
    return model