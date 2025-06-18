import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super(PredictHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.elu1 = nn.ELU(inplace=True)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.elu2 = nn.ELU(inplace=True)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.elu3 = nn.ELU(inplace=True)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(hidden_dim // 4, num_classes)
        
        
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.elu3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        
        return x