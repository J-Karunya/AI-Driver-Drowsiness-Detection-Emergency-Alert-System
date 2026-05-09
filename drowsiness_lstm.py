import torch
import torch.nn as nn

class DrowsinessLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=2, num_classes=2):
        """
        input_size: 2 (EAR, MAR)
        """
        super(DrowsinessLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

def get_model(model_path=None, device='cpu'):
    model = DrowsinessLSTM(input_size=2, hidden_size=32, num_layers=2, num_classes=2)
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except:
            print(f"Warning: Could not load model from {model_path}. Using untrained weights.")
    model.to(device)
    model.eval()
    return model
