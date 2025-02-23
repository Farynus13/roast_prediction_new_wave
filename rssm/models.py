import torch
import torch.nn as nn

class EncoderMLP(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 256):
        super(EncoderMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, embedding_dim//4)
        self.fc2 = nn.Linear(embedding_dim//4, embedding_dim//2)
        self.fc3 = nn.Linear(embedding_dim//2, embedding_dim)

        self.bn1 = nn.BatchNorm1d(embedding_dim//4)
        self.bn2 = nn.BatchNorm1d(embedding_dim//2)

    def forward(self, x):
        # x = torch.flatten(x, start_dim=1)  # Flatten input
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)  # No activation on the last layer
        return x


class DecoderMLP(nn.Module):
    def __init__(self, hidden_size: int, state_size: int, output_dim: int, embedding_size: int):
        super(DecoderMLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size//2) 
        self.fc3 = nn.Linear(embedding_size//2, embedding_size//4)
        self.fc4 = nn.Linear(embedding_size//4, output_dim)

        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size//2)
        self.bn3 = nn.BatchNorm1d(embedding_size//4)

    def forward(self, h: torch.Tensor, s: torch.Tensor):
        x = torch.cat([h, s], dim=-1)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # No activation to allow flexibility in output
        return x
    
    def init_hidden(self, batch_size: int, hidden_dim: int):
        return torch.zeros(batch_size, hidden_dim, device=next(self.parameters()).device)
        



if __name__ == "__main__":
    embedding_dim = 256
    encoder = EncoderMLP( input_dim = 2, embedding_dim = embedding_dim)
    decoder = DecoderMLP(hidden_size = embedding_dim, state_size = embedding_dim, output_dim = 2, embedding_size = embedding_dim)
    print(encoder)
    print(decoder)

    x = torch.randn(1, 2)
    s = encoder(x)
    h = torch.randn(1, embedding_dim)
    out = decoder(h, s)
    print(out.shape)