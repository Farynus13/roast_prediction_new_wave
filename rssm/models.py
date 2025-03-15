import torch
import torch.nn as nn

class EncoderMLP(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 256):
        super(EncoderMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, embedding_dim // 4)
        self.fc2 = nn.Linear(embedding_dim // 4, embedding_dim // 2)
        self.fc3 = nn.Linear(embedding_dim // 2, embedding_dim)

        self.ln1 = nn.LayerNorm(embedding_dim // 4)
        self.ln2 = nn.LayerNorm(embedding_dim // 2)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))  # Tanh for bounded latent space
        return x


class DecoderMLP(nn.Module):
    def __init__(self, hidden_size: int, state_size: int, output_dim: int, embedding_size: int):
        super(DecoderMLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size // 2)
        self.fc3 = nn.Linear(embedding_size // 2, embedding_size // 4)
        self.fc4 = nn.Linear(embedding_size // 4, output_dim)

        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size // 2)
        self.ln3 = nn.LayerNorm(embedding_size // 4)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, s: torch.Tensor):
        assert h.shape[-1] == s.shape[-1], f"Hidden and state sizes must match! Got {h.shape[-1]} vs {s.shape[-1]}"
        x = torch.cat([h, s], dim=-1)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.ln3(self.fc3(x)))
        x = self.fc4(x)  # No activation to allow flexibility in output
        return x


if __name__ == "__main__":
    embedding_dim = 256
    encoder = EncoderMLP(input_dim=2, embedding_dim=embedding_dim)
    decoder = DecoderMLP(hidden_size=embedding_dim, state_size=embedding_dim, output_dim=2, embedding_size=embedding_dim)

    print(encoder)
    print(decoder)

    x = torch.randn(1, 2)
    s = encoder(x)
    h = torch.randn(1, embedding_dim)
    out = decoder(h, s)
    print(out.shape)
