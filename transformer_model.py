import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from tqdm import tqdm
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.2) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert n_heads * self.head_dim == d_model, "d_model must be divisible by n_heads."

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, p_sampling: float) -> torch.Tensor:
        """
        Forward pass with scheduled sampling for p_sampling > 0.
        Falls back to the original implementation if p_sampling = 0.

        Args:
            inputs (torch.Tensor): Shape (B, seq_length, d_model).
            p_sampling (float): Probability of using the predicted output instead of ground truth.

        Returns:
            torch.Tensor: The output tensor after the attention mechanism.
        """
        B, seq_length, d_model = inputs.shape

        # Check if p_sampling is 0 to use the original forward method
        if p_sampling == 0:
            # Parallel attention computation for the entire sequence
            Q = self.query(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            K = self.key(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            V = self.value(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Masking for causal attention
            mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(inputs.device)
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))

            attention_weights = torch.softmax(attention_scores, dim=-1)
            attention_output = torch.matmul(self.dropout(attention_weights), V)

            # Combine heads and reshape back to the original dimension
            attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
            attention_output = attention_output.view(B, seq_length, d_model)

            return self.fc_out(attention_output)

        # Scheduled sampling (p_sampling > 0)
        Q = self.query(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Initialize outputs
        outputs = []
        context = torch.zeros_like(inputs[:, 0, :])  # (B, d_model)

        for t in range(seq_length):
            # Compute attention scores
            q = Q[:, :, t, :].unsqueeze(2)
            k = K[:, :, :t+1, :]
            v = V[:, :, :t+1, :]
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Masking for causal attention
            mask = torch.triu(torch.ones(1, t+1), diagonal=1).bool().to(inputs.device)
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))

            attention_weights = torch.softmax(attention_scores, dim=-1)
            attention_output = torch.matmul(self.dropout(attention_weights), v)

            # Combine heads and reshape back to the original dimension
            attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
            attention_output = attention_output.view(B, 1, self.n_heads * self.head_dim)

            # Apply dropout and linear layer
            attention_output = self.fc_out(attention_output)
            outputs.append(attention_output)

            # Scheduled sampling
            context = torch.where(torch.rand(B, 1).to(inputs.device) < p_sampling, attention_output.squeeze(1), context)

        # Concatenate outputs across the sequence
        outputs = torch.cat(outputs, dim=1)  # (B, seq_length, d_model)

        return outputs




class PositionalEncoding(nn.Module):
    def __init__(self, context_length, d_model) -> None:
        super().__init__()
        # przygotowujemy macierz of wymiarze (context_length, d_model) by przechowywać positional encodings
        pe = torch.zeros(context_length, d_model)

        # Tworzymy wektor pozycji [0, 1, 2, ..., context_length-1] o wymiarze (context_length, 1)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)

        # Tworzymy wektor z dzielnikami w oparciu o wymiar
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))


        # Obliczamy enkodowanie pozycyjne za pomocą sinusów i cosinusów
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, context_length, d_model)

        # Rejestrujemy pe jako buffer na potrzeby uwzględnienia w state dict modułu
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dodajemy positional encoding do zembedowanych wejść
        return x + self.pe[:,:x.size(1), :]


class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()
        self.att = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fcn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, logits, p_sampling=0.0):
        att_logits = self.att(logits,p_sampling)
        adn_logits = self.ln1(logits + att_logits)
        logits = self.dropout(adn_logits)
        logits = self.fcn(logits)
        logits = self.ln2(logits + adn_logits)
        return logits
    
class CoffeeRoasterModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim,max_length=800,device='cpu',dropout=0.2):
        super(CoffeeRoasterModel, self).__init__()
        self.device = device
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_length = max_length
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(max_length, d_model)  # Max length: 800
        self.blocks = nn.ModuleList([GPTBlock(d_model, nhead,dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, output_dim)
    
    def forward(self, src, targets=None,masks=None,p_sampling=0.0):
        logits = self.embedding(src)
        logits = self.positional_encoding(logits)
        for block in self.blocks:
            logits = block(logits, p_sampling)
        logits = self.fc(logits)
        loss = None
        if targets is not None and masks is not None:
            batch_size, sequence_length, d_model = logits.shape
            logits = logits.view(batch_size * sequence_length, d_model)
            targets = targets.view(batch_size * sequence_length, self.output_dim)
            masks = masks.view(batch_size * sequence_length)
            loss = F.mse_loss(logits[masks.bool()], targets[masks.bool()])
        return logits.view(batch_size, sequence_length, d_model), loss
    
    def generate(self, src, exogenous):
        with torch.no_grad():
            # empty tensor to store new states
            batch_size, _, _ = src.shape
            output = torch.zeros(batch_size, 0, self.input_dim, device=self.device)
            assert exogenous.size(1) + src.size(1) <= self.max_length, "Exogenous sequence too long"
            for i in tqdm(range(exogenous.size(1)), desc="Generating predictions", total=exogenous.size(1)):
                logits, _ = self(src)
                logits = logits[:, -1, :]
                e = exogenous[:, i, :]
                new_state = torch.cat([logits, e], dim=1)
                new_state = new_state.unsqueeze(1)
                output = torch.cat([output, new_state], dim=1)
                src = torch.cat([src, new_state], dim=1)
        return output
            

class DataLoader:
    def __init__(self, sequences, batch_size=1, device='cpu'):
        self.sequences = sequences
        self.batch_size = batch_size
        self.device = device
        self.indices = np.arange(len(sequences))
        np.random.shuffle(self.indices)
        self.idx = 0

    def __len__(self):
        return len(self.sequences) // self.batch_size
    
    def reset(self):
        self.idx = 0
        np.random.shuffle(self.indices)

    def get_batch(self):
        if self.idx + self.batch_size > len(self.sequences):
            self.reset()
        
        batch_indices = self.indices[self.idx:self.idx + self.batch_size]
        batch_sequences = [self.sequences[i] for i in batch_indices]
        self.idx += self.batch_size

        # Find the maximum sequence length in the batch
        max_len = max(len(seq) for seq in batch_sequences)

        x_batch = []
        y_batch = []
        mask_batch = []

        for seq in batch_sequences:
            x = seq[:-1]  # All but the last time step
            y = seq[1:, :2]  # All but the first time step, only bt and et

            # Pad sequences to the maximum length
            x_padded = np.pad(x, ((0, max_len - len(x)), (0, 0)), mode='constant', constant_values=0)
            y_padded = np.pad(y, ((0, max_len - len(y)), (0, 0)), mode='constant', constant_values=0)

            # Create mask for the sequence
            mask = np.zeros((max_len,), dtype=np.float32)
            mask[:len(x)] = 1

            x_batch.append(x_padded)
            y_batch.append(y_padded)
            mask_batch.append(mask)

        x_batch = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
        y_batch = torch.tensor(y_batch, dtype=torch.float32, device=self.device)
        mask_batch = torch.tensor(mask_batch, dtype=torch.float32, device=self.device)

        return x_batch, y_batch, mask_batch


def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=20,device='cpu'):
    k = 100

    for epoch in range(num_epochs):
        p_sampling = (k / (k + np.exp(epoch/num_epochs*1000 / k)))
        model.train()
        total_loss = 0
        batches = np.arange(len(train_dataloader))
        for _ in tqdm(batches, desc=f"Epoch {epoch + 1}", total=len(train_dataloader)):
            xb, yb,mb= train_dataloader.get_batch()
            logits, loss = model(xb, yb,mb,p_sampling)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for _ in tqdm(range(len(val_dataloader)), desc=f"Val Epoch {epoch + 1}", total=len(val_dataloader)):
                xb, yb,mb = val_dataloader.get_batch()
                logits, loss = model(xb, yb,mb)
                total_val_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}, Val Loss: {total_val_loss / len(val_dataloader)}")

# Scaler function
def get_scaler(data, type='MinMaxScaler', save=True, path='scaler.joblib'):
    data_concat = [point for roast in data for point in roast]
    data_concat = pd.DataFrame(data_concat, columns=['bt', 'et', 'burner'])
    data_concat.dropna(inplace=True)

    X = data_concat[['bt', 'et', 'burner']]
    scaler = MinMaxScaler() if type == 'MinMaxScaler' else StandardScaler()
    scaler.fit(X)

    if save:
        joblib.dump(scaler, path)
    return scaler

# Generate predictions
def generate_prediction(model, sequences,scaler, start_idx=120, pred_len=120,device='cpu'):
    src = [seq[:start_idx] for seq in sequences]
    src = torch.tensor(src, dtype=torch.float32, device=device)
    exogenous = [seq[start_idx:start_idx + pred_len, 2] for seq in sequences]
    exogenous = torch.tensor(exogenous, dtype=torch.float32, device=device)
    exogenous = exogenous.unsqueeze(2)
    predictions = model.generate(src, exogenous)
    #get loss
    y_true = [seq[start_idx:start_idx + pred_len] for seq in sequences]
    y_true = torch.tensor(y_true, dtype=torch.float32, device=device)
    y_pred = predictions.clone()
    loss = F.mse_loss(y_pred[:,:,:2], y_true[:,:,:2])
    y_pred_unscaled = y_pred.detach().cpu().numpy()
    y_true_unscaled = y_true.detach().cpu().numpy()
    for i in range(len(y_pred_unscaled)):
        y_pred_unscaled[i] = scaler.inverse_transform(y_pred_unscaled[i])
        y_true_unscaled[i] = scaler.inverse_transform(y_true_unscaled[i])
    y_pred = torch.tensor(y_pred_unscaled, device=device)
    y_true = torch.tensor(y_true_unscaled, device=device)
    unscaled_loss = F.mse_loss(y_pred[:,:,:2], y_true[:,:,:2])

    return predictions.detach().cpu().numpy(), loss.item(), unscaled_loss.item()

def plot_predictions(sequences,predictions,scaler, start_idx=120, pred_len=120):
    for i,(seq,pred) in enumerate(zip(sequences,predictions)):
        pred = pd.DataFrame(pred, columns=['bt', 'et', 'burner'])
        pred = scaler.inverse_transform(pred)
        forecast_idx = np.arange(start_idx, start_idx + pred_len)
        forecast_df = pd.DataFrame(pred, columns=['bt', 'et', 'burner'], index=forecast_idx/60)

        seq = pd.DataFrame(seq, columns=['bt', 'et', 'burner'])
        seq = scaler.inverse_transform(seq)
        seq_idx = np.arange(0, len(seq))
        seq_df = pd.DataFrame(seq, columns=['bt', 'et', 'burner'], index=seq_idx/60)

        plt.figure(figsize=(12, 6))
        plt.plot(seq_df['bt'], label='bt', color='blue')
        plt.plot(seq_df['et'], label='et', color='red')
        plt.plot(seq_df['burner'], label='burner', color='yellow')
        plt.plot(forecast_df['bt'], label='bt_forecast', linestyle='-', color='black')
        plt.plot(forecast_df['et'], label='et_forecast', linestyle='-', color='black')
        plt.plot(forecast_df['burner'], label='burner_forecast', linestyle='-', color='black')
        plt.legend()
        #save plot
        plt.savefig(f'figs/forecast_{i}.png')



# Prepare sequences
def prepare_sequences(data, scaler):
    sequences = []
    for roast in data:
        roast = pd.DataFrame(roast, columns=['bt', 'et', 'burner'])
        roast.dropna(inplace=True)
        roast = scaler.transform(roast)
        sequences.append(roast)
    return sequences

if __name__ == "__main__":
    # Load data
    path = 'data.npy'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'transformer_model.pth'
    data = np.load(path, allow_pickle=True)
    data = data[:100]
    np.random.shuffle(data)
    print(f"Data shape: {data.shape}")

    # Preprocess data
    scaler = get_scaler(data)
    sequences = prepare_sequences(data, scaler)

    split = int(len(sequences) * 0.8)
    train_sequences = sequences[:split]
    val_sequences = sequences[split:]

    # Hyperparameters
    input_dim = 3  # et, bt, gas%
    output_dim = 2  # et, bt
    d_model = 64
    nhead = 4
    num_layers = 3
    num_epochs = 50
    learning_rate = 0.001

    # Prepare dataset and dataloader
    train_dataset = DataLoader(train_sequences,device=device, batch_size=6)
    val_dataset = DataLoader(val_sequences,device=device)

    # Initialize model, loss, and optimizer
    model = CoffeeRoasterModel(input_dim=3, d_model=64, nhead=8, num_layers=4, output_dim=2, device=device)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    train_model(model, train_dataset,val_dataset, criterion, optimizer, num_epochs=num_epochs,device=device)

    # Save model
    torch.save(model.state_dict(), model_path)
    print("Model saved")

    # Load model
    model = CoffeeRoasterModel(input_dim=3, d_model=64, nhead=8, num_layers=4, output_dim=2, device=device)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    # Generate predictions
    predictions,loss,unscaled_loss = generate_prediction(model, val_sequences,scaler, start_idx=250, pred_len=120,device=device)
    print(f"Loss: {loss}, Unscaled loss: {unscaled_loss}")
    print(f"Predictions shape: {len(predictions)}")
    # Plot predictions
    plot_predictions(val_sequences, predictions, scaler, start_idx=250, pred_len=120)

