import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import joblib
import tensorflow as tf
from tqdm import tqdm
import random

FIG_IDX = 0

from struct import pack
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def generate_adversarial_examples(model, x, y, mask, epsilon,criterion):
    x.requires_grad = True
    output = model(x, mask)
    loss = criterion(output, y)  # Calculate loss using masked values
    model.zero_grad()
    loss.backward()

    # Apply mask to gradients
    data_grad = x.grad.data * mask.unsqueeze(-1)  # Adjust gradient based on mask

    # Create adversarial examples
    perturbed_data = x + epsilon * data_grad.sign()
    return perturbed_data

def train(model, X_train, y_train, mask_train,meta_train, X_test, y_test, mask_test,meta_test, epochs=3, batch_size=64,lr=0.01,checkpoint_rate=10):
    # Convert training data to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the GPU

    # Convert training data to tensors and move to device
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    mask_train = torch.from_numpy(mask_train).float()
    meta_train = torch.from_numpy(meta_train).float()

    # Convert test data to tensors and move to device
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    mask_test = torch.from_numpy(mask_test).float()
    meta_test = torch.from_numpy(meta_test).float()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    initial_ratio = 1.0

    for epoch in range(epochs):
        teacher_forcing_ratio = initial_ratio - (epoch / epochs)
        print(f'Epoch {epoch + 1}/{epochs}')
        model.train()
        epoch_loss = 0
        batches = 0
        for i in tqdm(range(0, X_train.shape[0], batch_size), total=X_train.shape[0] // batch_size + 1, desc='Training'):
            torch.cuda.empty_cache()
            # Create batches using slicing
            x_batch = X_train[i:i + batch_size].to(device)
            y_batch = y_train[i:i + batch_size].to(device)
            mask_batch = mask_train[i:i + batch_size].to(device)
            meta_batch = meta_train[i:i + batch_size].to(device)

            optimizer.zero_grad()
            output = model(x_batch, mask_batch,meta_batch)
            # Calculate the loss
            loss = criterion(output, y_batch)
            loss.backward()  # Backpropagation
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        epoch_loss = epoch_loss / batches

        model.eval()
        with torch.no_grad():
                vaL_loss = 0
                batches = 0
                for i in range(0, X_test.shape[0], batch_size):
                        x_batch = X_test[i:i + batch_size].to(device)
                        y_batch = y_test[i:i + batch_size].to(device)
                        mask_batch = mask_test[i:i + batch_size].to(device)
                        meta_batch = meta_test[i:i + batch_size].to(device)
                        output = model(x_batch, mask_batch,meta_batch)

                        y_batch = y_batch[:, :2]
                        loss = criterion(output, y_batch)
                        vaL_loss += loss.item()
                        batches += 1

                vaL_loss /= batches

        print(f'Epoch {epoch + 1}/{epochs}, vaL_loss: {vaL_loss}, Loss: {epoch_loss}')
        if (epoch + 1) % checkpoint_rate == 0:
                torch.save(model.state_dict(), f'model_checkpoint_{epoch + 1}.pth')

    print("Training completed.")
    return model

def prepare_variable_length_dataset(data,meta_data,start_idx, max_length,scaler,split_ratio=0.8):
        #Train-test split preparation
        split = data.shape[0]*split_ratio
        X_train,X_test,y_train,y_test,mask_train,mask_test = [],[],[],[],[],[]
        train_meta,test_meta = [],[]
        #Splitting into sequences
        for i,(roast,descriptors) in enumerate(zip(data,meta_data)):
                roast = pd.DataFrame(roast,columns=['bt','et','burner'])
                roast.dropna(inplace=True)
                roast = scaler.transform(roast)

                for j in range(start_idx,int((roast.shape[0]-1))):
                        x_seq = np.zeros((max_length,3))
                        mask = np.zeros(max_length)
                        mask[:j] = 1
                        x_seq[:j] = roast[:j]
                        y_seq = roast[j,:2]

                        #train-test split
                        if i < split:
                                X_train.append(x_seq)
                                mask_train.append(mask)
                                y_train.append(y_seq)
                                train_meta.append(descriptors)
                        else:
                                X_test.append(x_seq)
                                mask_test.append(mask)
                                y_test.append(y_seq)
                                test_meta.append(descriptors)

        #Convert to numpy arrays
        X_train = np.array(X_train,dtype=np.float32)
        y_train = np.array(y_train,dtype=np.float32)
        X_test = np.array(X_test,dtype=np.float32)
        y_test = np.array(y_test,dtype=np.float32)
        mask_train = np.array(mask_train,dtype=np.float32)
        mask_test = np.array(mask_test,dtype=np.float32)
        train_meta = np.array(train_meta,dtype=np.float32)
        test_meta = np.array(test_meta,dtype=np.float32)

        #shuffle by indices
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        mask_train = mask_train[indices]
        train_meta = train_meta[indices]

        indices = np.arange(X_test.shape[0])
        np.random.shuffle(indices)
        X_test = X_test[indices]
        y_test = y_test[indices]
        mask_test = mask_test[indices]
        test_meta = test_meta[indices]

        return X_train,y_train,X_test,y_test,mask_train,mask_test,train_meta,test_meta

def test_variable_length_predictions(model, scaler, data,meta_data, max_length, min_idx=60, eval_split=0.2,chunks=4,show_plot=False):
    test_data = data[int(len(data) * (1 - eval_split)):]
    test_meta_data = meta_data[int(len(data) * (1 - eval_split)):]
    mse = []
    for test_roast,test_descriptions in tqdm(zip(test_data,test_meta_data), desc='Evaluating model',total=len(test_data)):
      for chunk in range(chunks):
        test_roast = pd.DataFrame(test_roast, columns=['bt', 'et', 'burner'])
        test_roast.dropna(inplace=True)
        start_point = min_idx + chunk * (test_roast.shape[0] - min_idx - 1) // chunks
        pred_length = len(test_roast) - start_point



        scaled_test_roast = scaler.transform(test_roast)
        print(scaled_test_roast.shape)

        # Prepare initial data batch
        current_batch = np.zeros((max_length, 3))
        current_batch[:start_point] = scaled_test_roast[:start_point]
        current_mask = np.zeros(max_length)
        current_mask[:start_point] = 1
        # Expand dimensions for batch processing
        current_batch = np.expand_dims(current_batch, 0)  # Shape (1, max_length, 3)
        current_mask = np.expand_dims(current_mask, 0)  # Shape (1, max_length)
        current_descriptions = np.expand_dims(test_descriptions,0)
        # Move inputs to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)  # Move the model to GPU if not already
        current_batch = torch.tensor(current_batch, dtype=torch.float32).to(device)
        current_mask = torch.tensor(current_mask, dtype=torch.float32).to(device)
        current_descriptions = torch.tensor(current_descriptions, dtype=torch.float32).to(device)
        # Make Predictions in the loop
        model.eval()
        test_predictions = []
        for i in range(pred_length):
            # Get prediction 1 time stamp ahead
            with torch.no_grad():  # Disable gradient calculation

                current_pred = model(current_batch,current_mask,current_descriptions)[0].detach().cpu().numpy()  # Move to CPU for numpy operations

            burner = scaled_test_roast[(i + start_point), 2]
            current_pred = np.append(current_pred, burner)  # Add burner to prediction
            test_predictions.append(current_pred)

            #curr pred to torch tensor
            current_pred = torch.tensor(current_pred, dtype=torch.float32).unsqueeze(0).to(device)
            current_batch[0, i+start_point, :] = current_pred
            current_mask[0, i+start_point] = 1
        test_predictions = np.array(test_predictions)

        # Call plotting function
        if show_plot:
          plot_prediction(test_predictions, scaler, test_roast, start_point, 0, pred_length)
        mse.append(calculate_mse(test_predictions,test_roast,scaler,start_point))

    print(f'MSE: {np.mean(mse)}')

def calculate_mse(preds,test_roast,scaler,start_point):
  test_predictions = pd.DataFrame(preds,columns=['bt','et','burner'])
  test_predictions = scaler.inverse_transform(test_predictions)
  forecast_index = np.arange(start_point,start_point+len(test_predictions))
  forecast_df =  pd.DataFrame(test_predictions[:,:2],columns=['bt','et'],index=forecast_index)
  errors_bt = forecast_df['bt'] - test_roast['bt']
  errors_bt = errors_bt.dropna()
  errors_bt = np.abs(errors_bt)
  errors_bt = errors_bt.values

  errors_et = forecast_df['et'] - test_roast['et']
  errors_et = errors_et.dropna()
  errors_et = np.abs(errors_et)
  errors_et = errors_et.values
  # Calculate the mean squared error (MSE)
  mse = (np.mean(np.square(errors_bt))+np.mean(np.square(errors_et))) / 2
  return mse

def get_scaler(data,type='StandardScaler',save=True,path='scaler.joblib'):
        #concatenate data for scaler fitting
        data_concat = [point for roast in data for point in roast]
        data_concat = pd.DataFrame(data_concat,columns=['bt','et','burner'])
        data_concat.dropna(inplace=True)

        #scaler preparation
        X = data_concat[['bt', 'et', 'burner']]
        if type == 'StandardScaler':
                scaler = StandardScaler()
        elif type == 'MinMaxScaler':
                scaler = MinMaxScaler()
        scaler.fit(X)

        if save:
                joblib.dump(scaler,path)
        return scaler

def prepare_dataset(data, input_sequence_length,scaler,split_ratio=0.8,seed=42):
 #Train-test split preparation
        np.random.seed(seed)
        np.random.shuffle(data)
        split = data.shape[0]*split_ratio
        X_train,X_test,y_train,y_test = [],[],[],[]
        #Splitting into sequences
        for i,roast in enumerate(data):
                roast = pd.DataFrame(roast,columns=['bt','et','burner'])
                roast.dropna(inplace=True)
                roast = scaler.transform(roast)

                for j in range(int((roast.shape[0]-input_sequence_length-1))):
                        x_seq = roast[j:j+input_sequence_length]
                        y_seq = roast[j+input_sequence_length,:2]

                        #train-test split
                        if i < split:
                                X_train.append(x_seq)
                                y_train.append(y_seq)
                        else:
                                X_test.append(x_seq)
                                y_test.append(y_seq)

        #Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        #shuffle by indices
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        indices = np.arange(X_test.shape[0])
        np.random.shuffle(indices)
        X_test = X_test[indices]
        y_test = y_test[indices]

        return X_train,y_train,X_test,y_test

def create_model(n_features,input_sequence_length,print_summary=True):
        #Create model
        model = Sequential()
        model.add(LSTM(100, activation='tanh', input_shape=(input_sequence_length, n_features),
                return_sequences=True,unroll=False))
        model.add(LSTM(100, activation='tanh', return_sequences=True,unroll=False))
        model.add(LSTM(50, activation='tanh', return_sequences=False,unroll=False))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(2))
        model.compile(optimizer=Adam(0.0002), loss='mse')
        if print_summary:
                model.summary()
        return model

def print_loss(history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.ylim(0,0.00001)
        plt.show()

def test_predictions(model,scaler,data,input_sequence_length,offset=60,pred_length=240,roast_idx = -1):
        #Prepare roast data
        test_index = 0
        if roast_idx == -1:
                test_index = np.random.randint(0,data.shape[0])
        else:
                test_index = roast_idx
        test_roast = data[test_index]
        test_roast = pd.DataFrame(test_roast,columns=['bt','et','burner'])
        test_roast.dropna(inplace=True)
        scaled_test_roast = scaler.transform(test_roast)
        print(scaled_test_roast.shape)

        #Prepare initial data batch
        current_batch = scaled_test_roast[offset:offset+input_sequence_length]

        current_batch = np.expand_dims(current_batch,0)

        #Make Predictions in the loop
        test_predictions = []
        for i in range(pred_length):
                # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
                current_pred = model.predict(current_batch)[0]
                # store prediction
                if scaled_test_roast.shape[0] > (i+offset):
                        burner = scaled_test_roast[(i+offset),2]
                else:
                        burner = scaled_test_roast[-1,2]
                current_pred = np.append(current_pred,burner) #add burner to prediction
                test_predictions.append(current_pred)
                #shift it to the left getting rid of first and as last value insert prediction
                current_batch = np.roll(current_batch, -1, axis=1)
                current_batch[0,-1,:]=current_pred

        test_predictions = np.array(test_predictions)
        plot_prediction(test_predictions,scaler,test_roast,offset,input_sequence_length,pred_length)

def plot_prediction(test_predictions,scaler,test_roast,offset,input_sequence_length,pred_length):
        test_predictions = pd.DataFrame(test_predictions,columns=['bt','et','burner'])
        test_predictions = scaler.inverse_transform(test_predictions)
        forecast_index = np.arange(offset+input_sequence_length,pred_length+offset+input_sequence_length)

        forecast_df =  pd.DataFrame(test_predictions[:,:2],columns=['bt','et'],index=forecast_index)

        # Create a subplot layout with 2 rows and 1 column
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

        # Plot the main data on the first subplot
        test_roast.drop(columns=['burner']).plot(ax=ax1,color=['blue','red'])
        ax1.set_title('BT and ET')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Values')
        forecast_df.plot(ax=ax1,color=['black','black'],style='--')
        ax1.set_title('BT and ET with Forecast')
        #mark points where br is close to 160 and call it 'yellowing'
        ax1.axhline(y=160, color='y', label='Yellowing')
        ax1.axhline(y=198, color='purple', label='FC')
        ax1.axhline(y=211, color='black', label='DROP')

        # Plot the burner data on the second subplot
        ax2.plot(test_roast.index, test_roast['burner'], label='Burner', color='orange')
        ax2.set_title('Burner Data')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Burner')

        # Show legends
        ax1.legend()
        ax2.legend()

        # Display the plots
        plt.show()

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)  # For encoder hidden states
        self.Ua = nn.Linear(hidden_size, hidden_size)  # For decoder hidden state
        self.Va = nn.Linear(hidden_size, 1)             # To compute scores

    def forward(self, lstm_out, previous_hidden_state, mask):
        # Compute attention scores
        scores = self.Va(torch.tanh(self.Wa(lstm_out) + self.Ua(previous_hidden_state)))

        # Apply mask: setting scores to a very low value where mask is 0
        mask = mask.unsqueeze(2)
        scores.masked_fill_(mask == 0, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=1)  # Shape: (batch_size, seq_len, 1)

        # Weighted sum of the LSTM outputs
        context_vector = torch.bmm(attn_weights.permute(0, 2, 1), lstm_out)  # Shape: (batch_size, 1, hidden_size)

        return context_vector.squeeze(1), attn_weights

class LSTMWithProperAttention(nn.Module):
    def __init__(self, input_size=3, metadata_size=17, hidden_size=128, num_layers=2, output_size=1):
        super(LSTMWithProperAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        # Attention layer
        self.attention = Attention(hidden_size)

        # Transformation layer for metadata to hidden state initialization
        self.metadata_to_hidden = nn.Linear(metadata_size, hidden_size * num_layers * 2)  # 2 for hidden and cell states

        # Intermediate linear layer
        self.fc1 = nn.Linear(hidden_size, 32)  # Intermediate layer
        self.relu = nn.ReLU()                   # Activation function
        self.fc2 = nn.Linear(32, output_size)   # Output layer

    def forward(self, x, mask,metadata):
        # Transform metadata to initialize hidden and cell states
        metadata_hidden = self.metadata_to_hidden(metadata)
        hidden_state, cell_state = torch.split(metadata_hidden, self.hidden_size * self.num_layers, dim=1)
        hidden_state = hidden_state.reshape(self.num_layers, -1, self.hidden_size)
        cell_state = cell_state.reshape(self.num_layers, -1, self.hidden_size)

        # Forward pass through the LSTM
        seq_len = x.shape[1]
        lengths = torch.sum(mask, dim=1).long().cpu()
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(x, (hidden_state, cell_state))
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_len)

        # Use the last hidden state for attention
        previous_hidden_state = hidden_state[-1]

        # Apply attention mechanism with mask
        context_vector, attn_weights = self.attention(lstm_out, previous_hidden_state.unsqueeze(1), mask)

        # Pass through the intermediate linear layer
        x = self.fc1(context_vector)
        x = self.relu(x)

        # Final output
        x = self.fc2(x)

        return x


if __name__ == '__main__':
        #if using data directly from workspace
        path = 'data.npy'
        peth_meta = 'meta_data.npy'
        data = np.load(path, allow_pickle=True)
        print(data.shape)
        meta_data = np.load(peth_meta, allow_pickle=True)
        print(meta_data.shape)

        #constants
        n_features = 3
        input_sequence_length = 240
        meta_features = 17

        #plot all roasts
        for roast in data:
                plt.plot(roast)
        plt.show()

        max_length = 800
        meta_size = meta_data.shape[1]

        start_idx = 120+75
        scaler = get_scaler(data)
        #shuffle
        np.random.shuffle(data)
        train_test_data = data[:int(0.9*len(data))]
        train_test_meta_data = meta_data[:int(0.9*len(data))]
        validation_data = data[int(0.9*len(data)):]
        validation_meta_data = meta_data[int(0.9*len(data)):]
        X_train,y_train,X_test,y_test,mask_train,mask_test,meta_train,meta_test = prepare_variable_length_dataset(train_test_data,train_test_meta_data,start_idx,max_length,scaler)

        torch.cuda.empty_cache()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #model = LSTMWithAttention(input_size=3, hidden_size=256, output_size=2,future_len=future_len)
        model = LSTMWithProperAttention(input_size=3,metadata_size=meta_size, hidden_size=128, output_size=2)
        model.to(device)
        model = train(model, X_train, y_train,mask_train,meta_train,X_test, y_test,mask_test,meta_test, epochs=100, batch_size=256,lr=0.001)
        torch.save(model.state_dict(), 'model.pth')

        #load model
        model = LSTMWithProperAttention(input_size=3, hidden_size=128, output_size=2)
        model.load_state_dict(torch.load('model_checkpoint_20_meta.pth',map_location=torch.device('cpu')))

        test_variable_length_predictions(model,scaler,validation_data,validation_meta_data,max_length,min_idx=start_idx,eval_split=1.0,chunks=3,show_plot=True)



