import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import joblib
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import argparse

FIG_IDX = 0
np.random.seed(42)
torch.manual_seed(42)

def train(model, X_train,y_train,mask_train, X_test, y_test,mask_test, epochs=3, batch_size=64,lr=0.01,checkpoint_rate=10,start_epoch=0):
    # Convert training data to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the GPU

    # Convert training data to tensors and move to device
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    mask_train = torch.from_numpy(mask_train).float()
    lengths_train = torch.sum(mask_train, dim=1).long().cpu()



    # Convert test data to tensors and move to device
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    mask_test = torch.from_numpy(mask_test).float()
    lengths_test = torch.sum(mask_test, dim=1).long().cpu()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    k = 100
    for epoch in range(start_epoch,epochs):
        #linear decay
        #sampling_prob = 1.0 - (epoch / epochs)
        #inverse sigmoid decay
        #e_i = k / (k+exp(i/k)), k>=1
        sampling_prob = (k / (k + np.exp(epoch/epochs*1000 / k)))
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
            lengths_batch = lengths_train[i:i + batch_size]
            optimizer.zero_grad()
            output = model(x_batch,mask_batch,lengths_batch,y_batch,training=True,sampling_prob=sampling_prob)
            y_batch = y_batch[:, :, :2]

            # Calculate the loss
            loss = criterion(output, y_batch)
            loss.backward()  # Backpropagation
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        epoch_loss = epoch_loss / batches

        model.eval()
        with torch.no_grad():
                val_loss = 0
                batches = 0
                for i in range(0, X_test.shape[0], batch_size):
                        torch.cuda.empty_cache()
                        x_batch = X_test[i:i + batch_size].to(device)
                        y_batch = y_test[i:i + batch_size].to(device)
                        mask_batch = mask_test[i:i + batch_size].to(device)
                        lengths_batch = lengths_test[i:i + batch_size]
                        output = model(x_batch,mask_batch,lengths_batch,y_batch,training=True,sampling_prob=sampling_prob)
                        y_batch = y_batch[:, :, :2]
                        loss = criterion(output, y_batch)
                        val_loss += loss.item()
                        batches += 1
                val_loss = val_loss
        

        print(f'Epoch {epoch + 1}/{epochs}, vaL_loss: {val_loss}, Loss: {epoch_loss}')
        if (epoch + 1) % checkpoint_rate == 0:
                torch.save(model.state_dict(), f'model_checkpoint_{epoch + 1}.pth')

    print("Training completed.")
    return model

def prepare_variable_length_dataset(data,start_idx, max_length,scaler,split_ratio=0.8,future_len=50):
        #Train-test split preparation
        split = data.shape[0]*split_ratio
        X_train,y_train,X_test,y_test,mask_train,mask_test = [],[],[],[],[],[]
        #Splitting into sequences
        for i,roast in enumerate(data):
                roast = pd.DataFrame(roast,columns=['bt','et','burner'])
                roast.dropna(inplace=True)
                roast = scaler.transform(roast)
                for j in range(start_idx,int((roast.shape[0]-future_len-1)),future_len):
                        x_seq = np.zeros((max_length,3))
                        mask = np.zeros(max_length)
                        mask[:j] = 1
                        x_seq[:j] = roast[:j]
                        y_seq = roast[j:j+future_len]
                        if i < split:
                                X_train.append(x_seq)
                                mask_train.append(mask)
                                y_train.append(y_seq)
                        else:
                                #train-test split
                                X_test.append(x_seq)
                                mask_test.append(mask)
                                y_test.append(y_seq)

        #Convert to numpy arrays
        X_train = np.array(X_train,dtype=np.float32)
        y_train = np.array(y_train,dtype=np.float32)
        X_test = np.array(X_test,dtype=np.float32)
        y_test = np.array(y_test,dtype=np.float32)
        mask_train = np.array(mask_train,dtype=np.float32)
        mask_test = np.array(mask_test,dtype=np.float32)

        #shuffle by indices
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        mask_train = mask_train[indices]
        y_train = y_train[indices]

        indices = np.arange(X_test.shape[0])
        np.random.shuffle(indices)
        X_test = X_test[indices]
        y_test = y_test[indices]
        mask_test = mask_test[indices]

        return X_train,y_train,X_test,y_test,mask_train,mask_test

def test_variable_length_predictions(model, scaler, data, max_length, min_idx=60, eval_split=0.2,chunks=4,show_plot=False):
    test_data = data[int(len(data) * (1 - eval_split)):]
    mse = []
    for j,test_roast in tqdm(enumerate(test_data), desc='Evaluating model',total=len(test_data)):
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
        # Move inputs to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)  # Move the model to GPU if not already
        current_batch = torch.tensor(current_batch, dtype=torch.float32).to(device)
        current_mask = torch.tensor(current_mask, dtype=torch.float32).to(device)
        current_lengths = torch.sum(current_mask, dim=1).long().cpu()

        # Make Predictions in the loop
        model.eval()
        test_predictions = []
        test_predictions.append(scaled_test_roast[start_point - 1])  # Append last known value
        for i in range(pred_length):
            # Get prediction 1 time stamp ahead
            with torch.no_grad():  # Disable gradient calculation

                current_pred = model(current_batch,current_mask,current_lengths,training=False)[0].detach().cpu().numpy()  # Move to CPU for numpy operations

            burner = scaled_test_roast[(i + start_point), 2]
            current_pred = np.append(current_pred, burner)  # Add burner to prediction
            test_predictions.append(current_pred)

            #curr pred to torch tensor
            current_pred = torch.tensor(current_pred, dtype=torch.float32).unsqueeze(0).to(device)
            current_batch[0, i+start_point, :] = current_pred
            current_mask[0, i+start_point] = 1
            current_lengths = torch.sum(current_mask, dim=1).long().cpu()

        test_predictions = np.array(test_predictions)

        # Call plotting function
        plot_prediction(test_predictions, scaler, test_roast, start_point, 0, pred_length, j, show_plot)
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

def print_loss(history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.ylim(0,0.00001)
        plt.show()

def plot_prediction(test_predictions,scaler,test_roast,offset,input_sequence_length,pred_length,i,show_plot=True):
        test_predictions = pd.DataFrame(test_predictions,columns=['bt','et','burner'])
        test_predictions = scaler.inverse_transform(test_predictions)
        forecast_index = np.arange(offset+input_sequence_length-1,pred_length+offset+input_sequence_length)

        forecast_df =  pd.DataFrame(test_predictions[:,:2],columns=['bt','et'],index=(forecast_index-75)/60)

        # Create a subplot layout with 2 rows and 1 column
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

        # Plot the main data on the first subplot
        plot_roast = test_roast.drop(columns=['burner'])
        #to minutes
        plot_roast.index = (plot_roast.index-75)/60
        plot_roast.plot(ax=ax1,color=['blue','red'])
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
        ax2.plot((test_roast.index-75)/60, test_roast['burner'], label='Burner', color='orange')
        ax2.set_title('Burner Data')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Burner')

        #limit x to 13 min
        ax1.set_xlim(-2,13)

        # Show legends
        ax1.legend()
        ax2.legend()
        #change to black theme

        # Display the plots
        if show_plot:
                plt.show()

        #save fig to figs/fig_i
        fig.savefig(f'figs/fig_{i}.png')

 
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out, previous_hidden_state, mask):
        scores = self.Va(torch.tanh(self.Wa(lstm_out) + self.Ua(previous_hidden_state)))
        mask = mask.unsqueeze(2)
        scores.masked_fill_(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=1)
        context_vector = torch.bmm(attn_weights.permute(0, 2, 1), lstm_out)
        return context_vector.squeeze(1), attn_weights

class LSTMWithProperAttention(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, output_size=1):
        super(LSTMWithProperAttention, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x,masks, lengths,y=None, training=True, sampling_prob=1.0):
        batch_size, seq_len, _ = x.size()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h, c = self.init_hidden(batch_size, device)
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h, c) = self.lstm(packed_input, (h, c))
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len)
        previous_hidden_state = h[-1]
        context_vector, attn_weights = self.attention(lstm_out, previous_hidden_state.unsqueeze(1), masks)
        output = self.fc1(context_vector)
        output = self.relu(output)
        output = self.fc2(output)
        
        if training:
                outputs = torch.zeros((batch_size, y.size(1), 2), device=device)
                outputs[:,0] = output
                for i in range(y.size(1)-1):
                        new_point = y[:,i].unsqueeze(1).clone()
                        if random.random() > sampling_prob:
                                new_point[:,0,:2] = output
                        #use new point to predict next
                        #we dont need to pack padded sequence here
                        lstm_out_single, (h, c) = self.lstm(new_point, (h, c))
                        lstm_out = lstm_out.clone()
                        lstm_out[:,i+1] = lstm_out_single.squeeze(1)
                        masks[:,i+1] = 1
                        previous_hidden_state = h[-1]
                        context_vector, attn_weights = self.attention(lstm_out, previous_hidden_state.unsqueeze(1), masks)
                        output = self.fc1(context_vector)
                        output = self.relu(output)
                        output = self.fc2(output)
                        outputs[:,i+1] = output
                return outputs
        else:
                return output
                

    def init_hidden(self, batch_size, device):
        # Initialize hidden and cell states for LSTM
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))

if __name__ == '__main__':
        #pars args
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", help="path to model")
        parser.add_argument("--start_epoch", help="start epoch", default=0)
        args = parser.parse_args()
        start_epoch = int(args.start_epoch)
        model_path = args.model_path
        #if using data directly from workspace
        path = 'data.npy'
        data = np.load(path, allow_pickle=True)
        print(data.shape)

        #constants
        n_features = 3

        #plot all roasts
        # for roast in data:
        #         plt.plot(roast)
        # plt.show()

        max_length = 800
        future_len = 20
        start_idx = 150+75
        scaler = get_scaler(data)
        #shuffle
        #seed
        np.random.shuffle(data)
        train_test_data = data[:int(0.9*len(data))]
        validation_data = data[int(0.9*len(data)):]
        X_train,y_train,X_test,y_test,mask_train,mask_test = prepare_variable_length_dataset(train_test_data,start_idx,max_length,scaler,future_len=future_len)

        torch.cuda.empty_cache()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMWithProperAttention(input_size=3, hidden_size=128, output_size=2)
        model.to(device)
        if start_epoch > 0:
                model.load_state_dict(torch.load(model_path,map_location=device,weights_only=True))
        model = train(model, X_train,y_train,mask_train,X_test, y_test,mask_test, epochs=100, batch_size=128,lr=0.001,checkpoint_rate=1,start_epoch=start_epoch)
        torch.save(model.state_dict(), 'model.pth')

        # #load model
        # model = LSTMWithProperAttention(input_size=3, hidden_size=128, output_size=2)
        # model.load_state_dict(torch.load('model_checkpoint_20_meta.pth',map_location=torch.device('cpu')))

        # test_variable_length_predictions(model,scaler,validation_data,max_length,min_idx=start_idx,eval_split=1.0,chunks=3,show_plot=True)



