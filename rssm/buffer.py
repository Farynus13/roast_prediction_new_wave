import torch
import numpy as np
import pandas as pd
from utils import get_scaler

class RoastDataset:
    def __init__(self, data: np.ndarray, device: torch.device):

        self.data = data
        self.device = device
        self.scaler = get_scaler(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        roast = self.data[idx]
        roast = pd.DataFrame(roast,columns=['bt','et','burner'])
        roast.dropna(inplace=True)
        roast = self.scaler.transform(roast)
        roast = pd.DataFrame(roast,columns=['bt','et','burner'])
        obs = roast[['bt','et']].values
        action = roast['burner'].values
        done = np.zeros(len(roast),dtype=bool)
        done[-1] = True
        return obs, action, done

class RoastDataLoader:
    def __init__(self, dataset: RoastDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = self.dataset.device

        self.idx = 0
        self.n = len(self.dataset)
        self.indices = np.arange(self.n)

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return self.n // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.n:
            raise StopIteration

        idxs = self.indices[self.idx:self.idx + self.batch_size]
        obs_batch = []
        action_batch = []
        done_batch = []

        for idx in idxs:
            obs, action, done = self.dataset[idx]
            obs_batch.append(obs)
            action_batch.append(action)
            done_batch.append(done)

        #before creating tensors we need to pad sequences to the same length
        max_length = max([len(obs) for obs in obs_batch])
        #create padding mask
        mask_batch = np.zeros((len(obs_batch), max_length), dtype=bool).reshape(len(obs_batch), max_length, 1)
        for i, obs in enumerate(obs_batch):
            mask_batch[i, :len(obs)] = True
        #pad sequences
        obs_batch = [np.pad(obs, ((0, max_length - len(obs)), (0, 0))) for obs in obs_batch]
        action_batch = [np.pad(action, (0, max_length - len(action))) for action in action_batch]
        done_batch = [np.pad(done, (0, max_length - len(done))) for done in done_batch]

        obs_batch = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device)
        mask_batch = torch.tensor(mask_batch, dtype=torch.bool, device=self.device)

        self.idx += self.batch_size

        return obs_batch, action_batch, done_batch, mask_batch

class Buffer:
    def __init__(self, data_path: str, obs_shape: tuple, action_shape: tuple, device: torch.device):

        self.data = np.load(data_path, allow_pickle=True)
        self.obs_buffer = None
        self.action_buffer = None
        self.done_buffer = None

        self.construct_buffers(obs_shape, action_shape, device)

        self.device = device

        self.idx = self.buffer_size - 1

        self.sequence_start_idxs = np.where(self.done_buffer.cpu().numpy())[0]

    def construct_buffers(self, obs_shape: tuple, action_shape: tuple, device: torch.device):
        # data is n_roasts, n_steps, obs_dim+action_dim 
        # first we need to flatten the data
        
        data = self.data
        scaler = get_scaler(data)

        obs_buffer = []
        action_buffer = []
        done_buffer = []

        for i,roast in enumerate(data):
            roast = pd.DataFrame(roast,columns=['bt','et','burner'])
            roast.dropna(inplace=True)
            roast = scaler.transform(roast)
            roast = pd.DataFrame(roast,columns=['bt','et','burner'])
            for j in range(len(roast)):
                obs = roast.iloc[j][['bt','et']].values
                action = roast.iloc[j]['burner']
                done = j == len(roast) - 1
                obs_buffer.append(obs)
                action_buffer.append(action)
                done_buffer.append(done)

        self.obs_buffer = torch.tensor(obs_buffer, dtype=torch.float32, device=device)
        self.action_buffer = torch.tensor(action_buffer, dtype=torch.float32, device=device)
        self.done_buffer = torch.tensor(done_buffer, dtype=torch.bool, device=device)

        self.buffer_size = len(self.obs_buffer)
        assert len(self.action_buffer) == len(self.obs_buffer) == len(self.done_buffer)
            
    def add(self, obs: torch.Tensor, action: int, done: bool):
        self.obs_buffer[self.idx] = obs
        self.action_buffer[self.idx] = action
        self.done_buffer[self.idx] = done

        self.idx = (self.idx + 1) % self.buffer_size



    def sample(self, batch_size: int, sequence_length: int):
        # instead of sampling randomly we will always sample only from idxs that start sequences (1 after done)

        starting_idxs = np.random.choice(self.sequence_start_idxs[:-1], (batch_size,))
        starting_idxs = np.clip(starting_idxs, 0, self.buffer_size - sequence_length)
      
        # starting_idxs = np.random.randint(0, (self.idx % self.buffer_size) - sequence_length, (batch_size,))

        index_tensor = np.stack([np.arange(start, start + sequence_length) for start in starting_idxs])
        obs_sequence = self.obs_buffer[index_tensor]
        action_sequence = self.action_buffer[index_tensor]
        done_sequence = self.done_buffer[index_tensor]

        return obs_sequence, action_sequence, done_sequence
    
if __name__ == "__main__":
    data_path = "data.npy"
    obs_shape = (2,)
    action_shape = (1,)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    buffer = Buffer(data_path, obs_shape, action_shape, device)
    print(buffer.obs_buffer.shape)
    print(buffer.action_buffer.shape)
    print(buffer.done_buffer.shape)
    obs, action, done = buffer.sample(1, 10)
    print(obs.shape)
    print(action.shape)
    print(done.shape)
    print(obs)
    print(action)
    print(done)
