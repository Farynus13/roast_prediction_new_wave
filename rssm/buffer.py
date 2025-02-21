import torch
import numpy as np
import pandas as pd


class Buffer:
    def __init__(self, data_path: str, obs_shape: tuple, action_shape: tuple, device: torch.device):

        self.data = np.load(data_path, allow_pickle=True)
        self.obs_buffer = None
        self.action_buffer = None
        self.done_buffer = None

        self.construct_buffers(obs_shape, action_shape, device)

        self.device = device

        self.idx = self.buffer_size - 1

    def construct_buffers(self, obs_shape: tuple, action_shape: tuple, device: torch.device):
        # data is n_roasts, n_steps, obs_dim+action_dim 
        # first we need to flatten the data
        data = self.data

        obs_buffer = []
        action_buffer = []
        done_buffer = []

        for i,roast in enumerate(data):
            roast = pd.DataFrame(roast,columns=['bt','et','burner'])
            roast.dropna(inplace=True)
            for j in range(len(roast)):
                obs = roast.iloc[j][['bt','et']].values
                action = roast.iloc[j]['burner']
                done = j == len(roast) - 1
                obs_buffer.append(obs)
                action_buffer.append(action)
                done_buffer.append(done)

        self.obs_buffer = torch.tensor(obs_buffer, dtype=torch.float32, device=device)
        self.action_buffer = torch.tensor(action_buffer, dtype=torch.int64, device=device)
        self.done_buffer = torch.tensor(done_buffer, dtype=torch.bool, device=device)

        self.buffer_size = len(self.obs_buffer)
        assert len(self.action_buffer) == len(self.obs_buffer) == len(self.done_buffer)
            
    def add(self, obs: torch.Tensor, action: int, done: bool):
        self.obs_buffer[self.idx] = obs
        self.action_buffer[self.idx] = action
        self.done_buffer[self.idx] = done

        self.idx = (self.idx + 1) % self.buffer_size


    def sample(self, batch_size: int, sequence_length: int):
        starting_idxs = np.random.randint(0, (self.idx % self.buffer_size) - sequence_length, (batch_size,))

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
