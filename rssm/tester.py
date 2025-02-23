from rssm import RSSM
import torch
import torch.nn as nn
from models import EncoderMLP, DecoderMLP
from dynamics import DynamicsModel
import numpy as np
import pandas as pd
from utils import get_scaler

if __name__ == "__main__":
    hidden_size = 256
    embedding_dim = 256
    state_dim = 128
    action_dim = 1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    encoder = EncoderMLP(input_dim=2, embedding_dim=embedding_dim)
    decoder = DecoderMLP(hidden_size=hidden_size, state_size=state_dim, output_dim=2, embedding_size=embedding_dim)
    dynamics_model = DynamicsModel(hidden_dim=hidden_size, state_dim=state_dim, action_dim=action_dim, embedding_dim=embedding_dim)

    rssm = RSSM(dynamics_model=dynamics_model,
                encoder=encoder,
                decoder=decoder,
                hidden_dim=hidden_size,
                state_dim=state_dim,
                action_dim=action_dim,
                embedding_dim=embedding_dim,
                device=device)
    
    #load the model
    rssm.load("rssm_v2.pth")

    rssm.eval()

    #generate a rollout
    data = np.load("data.npy")
    scaler = get_scaler(data)
    roast = data[10]
    roast = scaler.transform(roast)
    roast = pd.DataFrame(roast,columns=['bt','et','burner'])
    roast.dropna(inplace=True)
    obs_buffer = roast[['bt','et']].values
    action_buffer = roast['burner'].values
    done_buffer = np.zeros(len(roast),dtype=bool)
    init_length = 240

    hiddens = torch.zeros(1, hidden_size).to(device)
    states = torch.zeros(1, state_dim).to(device)
    obs = torch.tensor(obs_buffer[:init_length], dtype=torch.float32).to(device).unsqueeze(0)
    actions = torch.tensor(action_buffer[:init_length], dtype=torch.float32).to(device).unsqueeze(-1).unsqueeze(0)
    dones = torch.tensor(done_buffer[:init_length], dtype=torch.float32).to(device).unsqueeze(-1).unsqueeze(0)

    encoded_obs = rssm.encoder(obs.reshape(-1, *obs.shape[2:]))
    encoded_obs = encoded_obs.reshape(1, init_length, -1)

    hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = rssm.generate_rollout(actions, hiddens, states, encoded_obs, dones)
    hiddens_reshaped = hiddens.reshape(init_length, -1)
    posterior_states_reshaped = posterior_states.reshape(init_length, -1)
    prior_states_reshaped = prior_states.reshape(init_length, -1)
    decoded_obs = rssm.decoder(hiddens_reshaped, posterior_states_reshaped)
    decoded_obs = decoded_obs.reshape(1, init_length, *obs.shape[2:])
    
    roast = data[10].copy()
    roast[:,2] = roast[:,2]
    time_shift =15
    roast[:,2] = np.concatenate([roast[time_shift:,2],roast[-1,2]*np.ones(time_shift)])
    # roast[:,2] = np.concatenate([roast[0,2]*np.ones(time_shift),roast[:-time_shift,2]])
    roast = scaler.transform(roast)
    roast = pd.DataFrame(roast,columns=['bt','et','burner'])
    roast.dropna(inplace=True)
    action_buffer = roast['burner'].values
    #extend rollout by appending last action for 60 steps
    remaining_steps = 60 + len(roast) - init_length
    action_buffer = np.concatenate([action_buffer[init_length:],action_buffer[-1]*np.ones(60)])
    done_buffer = np.zeros(remaining_steps,dtype=bool)

    actions = torch.tensor(action_buffer, dtype=torch.float32).to(device).unsqueeze(-1).unsqueeze(0)
    dones = torch.tensor(done_buffer, dtype=torch.float32).to(device).unsqueeze(-1).unsqueeze(0)

    # Generate the rest of the rollout
    hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = rssm.generate_rollout(
        actions, hiddens[:,-1,:],posterior_states[:,-1,:], None, dones)
    
    hiddens_reshaped = hiddens.reshape(remaining_steps, -1)
    posterior_states_reshaped = posterior_states.reshape(remaining_steps, -1)

    decoded_obs_new = rssm.decoder(hiddens_reshaped, posterior_states_reshaped)
    decoded_obs_new = decoded_obs_new.reshape(1, remaining_steps, *obs.shape[2:])

    decoded_obs = torch.cat([decoded_obs, decoded_obs_new], dim=1)

    action_buffer = np.concatenate([roast['burner'].values[:init_length],action_buffer],axis=0)
    preds = decoded_obs.detach().cpu().numpy()[0]
    #unscale the data and add burner as well
    preds = np.concatenate([preds,action_buffer.reshape(-1,1),],axis=1)
    preds = scaler.inverse_transform(preds)
    preds = pd.DataFrame(preds,columns=['bt','et','burner'])
    
    roast = data[10].copy()
    gt = roast
    gt = pd.DataFrame(gt,columns=['bt','et','burner'])
    #plot real vs predicted
    
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].plot(gt['bt'], label='GT BT')
    axs[0].plot(preds['bt'], label='Pred BT')
    axs[0].plot(gt['et'], label='GT ET')
    axs[0].plot(preds['et'], label='Pred ET')
    axs[0].set_title('Bean and Environmental Temperature')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Temperature')
    axs[0].legend()

    axs[1].plot(gt['burner'], label='GT Burner')
    axs[1].plot(preds['burner'], label='Pred Burner')
    axs[1].set_title('Burner')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Burner Level')
    axs[1].legend()

    plt.tight_layout()
    plt.show()






