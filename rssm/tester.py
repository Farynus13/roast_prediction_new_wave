from rssm import RSSM
import torch
from models import EncoderMLP, DecoderMLP
from dynamics import DynamicsModel
import numpy as np
import pandas as pd
from utils import get_scaler
import imageio
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__ == "__main__":
    hidden_size = 128
    embedding_dim = 128
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
    # rssm.load("rssm/models/rssm_v2_128.pth")
    rssm.load("rssm_v2.pth")
    rssm.eval()

    #generate a rollout
    data = np.load("data.npy")
    scaler = get_scaler(data)

    # perturbations = range(1)
    # i = 0
    # for j,perturbation in enumerate(perturbations):
    mse = 0
    for i in tqdm(range(len(data)),desc="Generating Rollout",total=len(data)):
        roast = data[i].copy()
        roast = scaler.transform(roast)
        roast = pd.DataFrame(roast,columns=['bt','et','burner'])
        roast.dropna(inplace=True)
        obs_buffer = roast[['bt','et']].values
        action_buffer = roast['burner'].values
        done_buffer = np.zeros(len(roast),dtype=bool)
        init_length = 60 * 3


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
        
        roast = data[i].copy()
        # roast[:,2] = np.maximum(roast[:,2] + perturbation, 0)
        # roast[:,2] = np.concatenate([roast[perturbation:,2],roast[-1,2]*np.ones(perturbation)])
        # roast[:,2] = np.concatenate([roast[0,2]*np.ones(perturbation),roast[:-perturbation,2]])
        roast = scaler.transform(roast)
        roast = pd.DataFrame(roast,columns=['bt','et','burner'])
        roast.dropna(inplace=True)
        action_buffer = roast['burner'].values[init_length:]
        #extend rollout by appending last action for 60 steps
        # action_buffer = np.concatenate([action_buffer,action_buffer[-1]*np.ones(0)])
        remaining_steps = len(action_buffer)
        done_buffer = np.zeros(remaining_steps,dtype=bool)

        actions = torch.tensor(action_buffer, dtype=torch.float32).to(device).unsqueeze(-1).unsqueeze(0)
        dones = torch.tensor(done_buffer, dtype=torch.float32).to(device).unsqueeze(-1).unsqueeze(0)

        # Generate the rest of the rollout
        hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = rssm.generate_rollout(
            actions, hiddens[:,-1,:],posterior_states[:,-1,:], None, dones)
        
        hiddens_reshaped = hiddens.reshape(remaining_steps, -1)
        posterior_states_reshaped = posterior_states.reshape(remaining_steps, -1)

        decoded_obs_remaining = rssm.decoder(hiddens_reshaped, posterior_states_reshaped)
        decoded_obs_remaining = decoded_obs_remaining.reshape(1, remaining_steps, *obs.shape[2:])

        decoded_obs = torch.cat([decoded_obs, decoded_obs_remaining], dim=1)

        roast = data[i].copy()
        roast = pd.DataFrame(roast,columns=['bt','et','burner'])
        roast.dropna(inplace=True)
        roast = scaler.transform(roast)

        action_buffer = np.concatenate([roast[:init_length,2],action_buffer],axis=0)
        preds = decoded_obs.detach().cpu().numpy()[0]
        #unscale the data and add burner as well
        preds = np.concatenate([preds,action_buffer.reshape(-1,1),],axis=1)
        preds = scaler.inverse_transform(preds)
        preds = pd.DataFrame(preds,columns=['bt','et','burner'])
        # #add smoothing
        # preds['bt'] = preds['bt'].rolling(window=7, center=True).mean()
        # preds['et'] = preds['et'].rolling(window=7, center=True).mean()

        #plot bt_ror
        preds['bt_ror'] = preds['bt'].diff()*60
        preds['bt_ror'] = preds['bt_ror'].rolling(window=13, center=True).mean()
        
        # Remove spikes in predictions
        for _ in range(5):  # Apply median filter multiple times
            preds['bt_ror'] = preds['bt_ror'].rolling(window=17, center=True).median()
        
        roast = data[i].copy()
        gt = roast
        gt = pd.DataFrame(gt,columns=['bt','et','burner'])
        gt.dropna(inplace=True)

        mse += np.mean((preds['bt'].values[init_length:] - gt['bt'].values[init_length:])**2)

        # plot bt_ror
        gt['bt_ror'] = gt['bt'].diff()*60
        gt['bt_ror'] = gt['bt_ror'].rolling(window=9, center=True).mean()
        
        # Remove spikes in ground truth
        gt['bt_ror'] = gt['bt_ror'].rolling(window=5, center=True).median()


        fig, ax0 = plt.subplots(figsize=(10, 5))

        ax1 = ax0.twinx()

        ax0.plot(gt['bt'], label='GT BT', color='blue')
        ax0.plot(preds['bt'], label='Pred BT', color='blue', alpha=0.5)
        ax0.plot(gt['et'], label='GT ET', color='red')
        ax0.plot(preds['et'], label='Pred ET', color='red', alpha=0.5)
        ax0.plot(gt['burner'], label='GT Burner', color='orange')
        ax0.plot(preds['burner'], label='Pred Burner', color='orange', alpha=0.5)
        ax0.set_title('Bean and Environmental Temperature with Burner Level')
        ax0.set_xlabel('Time')
        ax0.set_ylabel('Temperature')
        ax0.set_ylim(0, 250)
        ax0.legend(loc='upper left')

        ax1.plot(gt['bt_ror'], label='GT BT ROR', color='blue')
        ax1.plot(preds['bt_ror'], label='Pred BT ROR', color='blue', alpha=0.5)
        ax1.set_ylabel('Rate of Rise')
        ax1.set_ylim(-3, 30)
        ax1.legend(loc='upper right')

        plt.tight_layout()
        # plt.show()

        #save image as rssm/media/{id}.png
        # fig.savefig(f"rssm/media/{j}.png")

        #close the plot
        plt.close(fig)

    rmse = np.sqrt(mse / len(data))
    print(rmse)

    # #create gif
    # images = []
    # for i in range(len(perturbations)):
    #     images.append(imageio.imread(f"rssm/media/{i}.png"))
    # imageio.mimsave('rssm/media/roast.gif', images, duration=1.0)

    # #clean up
    # for i in range(len(perturbations)):
    #     os.remove(f"rssm/media/{i}.png")
    # print("Roast gif generated and saved in rssm/media/roast.gif")
          





