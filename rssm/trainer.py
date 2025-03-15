import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from tqdm import tqdm
import logging
from rssm import RSSM
from buffer import Buffer, RoastDataLoader, RoastDataset
from models import EncoderMLP, DecoderMLP
from dynamics import DynamicsModel
from torch import nn


from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output logs to console
        logging.FileHandler("training.log", mode="w")
    ]
)

logger = logging.getLogger(__name__)

class Trainerv2:
    def __init__(self, rssm: RSSM, train_dataset: RoastDataset,val_dataset:RoastDataset, optimizer: torch.optim.Optimizer, device: torch.device):
        self.rssm = rssm
        self.optimizer = optimizer
        self.device = device
        #shuffle the dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.writer = SummaryWriter()

    def train_batch(self, obs: torch.Tensor, actions: torch.Tensor, dones: torch.Tensor, masks: torch.Tensor):
        seq_len = obs.shape[1]
        batch_size = obs.shape[0]
        actions = actions.clone().detach().float().to(self.device)
        actions = actions.unsqueeze(-1)  

        obs = obs.clone().detach().float().to(self.device)
        dones = dones.clone().detach().float().to(self.device)
        dones = dones.unsqueeze(-1)

        encoded_obs = self.rssm.encoder(obs.reshape(-1, *obs.shape[2:]))
        encoded_obs = encoded_obs.reshape(batch_size, seq_len, -1)

        rollout = self.rssm.generate_rollout(actions, obs=encoded_obs, dones=dones)

        hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = rollout

        hiddens_reshaped = hiddens.reshape(batch_size * seq_len, -1)
        posterior_states_reshaped = posterior_states.reshape(batch_size * seq_len, -1)

        decoded_obs = self.rssm.decoder(hiddens_reshaped, posterior_states_reshaped)
        decoded_obs = decoded_obs.reshape(batch_size, seq_len, *obs.shape[2:])

        reconstruction_loss = self._reconstruction_loss(decoded_obs, obs, masks)
        kl_loss = self._kl_loss(prior_means, F.softplus(prior_logvars), posterior_means, F.softplus(posterior_logvars))

        loss = reconstruction_loss + kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.rssm.parameters(), 1, norm_type=2)
        self.optimizer.step()

        return loss.item(), reconstruction_loss.item(), kl_loss.item()
    
    def eval_batch(self, obs: torch.Tensor, actions: torch.Tensor, dones: torch.Tensor, masks: torch.Tensor):
        seq_len = obs.shape[1]
        batch_size = obs.shape[0]
        actions = actions.clone().detach().float().to(self.device)
        actions = actions.unsqueeze(-1)  

        obs = obs.clone().detach().float().to(self.device)
        dones = dones.clone().detach().float().to(self.device)
        dones = dones.unsqueeze(-1)

        encoded_obs = self.rssm.encoder(obs.reshape(-1, *obs.shape[2:]))
        encoded_obs = encoded_obs.reshape(batch_size, seq_len, -1)

        rollout = self.rssm.generate_rollout(actions, obs=encoded_obs, dones=dones)

        hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = rollout

        hiddens_reshaped = hiddens.reshape(batch_size * seq_len, -1)
        posterior_states_reshaped = posterior_states.reshape(batch_size * seq_len, -1)

        decoded_obs = self.rssm.decoder(hiddens_reshaped, posterior_states_reshaped)
        decoded_obs = decoded_obs.reshape(batch_size, seq_len, *obs.shape[2:])

        reconstruction_loss = self._reconstruction_loss(decoded_obs, obs, masks)
        kl_loss = self._kl_loss(prior_means, F.softplus(prior_logvars), posterior_means, F.softplus(posterior_logvars))

        loss = reconstruction_loss + kl_loss

        return loss.item(), reconstruction_loss.item(), kl_loss.item()
    
    def train(self, epochs: int, batch_size: int, model_path: str = "rssm.pth"):
        self.rssm.train()
        last_loss = float("inf")
        for epoch in range(epochs):
            self.rssm.train()

            dataloader = RoastDataLoader(self.train_dataset, batch_size)
            epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", total=len(dataloader))

            epoch_losses = []
            epoch_reconstruction_losses = []
            epoch_kl_losses = []
            for obs, actions, dones, masks in epoch_iterator:
                loss, reconstruction_loss, kl_loss = self.train_batch(obs, actions, dones, masks)
                epoch_losses.append(loss)
                epoch_reconstruction_losses.append(reconstruction_loss)
                epoch_kl_losses.append(kl_loss)

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_reconstruction_loss = sum(epoch_reconstruction_losses) / len(epoch_reconstruction_losses)
            avg_kl_loss = sum(epoch_kl_losses) / len(epoch_kl_losses)

            self.writer.add_scalar("Avg Loss", avg_loss, epoch)
            self.writer.add_scalar("Avg Reconstruction Loss", avg_reconstruction_loss, epoch)
            self.writer.add_scalar("Avg KL Loss", avg_kl_loss, epoch)

            if avg_loss < last_loss:
                self.rssm.save(model_path)
                last_loss = avg_loss

            logger.info("\n----------------------------")
            logger.info(f"Epoch: {epoch}")
            logger.info(f"Avg Loss: {avg_loss:.4f}")
            logger.info(f"Avg Reconstruction Loss: {avg_reconstruction_loss:.4f}")
            logger.info(f"Avg KL Loss: {avg_kl_loss:.4f}")

            # Validation loop
            self.rssm.eval()
            val_dataloader = RoastDataLoader(self.val_dataset, batch_size)
            val_losses = []
            val_reconstruction_losses = []
            val_kl_losses = []
            with torch.no_grad():
                for obs, actions, dones, masks in val_dataloader:
                    loss, reconstruction_loss, kl_loss = self.eval_batch(obs, actions, dones, masks)
                    val_losses.append(loss)
                    val_reconstruction_losses.append(reconstruction_loss)
                    val_kl_losses.append(kl_loss)

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_reconstruction_loss = sum(val_reconstruction_losses) / len(val_reconstruction_losses)
            avg_val_kl_loss = sum(val_kl_losses) / len(val_kl_losses)

            self.writer.add_scalar("Avg Val Loss", avg_val_loss, epoch)
            self.writer.add_scalar("Avg Val Reconstruction Loss", avg_val_reconstruction_loss, epoch)
            self.writer.add_scalar("Avg Val KL Loss", avg_val_kl_loss, epoch)

            logger.info(f"Avg Val Loss: {avg_val_loss:.4f}")
            logger.info(f"Avg Val Reconstruction Loss: {avg_val_reconstruction_loss:.4f}")
            logger.info(f"Avg Val KL Loss: {avg_val_kl_loss:.4f}")

            # Save model
            self.rssm.save(model_path)

    def _reconstruction_loss(self, decoded_obs, obs, masks):
        mse_loss = nn.MSELoss(reduction='none')

        loss = mse_loss(decoded_obs, obs)
        masked_loss = torch.where(masks, loss, torch.nan)
        mse_loss_val = masked_loss.nanmean()

        return mse_loss_val
    
    def _kl_loss(self, prior_means, prior_logvars, posterior_means, posterior_logvars):
        prior_dist = Normal(prior_means, torch.exp(prior_logvars))
        posterior_dist = Normal(posterior_means, torch.exp(posterior_logvars))

        return kl_divergence(posterior_dist, prior_dist).mean()
    

class Trainer:
    def __init__(self, rssm: RSSM,buffer: Buffer, optimizer: torch.optim.Optimizer, device: torch.device):
        self.rssm = rssm
        self.optimizer = optimizer
        self.device = device

        self.writer = SummaryWriter()
        self.buffer = buffer


    def train_batch(self, batch_size: int, seq_len: int, iteration: int):
        obs, actions, dones = self.buffer.sample(batch_size, seq_len)

        actions = torch.tensor(actions).float().to(self.device)
        actions = actions.unsqueeze(-1)        
        # actions = F.one_hot(actions, self.rssm.action_dim).float()
        

        obs = torch.tensor(obs, requires_grad=True).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)
        dones = dones.unsqueeze(-1)

        encoded_obs = self.rssm.encoder(obs.reshape(-1, *obs.shape[2:]))
        encoded_obs = encoded_obs.reshape(batch_size, seq_len, -1)

        rollout = self.rssm.generate_rollout(actions, obs=encoded_obs, dones=dones)

        hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = rollout

        hiddens_reshaped = hiddens.reshape(batch_size * seq_len, -1)
        posterior_states_reshaped = posterior_states.reshape(batch_size * seq_len, -1)

        decoded_obs = self.rssm.decoder(hiddens_reshaped, posterior_states_reshaped)
        decoded_obs = decoded_obs.reshape(batch_size, seq_len, *obs.shape[2:])



        reconstruction_loss = self._reconstruction_loss(decoded_obs, obs)
        kl_loss = self._kl_loss(prior_means, F.softplus(prior_logvars), posterior_means, F.softplus(posterior_logvars))

        loss = reconstruction_loss + kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.rssm.parameters(), 1, norm_type=2)
        self.optimizer.step()

        return loss.item(), reconstruction_loss.item(), kl_loss.item()

    def train(self, iterations: int, batch_size: int, seq_len: int):
        self.rssm.train()
        iterator = tqdm(range(iterations), desc="Training", total=iterations)
        losses = []
        infos = []
        last_loss = float("inf")
        for i in iterator:
            loss, reconstruction_loss, kl_loss = self.train_batch(batch_size, seq_len, i)

            self.writer.add_scalar("Loss", loss, i)
            self.writer.add_scalar("Reconstruction Loss", reconstruction_loss, i)
            self.writer.add_scalar("KL Loss", kl_loss, i)

            if loss < last_loss:
                self.rssm.save("rssm.pth")
                last_loss = loss

            info = {
                "Loss": loss,
                "Reconstruction Loss": reconstruction_loss,
                "KL Loss": kl_loss,
            }
            losses.append(loss)
            infos.append(info)

            if i % 10 == 0:
                logger.info("\n----------------------------")
                logger.info(f"Iteration: {i}")
                logger.info(f"Loss: {loss:.4f}")
                logger.info(f"Running average last 20 losses: {sum(losses[-20:]) / 20: .4f}")
                logger.info(f"Reconstruction Loss: {reconstruction_loss:.4f}")
                logger.info(f"KL Loss: {kl_loss:.4f}")

                #save model
                self.rssm.save("rssm.pth")

        
        self.writer.close()
        


    def _reconstruction_loss(self, decoded_obs, obs):
        return F.mse_loss(decoded_obs, obs)

    def _kl_loss(self, prior_means, prior_logvars, posterior_means, posterior_logvars):
        prior_dist = Normal(prior_means, torch.exp(prior_logvars))
        posterior_dist = Normal(posterior_means, torch.exp(posterior_logvars))

        return kl_divergence(posterior_dist, prior_dist).mean()


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

    optimizer = torch.optim.Adam(rssm.parameters(), lr=1e-3)
    # buffer = Buffer("augmented_data.npy",(2,), (1,), device)
    # trainer = Trainer(rssm, buffer, optimizer=optimizer, device=device)
    # trainer.train(15000,32, 850)

    data_path = "augmented_data.npy"
    data = np.load(data_path, allow_pickle=True)
    np.random.shuffle(data)
    train_data = data[:int(len(data)*0.8)]
    val_data = data[int(len(data)*0.8):]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_dataset = RoastDataset(train_data, device)
    val_dataset = RoastDataset(val_data, device)
    trainer = Trainerv2(rssm, train_dataset, val_dataset, optimizer=optimizer, device=device)
    trainer.train(2000,4,"rssm_v2.pth")
