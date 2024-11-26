import glob
import pickle
import numpy as np
from einops import rearrange, repeat

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, Dataset

import hydra
from omegaconf import DictConfig, OmegaConf


class VRTaskTrial:
    def __init__(
        self,
        session: str,
        neural: np.ndarray,
        behavioral: np.ndarray,
        masked: bool, cues: np.ndarray
    ):
        self.session = session
        self.neural = neural
        self.behavioral = behavioral
        self.masked = masked
        self.cues = cues

    def num_neural_ts(self):
        return self.neural.shape[0]
    
    def num_neurons(self):
        return self.neural.shape[1]
    
    def num_behavioral_ts(self):
        return self.behavioral.shape[0]

    def __len__(self):
        return self.neural.shape[0]


class DomaineDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
    
    def load_data(self):
        files = glob.glob(self.data_path + '/*.pkl')
        self.data = []
        for file in files:
            session_name = file.split('.')[0]
            session_obj = pickle.load(open(file, 'rb'))
            num_trials = session_obj['num_trials']
            for trial in range(num_trials):
                trial_obj = VRTaskTrial(
                    session=session_name,
                    neural=session_obj['df'][trial],
                    behavioral=session_obj['position'][trial],
                    masked=session_obj['trial_mask'][trial],
                    cues=session_obj['cue_onsets'][trial]
                )
                self.data.append(trial_obj)
        return self.data

    def __len__(self):
        if self.data is None:
            return 0
        return len(self.data)

    def __getitem__(self, idx):
        trial_data = self.data[idx]
        return trial_data.__dict__


class DopamineDataModule(L.LightningDataModule):
    def __init__(self, dataset: Dataset, batch_size: int):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def prepare_data(self):
        if len(self.dataset) == 0:
            self.dataset.load_data()

    def setup(self, stage=None):
        train_size = int(0.7 * len(self.dataset))
        val_size = int(0.1 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class TrainWrapper(L.LightningModule):
    def __init__(self, cfg, encoder, predictor, decoder):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.predictor = predictor
        self.decoder = decoder

    def configure_optimizers(self):
        pass
    
    def training_step(self, batch, batch_idx):
        self.encoder(batch)

    def on_validation_epoch_start(self):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_start(self):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass


class VAEEncoder(nn.Module):
    """
    Encoder for Variational Autoencoder (VAE)
    """

    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, latent_size)
    
    def forward(self, x):
        breakpoint()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(cfg : DictConfig):
    dataset = DomaineDataset(data_path=cfg.data_root)
    data_module = DopamineDataModule(dataset, batch_size=cfg.batch_size)
    
    vae_encoder = VAEEncoder(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        latent_size=cfg.latent_size
    )
    wrapper = TrainWrapper(cfg, encoder=vae_encoder, predictor=None, decoder=None)

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        # progress_bar_refresh_rate=cfg.progress_bar_refresh_rate,
        # devices='cpu'
    )
    trainer.fit(wrapper, data_module)


if __name__ == '__main__':
    main()
    