import lightning as L
import torch
import torch.nn as nn

import hydra
from omegaconf import DictConfig, OmegaConf

from wrapper import TrainWrapper
from data import DopamineDataset, DopamineDataModule
from model import PCAEncoder


# class VAEEncoder(nn.Module):
#     """
#     Encoder for Variational Autoencoder (VAE)
#     """

#     def __init__(self, input_size, hidden_size, latent_size):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.latent_size = latent_size
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, latent_size)
    
#     def forward(self, x):
#         breakpoint()
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(cfg : DictConfig):
    torch.manual_seed(42)

    dataset = DopamineDataset(data_path=cfg.data_root)
    data_module = DopamineDataModule(dataset, None, batch_size=cfg.batch_size)

    pca_encoder = PCAEncoder(
        num_components=cfg.model.num_components,
        data_module=data_module,
    )
    linear_decoder = nn.Linear(
        in_features=cfg.model.num_components,
        out_features=2,
    )
    
    wrapper = TrainWrapper(
        cfg,
        encoder=pca_encoder,
        predictor=None,
        decoder=linear_decoder,
    )

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        # progress_bar_refresh_rate=cfg.progress_bar_refresh_rate,
        # devices='cpu'
    )
    trainer.fit(wrapper, data_module)


if __name__ == '__main__':
    main()
    