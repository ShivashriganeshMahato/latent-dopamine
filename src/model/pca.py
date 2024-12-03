from einops import rearrange, repeat
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

import torch.nn as nn
import torch.nn.functional as F


class PCAEncoder(nn.Module):
    """
    Encoder for PCA
    """

    def __init__(self, num_components, data_module):
        super().__init__()
        self.num_components = num_components
        self.data_module = data_module
        self.pca = None
    
    def on_train_start(self):
        self.pca = dict()
        for session in self.data_module.train_dataset:
            neural = session.neural.df
            centered = neural - neural.mean(axis=0, keepdims=True)

            ############################################################
            pca_full = PCA()
            pca_full.fit(centered)

            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            num_components = np.argmax(cumulative_variance >= 0.90) + 1

            pca = PCA(n_components=num_components)
            pca.fit(centered)

            self.pca[session.session_id] = pca
            ############################################################
            # self.pca[session.session_id] = PCA(n_components=self.num_components)
            # self.pca[session.session_id].fit(centered)
            ############################################################

            model = LinearRegression()
            y = session.behavior.position
            print(centered.shape, y.shape)
            model.fit(centered, y)
            y_pred = model.predict(centered)
            r2 = r2_score(y, y_pred)
            print(session.session_id, r2)

    def forward(self, trial):
        exit()
        neural = trial.neural.df
        centered = neural - neural.mean(axis=0, keepdims=True)
        latents = self.pca[trial.session_id].transform(centered)
        return latents