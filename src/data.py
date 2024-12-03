import glob
import pickle
import numpy as np
from typing import List, Iterator

from lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader, Dataset, Sampler

import h5py
from temporaldata import Data, Interval


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


class DopamineDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
    
    def load_data(self):
        files = glob.glob(self.data_path + '/*.h5')
        self.data = []
        for file in files:
            f = h5py.File(file, 'r')
            session = Data.from_hdf5(f)
            session.materialize()
            self.data.append(session)
        return self.data

    def __len__(self):
        if self.data is None:
            return 0
        return len(self.data)

    def __getitem__(self, idx):
        trial_data = self.data[idx]
        return trial_data.__dict__


class TrialSampler(Sampler[List[int]]):
    def __init__(self, data: Dataset, batch_size: int):
        self.data = data
        self.batch_size = batch_size

    def __len__(self) -> int:
        num_trials = len(self.data)
        return num_trials // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        trial_indices = torch.randperm(len(self.data))
        for batch in torch.chunk(trial_indices, self.batch_size):
            yield batch


class DopamineDataModule(LightningDataModule):
    def __init__(self, dataset: Dataset, sampler_cls, batch_size: int):
        super().__init__()
        self.dataset = dataset
        self.sampler_cls = sampler_cls
        self.batch_size = batch_size

    def prepare_data(self):
        if len(self.dataset) == 0:
            self.dataset.load_data()

    def setup(self, stage=None):
        train_sets, val_sets, test_sets = [], [], []
        for session in self.dataset.data:
            num_trials = len(session.trials)
            num_train_trials = int(num_trials * 0.7)
            num_val_trials = int(num_trials * 0.1)
            
            train_interval = Interval(session.trials.start[0], session.trials.end[num_train_trials - 1])
            val_interval = Interval(session.trials.start[num_train_trials], session.trials.end[num_train_trials + num_val_trials - 1])
            test_interval = Interval(session.trials.start[num_train_trials + num_val_trials], session.trials.end[-1])
            
            train_sets.append(session.select_by_interval(train_interval))
            val_sets.append(session.select_by_interval(val_interval))
            test_sets.append(session.select_by_interval(test_interval))
        
        self.train_dataset = train_sets
        self.val_dataset = val_sets
        self.test_dataset = test_sets

        print(f"Train sessions: {len(self.train_dataset)}")
        print(f"Train trials: {sum([len(session.trials) for session in self.train_dataset])}")
        print(f"Train total length: {sum([np.sum(session.trials.end - session.trials.start) for session in self.train_dataset])}")
        print(f"Val sessions: {len(self.val_dataset)}")
        print(f"Val trials: {sum([len(session.trials) for session in self.val_dataset])}")
        print(f"Val total length: {sum([np.sum(session.trials.end - session.trials.start) for session in self.val_dataset])}")
        print(f"Test sessions: {len(self.test_dataset)}")
        print(f"Test trials: {sum([len(session.trials) for session in self.test_dataset])}")
        print(f"Test total length: {sum([np.sum(session.trials.end - session.trials.start) for session in self.test_dataset])}")

    def collate_fn(self, batch):
        return batch

    def train_dataloader(self):
        unraveled = [
            session.select_by_interval(Interval(start, end))
            for session in self.train_dataset
            for (start, end) in zip(session.trials.start, session.trials.end)
        ]
        # sampler = self.sampler_cls(unraveled, self.batch_size)
        return DataLoader(unraveled, batch_size=self.batch_size, collate_fn=self.collate_fn)#, sampler=sampler)

    def val_dataloader(self):
        unraveled = [
            session.select_by_interval(Interval(start, end))
            for session in self.val_dataset
            for (start, end) in zip(session.trials.start, session.trials.end)
        ]
        # sampler = self.sampler_cls(unraveled, self.batch_size)
        return DataLoader(unraveled, batch_size=self.batch_size, collate_fn=self.collate_fn)#, sampler=sampler)
    
    def test_dataloader(self):
        unraveled = [
            session.select_by_interval(Interval(start, end))
            for session in self.test_dataset
            for (start, end) in zip(session.trials.start, session.trials.end)
        ]
        # sampler = self.sampler_cls(unraveled, self.batch_size)
        return DataLoader(unraveled, batch_size=self.batch_size, collate_fn=self.collate_fn)#, sampler=sampler)