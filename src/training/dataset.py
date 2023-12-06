import torch

import numpy as np
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from src.config import FT_COLUMNS, LABELS_COLUMN
from src.helper.plot import plot
from src.helper.data import load_event, load_full_dataset
from src.training.transform import RandomCrop1D

TEST_SIZE = 1
VAL_SIZE = 0.20
CROP_SIZE = int(7 * 24 * 3600 / 5)  # 7 days

# Luckily it fits in memory
FULL_DATASET = load_full_dataset()


class DSSDataset(Dataset):
    def __init__(self, event_id_list, transform=[], train=True, use_cache=False):
        self.event_id_list = event_id_list
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.event_id_list)

    def __getitem__(self, idx):
        event_id = self.event_id_list[idx]

        df = FULL_DATASET[event_id]
        metadatas = []

        for transform in self.transform:
            df, transform_metadata = transform(df)
            metadatas.append(transform_metadata)

        data = df[FT_COLUMNS].values
        labels = df[LABELS_COLUMN].values

        return (data, labels, (event_id, metadatas)) if self.train else data

    def plot_random(self, start=0, end=0):
        selected_event = np.random.choice(self.event_id_list, size=1)[0]

        df = load_event(selected_event)

        for transform in self.transform:
            df, _ = transform(df)

        plot(df, start=start, end=end)


def collate_fn(batch):
    events = [torch.from_numpy(item[0]) for item in batch]
    targets = [torch.from_numpy(item[1]) for item in batch]
    metadata = [item[2] for item in batch]

    # Pad sequences to the length of the longest sequence in the batch
    padded_events = pad_sequence(events, batch_first=True)
    padded_targets = pad_sequence(targets, batch_first=True)

    return padded_events, padded_targets, metadata


class DSSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
    ):
        super().__init__()
        self.event_id_list = list(FULL_DATASET.keys())
        self.event_id_list.sort()
        self.batch_size = batch_size

        self.transform = [RandomCrop1D(CROP_SIZE)]

    def setup(self, stage=None):
        train_ids = self.event_id_list[:-TEST_SIZE]
        test_ids = self.event_id_list[-TEST_SIZE:]

        if stage == "fit" or stage is None:
            self.full = DSSDataset(train_ids, train=True, transform=self.transform)
            self.train, self.val = random_split(self.full, [1 - VAL_SIZE, VAL_SIZE])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = DSSDataset(test_ids, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=collate_fn)
