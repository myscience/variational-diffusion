
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from typing import Callable

from lightning import LightningDataModule

class CIFAR10DM(LightningDataModule):

    def __init__(
        self,
        root : str,
        download : bool = False,
        
        batch_size : int = 16,
        num_workers : int = 0,
        train_shuffle : bool = True,
        val_shuffle   : bool = False,
        val_batch_size : int | None = None,
        worker_init_fn : Callable | None = None,
        collate_fn     : Callable | None = None,
        train_sampler  : Callable | None = None, 
        val_sampler    : Callable | None = None,
        test_sampler   : Callable | None = None,
    ) -> None:
        super().__init__()

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x : 2 * x - 1.)
        ])

        self.root = root
        self.download = download
        self.transform = transform

        self.num_workers    = num_workers
        self.batch_size     = batch_size
        self.train_shuffle  = train_shuffle
        self.val_shuffle    = val_shuffle
        self.train_sampler  = train_sampler
        self.valid_sampler  = val_sampler
        self.test__sampler  = test_sampler
        self.collate_fn     = collate_fn
        self.worker_init_fn = worker_init_fn
        self.val_batch_size = val_batch_size

    def setup(self, stage = None):
        cifar_train = CIFAR10(self.root,
                        train = True,
                        transform = self.transform,
                        download=self.download,
                    )

        cifar_val = CIFAR10(self.root,
                        train = False,
                        transform = self.transform,
                        download=self.download,
                    )

        # Assign train/val datasets for use in dataloader
        if stage == "fit" or stage is None:
            self.train_dataset = cifar_train
            self.valid_dataset = cifar_val

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = cifar_val

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            sampler        = self.train_sampler,
            batch_size     = self.batch_size,
            shuffle        = self.train_shuffle,
            collate_fn     = self.collate_fn,
            num_workers    = self.num_workers,
            worker_init_fn = self.worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            sampler        = self.valid_sampler,
            batch_size     = self.val_batch_size,
            shuffle        = self.val_shuffle,
            collate_fn     = self.collate_fn,
            num_workers    = self.num_workers,
            worker_init_fn = self.worker_init_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            sampler        = self.test__sampler,
            batch_size     = self.val_batch_size,
            shuffle        = self.val_shuffle,
            collate_fn     = self.collate_fn,
            num_workers    = self.num_workers,
            worker_init_fn = self.worker_init_fn,
        )