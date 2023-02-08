from dataclasses import dataclass

import torch as t
from torch.utils.data.dataloader import DataLoader


class ExtendedDataLoader(DataLoader):
    """
    A class which wraps dataloader that ensures that the order of samples is independent of the batch size.
    Also allows for easier seed management.
    """

    def __init__(self, *args, seed_shuffle=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed_shuffle = seed_shuffle
        self.step = 0

    @property
    def hyperparams(self):
        return {
            "batch_size": self.batch_size,
            "seed_shuffle": self.seed_shuffle,
        }

    # def __iter__(self):
    #     t.manual_seed(self.seed_shuffle + self.step)

    #     # First, collect the indices of the samples in the dataset
    #     indices = list(range(len(self.dataset)))

    #     # Shuffle the indices (independent of batch_size)
    #     t.randperm(len(indices), out=t.LongTensor(indices))

    #     indices = [
    #         indices[i : i + self.batch_size]
    #         for i in range(0, len(indices), self.batch_size)
    #     ]

    #     for batch in indices:
    #         # Yield the minibatches (x, y)
    #         x = t.tensor([self.dataset[i][0] for i in batch])
    #         y = t.tensor([self.dataset[i][1] for i in batch])
            
    #         yield x, y

    #         self.step += 1


    # TODO: Double check sampling strategy (is it consistent across batch sizes?)

    def __iter__(self):
        t.manual_seed(self.seed_shuffle + self.step)
        
        for batch in super().__iter__():
            yield batch
            self.step += 1
