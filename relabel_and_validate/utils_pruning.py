import torch
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler

class EpochSampler(RandomSampler):
    def __init__(self, data_source, generator=None, initial_epoch=0):
        super().__init__(data_source, generator=generator)
        self.epoch = initial_epoch
        self.indices_epoch = {}

    def set_epoch(self, epoch):
        """Set the current epoch to sample the data, for epoch tracking."""
        self.epoch = epoch

    def __iter__(self):
        # If the epoch's indices are already stored, use them;
        # otherwise, use the parent class's iterator to generate and store them.
        if self.epoch in self.indices_epoch:
            indices = self.indices_epoch[self.epoch]
        else:
            indices = list(super().__iter__())  # Generate indices using RandomSampler's mechanism
            self.indices_epoch[self.epoch] = indices
        return iter(indices)

class EpochBatchSampler(RandomSampler):
    """
    Allow the sampler to be specified for a certain epoch or batch
    """
    def __init__(self, data_source, generator=None, initial_epoch=0, use_batch=False):
        super().__init__(data_source, generator=generator)
        self.epoch = initial_epoch
        self.indices_epoch = {}
        # batch related
        self.batch_num = -1
        self.batch_num_per_epoch = -1
        self.batch_size = -1
        self.batch_idx = -1 # use to set the batch index
        self.batch_list = None  # recieve a list of batch indices
                                # to ensemble an epoch
        self.indices_batch = None

    def use_batch(self, batch_size):
        self.batch_size = batch_size
        self._create_batch_indices(batch_size)
        # set sampler length to 1
    
    def set_batch(self, batch_idx):
        self.batch_idx = batch_idx
        self.batch_list = None
        
    def set_batch_list(self, batch_list):
        """Receive a list of batch indices to ensemble an epoch"""
        self.batch_list = batch_list
        self.batch_idx = -1
    
    def get_batch_list_img_mapping(self):
        return {self.indices_batch[batch_idx][0]: batch_idx for batch_idx in self.batch_list}

    def _create_batch_indices(self, batch_size):
        # create the indices for the batch
        self.indices_batch = {}
        batch_index = 0
        for epoch in range(max(self.indices_epoch.keys()) + 1):
            epoch_indices = self.indices_epoch[epoch]

            for i in range(0, len(epoch_indices), self.batch_size):
                self.indices_batch.update({batch_index: epoch_indices[i:i + self.batch_size]})
                batch_index += 1  # Increment batch index for the next batch
            
        self.batch_num = self.indices_batch.keys().__len__()

    def set_epoch(self, epoch):
        """Set the current epoch to sample the data, for epoch tracking."""
        self.epoch = epoch

    def __iter__(self):
        # If the epoch's indices are already stored, use them;
        # otherwise, use the parent class's iterator to generate and store them.
        if self.indices_batch is not None:
            if self.batch_list is not None:
                """ensemble the batches"""
                indices = [self.indices_batch[batch_idx] for batch_idx in self.batch_list]
                # flatten the list
                indices = [item for sublist in indices for item in sublist]
            else:
                """use the batch indices if it is set"""
                indices = self.indices_batch[self.batch_idx]
        elif self.epoch in self.indices_epoch:
            indices = self.indices_epoch[self.epoch]
        else:
            indices = list(super().__iter__())  # Generate indices using RandomSampler's mechanism
            self.indices_epoch[self.epoch] = indices
        return iter(indices)