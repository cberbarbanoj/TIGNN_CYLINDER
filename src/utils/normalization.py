import torch
import torch.nn as nn

class Normalizer(nn.Module):
    def __init__(self, size, max_accumulations=10**6, std_epsilon=1e-8, name='Normalizer', device='cpu'):
        super(Normalizer, self).__init__()
        self.name=name
        self.max_accumulations = max_accumulations
        self.std_epsilon = torch.tensor(std_epsilon, dtype=torch.float32, requires_grad=False, device=device)
        self.acc_count = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device))
        self.num_accumulations = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device))
        self.acc_sum = torch.nn.Parameter(torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device))
        self.acc_sum_squared = torch.nn.Parameter(torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device))

    def forward(self, batched_data, mode=True):
        """Normalizes input data and accumulates statistics."""
        if mode=='train':
        # stop accumulating after a million updates, to prevent accuracy issues
            if self.num_accumulations.data < self.max_accumulations:
                self._accumulate(batched_data.detach())
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data):
        """Function to perform the accumulation of the batch_data statistics."""
        count = batched_data.shape[0]
        data_sum = torch.sum(batched_data, axis=0, keepdims=True)
        squared_data_sum = torch.sum(batched_data**2, axis=0, keepdims=True)

        self.acc_sum.data += data_sum
        self.acc_sum_squared.data += squared_data_sum
        self.acc_count.data += count
        self.num_accumulations.data += 1

    def _mean(self):
        safe_count = torch.maximum(self.acc_count.data, torch.tensor(1.0, dtype=torch.float32, device=self.acc_count.data.device))
        return self.acc_sum.data / safe_count

    def _std_with_epsilon(self):
        safe_count = torch.maximum(self.acc_count.data, torch.tensor(1.0, dtype=torch.float32, device=self.acc_count.data.device))
        std = torch.sqrt(self.acc_sum_squared.data / safe_count - self._mean() ** 2)
        return torch.maximum(std, self.std_epsilon)

    def get_variable(self):
        
        dict = {'_max_accumulations':self.max_accumulations,
        '_std_epsilon':self.std_epsilon,
        '_acc_count': self.acc_count.data,
        '_num_accumulations':self.num_accumulations.data,
        '_acc_sum': self.acc_sum.data,
        '_acc_sum_squared':self.acc_sum_squared.data,
        'name':self.name
        }

        return dict