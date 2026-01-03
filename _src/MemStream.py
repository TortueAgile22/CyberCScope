import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

# Device par défaut
default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MemStream(nn.Module):
    def __init__(self, in_dim, params):
        super(MemStream, self).__init__()
        self.params = params
        self.in_dim = in_dim
        self.out_dim = in_dim*2
        self.memory_len = params['memory_len']
        
        self.device = params.get('device', default_device)
        
        self.max_thres = torch.tensor(params['beta']).to(self.device)
        self.memory = torch.randn(self.memory_len, self.out_dim).to(self.device)
        self.mem_data = torch.randn(self.memory_len, self.in_dim).to(self.device)
        self.memory.requires_grad = False
        self.mem_data.requires_grad = False
        self.batch_size = params['memory_len']
        self.num_mem_update = 0
        
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.Tanh(),
        ).to(self.device)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.out_dim, self.in_dim)
        ).to(self.device)
        
        self.clock = 0
        self.last_update = -1
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['lr'])
        self.loss_fn = nn.MSELoss()
        self.count = 0

    def train_autoencoder(self, data, epochs):
        self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
        new = (data - self.mean) / self.std
        new[:, self.std == 0] = 0
        new = Variable(new).to(self.device)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            noise = 0.001 * torch.randn_like(new).to(self.device)
            output = self.decoder(self.encoder(new + noise))
            loss = self.loss_fn(output, new)
            loss.backward()
            self.optimizer.step()

    def update_memory(self, output_loss, encoder_output, data):
        if output_loss <= self.max_thres:
            least_used_pos = self.count % self.memory_len
            # CORRECTION : on détache pour ne pas garder l'historique du gradient
            self.memory[least_used_pos] = encoder_output.detach()
            self.mem_data[least_used_pos] = data
            self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
            self.count += 1
            return 1
        return 0

    def initialize_memory(self, x):
        mean, std = self.mem_data.mean(0), self.mem_data.std(0)
        new = (x - mean) / std
        new[:, std == 0] = 0
        
        # CORRECTION : Ajout de .detach() ici pour éviter l'erreur RuntimeError
        self.memory = self.encoder(new.to(self.device)).detach()
        self.memory.requires_grad = False
        self.mem_data = x.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        new = (x - self.mean) / self.std
        new[:, self.std == 0] = 0
        
        encoder_output = self.encoder(new)
        loss_values = torch.norm(self.memory - encoder_output, dim=1, p=1).min()
        
        self.update_memory(loss_values, encoder_output, x)
        
        return loss_values