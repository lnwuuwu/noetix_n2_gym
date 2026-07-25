import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class AMPDiscriminator(nn.Module):
    def __init__(self, input_dim=110, hidden_dims=[1024, 512]):
        super().__init__()
        
        layers = []
        curr_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(spectral_norm(nn.Linear(curr_dim, hidden_dim)))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
            
        layers.append(spectral_norm(nn.Linear(curr_dim, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
        
    def compute_style_reward(self, s_t, s_t1):
        """
        Compute style reward: max(0, 1 - 0.25 * (D(s) - 1)^2)
        Input: s_t (current amp obs), s_t1 (next amp obs)
        """
        x = torch.cat([s_t, s_t1], dim=-1)
        d_val = self.forward(x)
        style_reward = torch.clamp(1.0 - 0.25 * (d_val - 1.0) ** 2, 0.0, 1.0)
        return style_reward.squeeze(-1)

    def compute_grad_pen(self, real_data, fake_data, lambda_gp=10.0):
        """
        WGAN-GP Gradient penalty computation
        """
        alpha = torch.rand(real_data.size(0), 1, device=real_data.device)
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
        d_interpolates = self.forward(interpolates)
        
        fake = torch.ones(real_data.size(0), 1, device=real_data.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
        return gradient_penalty
