import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical

class PPO(nn.Module):
    """Implementation of a PPO model. The same backbone is used to get actor and critic values."""

    def __init__(self, in_shape, n_actions, hidden_d=100, share_backbone=False):
        # Super constructor
        super(PPO, self).__init__()

        # Attributes
        self.in_shape = in_shape
        self.n_actions = n_actions
        self.hidden_d = hidden_d
        self.share_backbone = share_backbone

        # Shared backbone for policy and value functions
        in_dim = np.prod(in_shape)

        def to_features():
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, hidden_d),
                nn.ReLU(),
                nn.Linear(hidden_d, hidden_d),
                nn.ReLU()
            )

        self.backbone = to_features() if self.share_backbone else nn.Identity()

        # State action function
        self.actor = nn.Sequential(
            nn.Identity() if self.share_backbone else to_features(),
            nn.Linear(hidden_d, hidden_d),
            nn.ReLU(),
            nn.Linear(hidden_d, n_actions),
            nn.Softmax(dim=-1)
        )

        # Value function
        self.critic = nn.Sequential(
            nn.Identity() if self.share_backbone else to_features(),
            nn.Linear(hidden_d, hidden_d),
            nn.ReLU(),
            nn.Linear(hidden_d, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        action = self.actor(features)
        value = self.critic(features)
        return Categorical(action).sample(), action, value