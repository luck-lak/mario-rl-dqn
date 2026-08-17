from torch import nn
import copy


class MarioNet(nn.Module):
    """
    小型 CNN:
    input -> (conv + relu) x 3 -> flatten -> dense -> output
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84 or w != 84:
            raise ValueError(f"Expecting input shape (*, 84, 84), got ({c}, {h}, {w})")

        self.online = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, model="online"):
        if model == "online":
            return self.online(x)
        if model == "target":
            return self.target(x)
        raise ValueError("model must be 'online' or 'target'")
