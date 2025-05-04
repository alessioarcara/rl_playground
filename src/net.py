from typing import Sequence

from flax import nnx
from loguru import logger


# Universal approximation theorem
class FullyConnectedNetwork(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(din, dmid, rngs=rngs)
        self.fc2 = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.fc1(x))
        return self.fc2(x)


# Playing Atari with Deep Reinforcement Learning implementation
class ConvNetwork(nnx.Module):
    def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
        H, W, C = din  # 84, 84, 4
        logger.debug("H, W, C : {}, {}, {}", H, W, C)
        self.conv1 = nnx.Conv(
            C, 32, kernel_size=(8, 8), strides=4, padding="VALID", rngs=rngs
        )  # H1 = (H - 8) // 4 + 1 = 20
        self.conv2 = nnx.Conv(
            32, 64, kernel_size=(4, 4), strides=2, padding="VALID", rngs=rngs
        )  # H2 = (H1 - 4) // 2 + 1 = 9
        self.conv3 = nnx.Conv(
            64, 64, kernel_size=(3, 3), strides=1, padding="VALID", rngs=rngs
        )  # H3 = (H2 - 3) // 1 + 1 = 7
        self.fc1 = nnx.Linear(64 * 7 * 7, 512, rngs=rngs)
        self.fc2 = nnx.Linear(512, dout, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.fc1(x))
        return self.fc2(x)


class GridNet(nnx.Module):
    def __init__(self, din: Sequence[int], dout: int, rngs: nnx.Rngs):
        H, W, C = din
        self.conv1 = nnx.Conv(
            C,
            32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            32, 64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs
        )
        self.fc1 = nnx.Linear(64 * H * W, 128, rngs=rngs)
        self.out = nnx.Linear(128, dout, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.fc1(x))
        return self.out(x)
