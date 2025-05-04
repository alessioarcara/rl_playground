from flax import nnx


class QNetwork(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dmid, rngs=rngs)
        self.linear3 = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        return self.linear3(x)
