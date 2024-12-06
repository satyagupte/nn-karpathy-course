import torch
from engine import Value

def test_grad():
    # micrograd
    x = Value(2.0)
    y = x * 2 + x + 10
    z =  y.relu() + y * x
    z.backward()
    xmg, zmg = x, z

    #torch
    x = torch.Tensor([2.0]).double()
    x.requires_grad = True
    y = x * 2 + x + 10
    z = y.relu() + y * x
    z.backward()
    xpt, zpt = x, z

    # forward pass 
    assert zmg.data == zpt.data.item()

    # backward pass, check gradients
    assert xmg.grad == xpt.grad.item()
    print(xmg.grad)


if __name__ == "__main__":
    test_grad()