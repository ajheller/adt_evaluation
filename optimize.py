#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 02:48:26 2019

@author: heller
"""

# install pytorch with
#    conda install pytorch torchvision -c pytorch

import torch
import spherical_grids as sg
import real_spherical_harmonics as rsh
import basic_decoders as bd


if False:
    x = torch.rand(5, 3)
    print(x)

    T = sg.t_design5200()

    Tt = torch.Tensor(T.u)

    l, m = zip(*rsh.lm_generator(6))

    Y = torch.Tensor(rsh.real_sph_harm_transform(l, m, T.az, T.el))

    S240 = sg.t_design240()
    Su = torch.Tensor(S240.u)

    M = torch.Tensor(bd.inversion(l, m, S240.az, S240.el))

    G = M @ Y

    P = torch.sum(G, 0)
    rV = (Su @ G) / P  # pytorch does the broadcasting

    G2 = G * G

    E = torch.sum(G2, 0)

    rE = (Su @ G2) / E

    rEaz, rEel, rEr = sg.cart2sph(*rE)
    rEu = sg.sph2cart(rEaz, rEel, 1)



class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
