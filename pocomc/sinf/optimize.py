import torch
import torch.optim as optim
import time
from .SlicedWasserstein import Stiefel_SGD
from .SINF import SlicedTransport, whiten
import copy
import numpy as np

def calculate_Z(logp, logq, gamma):
    return torch.exp(torch.logsumexp(logp+(1-gamma)*logq, dim=0) - torch.logsumexp((2-gamma)*logq, dim=0))

def lossfunc0(model, x, logp, logq_uw=None, gamma=1., weight=None):
    #loss = E(q^(-gamma)*(p - Zq)^2)
    logq = model.evaluate_density(x)
    Z = calculate_Z(logp, logq, gamma)
    return torch.mean(torch.exp(-gamma*logq)*(torch.exp(logp) - Z*torch.exp(logq))**2), Z

def lossfunc1(model, x, logp, logq_uw=None, gamma=1., weight=None):
    #loss = log E(q^(-gamma)*(p - Zq)^2)
    logq = model.evaluate_density(x)
    Z = calculate_Z(logp, logq, gamma)
    return torch.logsumexp(-gamma*logq + 2*torch.log(torch.abs(torch.exp(logp) - Z*torch.exp(logq))), dim=0) - np.log(len(x)), Z

def lossfunc2(model, x, logp, logq_uw=None, gamma=None, weight=None): 
    #loss = sum((logp-logq-logZ)^2)
    #gamma is not useful here
    logq = model.evaluate_density(x)
    logZ = torch.mean(logp-logq)
    return torch.mean((logp-logq-logZ)**2), torch.exp(logZ)

def lossfunc3(model, x, logp=None, logq_uw=None, gamma=None, weight=None): 
    #loss = -E(logq)
    #logp and gamma are not useful here
    logq = model.evaluate_density(x)
    if weight is None:
        return -torch.mean(logq), torch.tensor(1)
    else:
        return -torch.sum(weight*logq) / torch.sum(weight), torch.tensor(1)

def lossfunc4(model, x, logp, logq_uw, gamma=None, weight=None):
    #loss = log(E((q/q_uw)(p/q - Z)^2))
    #gamma is not useful here
    logq = model.evaluate_density(x)
    Z = torch.exp(torch.logsumexp(logp - logq_uw, dim=0) - torch.logsumexp(logq - logq_uw, dim=0))
    loss = torch.logsumexp(logq - logq_uw + 2*torch.log(torch.abs(torch.exp(logp.double()-logq.double()) - Z)).float(), dim=0) - np.log(len(x))
    return loss, Z

def loss_grad(model, x, dlogp, Lambda):
    #loss = -sum((dlogp - dlogq)^2)
    x.requires_grad_(True)
    logq = torch.sum(model.evaluate_density(x))
    dlogq = torch.autograd.grad(logq, x, create_graph=True, retain_graph=True)[0]
    x.requires_grad_(False)
    return Lambda * torch.mean((dlogp-dlogq)**2)

def regularization(model, reg, reg1, reg2):
    #regularization = reg * (y-x)^2 + reg1 * (deriv-1)^2 + reg2 * (2nd_derivative_1- 2nd_derivative_2)^2
    loss = 0
    for m in model.layer:
        if isinstance(m, SlicedTransport):
            if reg > 0 or reg1 > 0:
                x, y, deriv = m.transform1D._prepare()
                loss = loss + reg * torch.sum((y-x)**2) + reg1 * torch.sum((deriv-1)**2)
            if reg2 > 0:
                deriv2_1, deriv2_2 = m.transform1D.second_derivative()
                loss = loss + reg2 * torch.sum((deriv2_1-deriv2_2)**2)
    return loss


def optimize_SINF(model, x, logp, logq_uw=None, dlogp=None, lossfunc=0, gamma=0, weight=None, lr=1e-2, lrA=2e-5, Nepoch=200, optimize_A=True, optimizer_Psi=None, optimizer_A=None, logp_threshold=None, val_frac=0, Lambda=0, reg=0, reg1=0, reg2=0, batchsize=None, return_optimizer=False, verbose=True):

    model.train()
    model.requires_grad_(True)

    if logp_threshold is not None and logp is not None:
        select = logp > logp_threshold
        x = x[select]
        logp = logp[select]
        if logq_uw is not None:
            logq_uw = logq_uw[select]
        if Lambda != 0:
            dlogp = dlogp[select]

    assert lossfunc in [0, 1, 2, 3, 4]
    if lossfunc == 0:
        loss_fun = lossfunc0
    elif lossfunc == 1:
        loss_fun = lossfunc1
    elif lossfunc == 2:
        loss_fun = lossfunc2
    elif lossfunc == 3 or logp is None:
        loss_fun = lossfunc3
        logp = torch.zeros(len(x), device=x.device)
    elif lossfunc == 4:
        loss_fun = lossfunc4
        assert logq_uw is not None

    if Lambda != 0:
        assert dlogp is not None

    if val_frac > 0:
        Nval = int(len(x)*val_frac)
        order = torch.randperm(len(x))
        x = x[order]
        logp = logp[order]
        x_val = x[:Nval]
        logp_val = logp[:Nval]
        x = x[Nval:]
        logp = logp[Nval:]
        if logq_uw is not None:
            logq_uw = logq_uw[order]
            logq_uw_val = logq_uw[:Nval]
            logq_uw = logq_uw[Nval:]
        else:
            logq_uw_val = None
        if weight is not None:
            weight = weight[order]
            weight_val = weight[:Nval]
            weight = weight[Nval:]
        else:
            weight_val = None
        if Lambda != 0:
            dlogp = dlogp[order]
            dlogp_val = dlogp[:Nval]
            dlogp = dlogp[Nval:]
    group_A = []
    group_Psi = []

    for m in model.layer:
        if isinstance(m, SlicedTransport):
            group_A.append(m.A)
            for param in m.transform1D.parameters():
                group_Psi.append(param)
        elif isinstance(m, whiten):
            group_A.append(m.E)
            group_Psi.append(m.mean)
            group_Psi.append(m.D)

    if len(group_Psi) == 0:
        print('There are no parameters in SNF!')
        model.requires_grad_(False)
        if return_optimizer:
            return model, loss_fun(model, x, logp, logq_uw, gamma, weight)[1], None, None
        else:
            return model, loss_fun(model, x, logp, logq_uw, gamma, weight)[1]

    if optimize_A and optimizer_A is None:
        optimizer_A = Stiefel_SGD(group_A, lr=lrA, momentum=0.9)
    if optimizer_Psi is None:
        optimizer_Psi = optim.Adam(group_Psi, lr=lr)

    if batchsize is None or batchsize >= len(x):

        best_loss = float('inf') 
        best_state = copy.deepcopy(model.state_dict())
        best_Z = None
        best_optimizer_A = copy.deepcopy(optimizer_A.state_dict()) if optimize_A else None
        best_optimizer_Psi = copy.deepcopy(optimizer_Psi.state_dict())
        for epoch in range(Nepoch):
            t = time.time()

            # zero the parameter gradients
            optimizer_Psi.zero_grad()
            if optimize_A:
                optimizer_A.zero_grad()

            # forward + backward + optimize
            loss, Z = loss_fun(model, x, logp, logq_uw, gamma, weight)
            if Lambda != 0:
                loss = loss + loss_grad(model, x, dlogp, Lambda)
            if reg > 0 or reg1 > 0 or reg2 > 0:
                loss = loss + regularization(model, reg, reg1, reg2)
            if val_frac:
                with torch.no_grad():
                    loss_val, Z_val = loss_fun(model, x_val, logp_val, logq_uw_val, gamma, weight_val)
                    if reg > 0 or reg1 > 0 or reg2 > 0:
                        loss_val = loss_val + regularization(model, reg, reg1, reg2)
                if Lambda != 0:
                    loss_val = loss_val + loss_grad(model, x_val, dlogp_val, Lambda)
            else:
                loss_val, Z_val = loss, Z
            if loss_val < best_loss:
                best_loss = loss_val.item()
                best_Z = Z_val.item()
                best_state = copy.deepcopy(model.state_dict())
                best_optimizer_A = copy.deepcopy(optimizer_A.state_dict()) if optimize_A else None
                best_optimizer_Psi = copy.deepcopy(optimizer_Psi.state_dict())

            loss.backward()
            optimizer_Psi.step()
            if optimize_A:
                optimizer_A.step()

            t = time.time()-t
            if verbose:
                print(f'Epoch {epoch}, Loss {loss:.8f} {loss_val:.8f}, Z {Z:.3f} {Z_val:.3f},Time {t:.3f}')

        if val_frac:
            with torch.no_grad():
                loss_val, Z_val = loss_fun(model, x_val, logp_val, logq_uw_val, gamma, weight_val)
            if Lambda != 0:
                loss_val = loss_val + loss_grad(model, x_val, dlogp_val, Lambda)
            if reg > 0 or reg1 > 0 or reg2 > 0:
                loss_val = loss_val + regularization(model, reg, reg1, reg2)
        else:
            with torch.no_grad():
                loss_val, Z_val = loss_fun(model, x, logp, logq_uw, gamma, weight)
                if reg > 0 or reg1 > 0 or reg2 > 0:
                    loss_val = loss_val + regularization(model, reg, reg1, reg2)
            if Lambda != 0:
                loss_val = loss_val + loss_grad(model, x, dlogp, Lambda)
        if loss_val < best_loss:
            best_loss = loss_val.item()
            best_Z = Z_val.item()
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer_A = copy.deepcopy(optimizer_A.state_dict()) if optimize_A else None
            best_optimizer_Psi = copy.deepcopy(optimizer_Psi.state_dict())

    else:
        if logq_uw is None:
            logq_uw = torch.zeros(len(x), device=x.device)
        if weight is None:
            weight = torch.ones(len(x), device=x.device) / len(x)
        if Lambda == 0:
            trainset = torch.utils.data.TensorDataset(x, logp, logq_uw, weight)
        else:
            trainset = torch.utils.data.TensorDataset(x, logp, logq_uw, weight, dlogp)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, drop_last=True)

        if val_frac:
            with torch.no_grad():
                loss_val, Z_val = loss_fun(model, x_val, logp_val, logq_uw_val, gamma, weight_val)
            if Lambda != 0:
                loss_val = loss_val + loss_grad(model, x_val, dlogp_val, Lambda)
            if reg > 0 or reg1 > 0 or reg2 > 0:
                loss_val = loss_val + regularization(model, reg, reg1, reg2)
        else:
            with torch.no_grad():
                loss_val, Z_val = loss_fun(model, x, logp, logq_uw, logq_uw, gamma, weight)
                if reg > 0 or reg1 > 0 or reg2 > 0:
                    loss_val = loss_val + regularization(model, reg, reg1, reg2)
            if Lambda != 0:
                loss_val = loss_val + loss_grad(model, x, dlogp, Lambda)
        best_loss = loss_val.item()
        best_Z = Z_val.item()
        best_state = copy.deepcopy(model.state_dict())
        best_optimizer_A = copy.deepcopy(optimizer_A.state_dict()) if optimize_A else None
        best_optimizer_Psi = copy.deepcopy(optimizer_Psi.state_dict())
        if verbose:
            print(f'Initial Loss {loss_val:.8f}')

        for epoch in range(Nepoch):
            t = time.time()
            for i, data in enumerate(trainloader, 0):
                if Lambda == 0:
                    x0, logp0, logq_uw0, weight0 = data
                else:
                    x0, logp0, logq_uw0, weight0, dlogp0 = data

                # zero the parameter gradients
                optimizer_Psi.zero_grad()
                if optimize_A:
                    optimizer_A.zero_grad()

                # forward + backward + optimize
                loss, Z = loss_fun(model, x0, logp0, logq_uw0, gamma, weight0)
                if Lambda != 0:
                    loss = loss + loss_grad(model, x0, dlogp0, Lambda)
                if reg > 0 or reg1 > 0 or reg2 > 0:
                    loss = loss + regularization(model, reg, reg1, reg2)
                loss.backward()
                optimizer_Psi.step()
                if optimize_A:
                    optimizer_A.step()

            if val_frac:
                with torch.no_grad():
                    loss_val, Z_val = loss_fun(model, x_val, logp_val, logq_uw_val, gamma, weight_val)
                    if reg > 0 or reg1 > 0 or reg2 > 0:
                        loss_val = loss_val + regularization(model, reg, reg1, reg2)
                if Lambda != 0:
                    loss_val = loss_val + loss_grad(model, x_val, dlogp_val, Lambda)
            else:
                with torch.no_grad():
                    loss_val, Z_val = loss_fun(model, x, logp, logq_uw, gamma, weight)
                    if reg > 0 or reg1 > 0 or reg2 > 0:
                        loss_val = loss_val + regularization(model, reg, reg1, reg2)
                if Lambda != 0:
                    loss_val = loss_val + loss_grad(model, x, dlogp, Lambda)

                if loss_val < best_loss:
                    best_loss = loss_val.item()
                    best_Z = Z_val.item()
                    best_state = copy.deepcopy(model.state_dict())
                    best_optimizer_A = copy.deepcopy(optimizer_A.state_dict()) if optimize_A else None
                    best_optimizer_Psi = copy.deepcopy(optimizer_Psi.state_dict())
                t = time.time()-t
                if verbose:
                    print(f'Epoch {epoch}, Loss {loss_val:.8f}, Z {Z_val:.3f},Time {t:.3f}')

    model.load_state_dict(best_state)
    if optimize_A:
        optimizer_A.load_state_dict(best_optimizer_A)
    optimizer_Psi.load_state_dict(best_optimizer_Psi)
    if return_optimizer:
        return model.requires_grad_(False), best_Z, optimizer_Psi, optimizer_A
    else:
        return model.requires_grad_(False), best_Z
