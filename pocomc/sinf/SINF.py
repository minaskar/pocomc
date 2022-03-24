import torch
import torch.nn as nn
import numpy as np
import time
import math
from .SlicedWasserstein import *
from .RQspline import *
import torch.multiprocessing as mp
import copy


class SINF(nn.Module):

    #Sliced Iterative Normalizing Flow model

    def __init__(self, ndim):

        super().__init__()

        self.layer = nn.ModuleList([])
        self.ndim = ndim

    def forward(self, data, start=0, end=None, param=None):

        if data.ndim == 1:
            data = data.view(1,-1)
        if end is None:
            end = len(self.layer)
        elif end < 0:
            end += len(self.layer)
        if start < 0:
            start += len(self.layer)

        assert start >= 0 and end >= 0 and end >= start

        logj = torch.zeros(data.shape[0], device=data.device)

        for i in range(start, end):
            data, log_j = self.layer[i](data, param=param)
            logj += log_j

        return data, logj


    def inverse(self, data, start=None, end=0, d_dz=None, param=None):

        if data.ndim == 1:
            data = data.view(1,-1)
        if end < 0:
            end += len(self.layer)
        if start is None:
            start = len(self.layer)
        elif start < 0:
            start += len(self.layer)

        assert start >= 0 and end >= 0 and end <= start

        logj = torch.zeros(data.shape[0], device=data.device)

        for i in reversed(range(end, start)):
            if d_dz is None:
                data, log_j = self.layer[i].inverse(data, param=param)
            else:
                data, log_j, d_dz = self.layer[i].inverse(data, d_dz=d_dz, param=param)
            logj += log_j

        if d_dz is None:
            return data, logj
        else:
            return data, logj, d_dz


    def transform(self, data, start, end, param=None):

        if start is None:
            return self.inverse(data=data, start=start, end=end, param=param)
        elif end is None:
            return self.forward(data=data, start=start, end=end, param=param)
        elif start < 0:
            start += len(self.layer)
        elif end < 0:
            end += len(self.layer)

        if start < 0:
            start = 0
        elif start > len(self.layer):
            start = len(self.layer)
        if end < 0:
            end = 0
        elif end > len(self.layer):
            end = len(self.layer)

        if start <= end:
            return self.forward(data=data, start=start, end=end, param=param)
        else:
            return self.inverse(data=data, start=start, end=end, param=param)


    def add_layer(self, layer, position=None):

        if position is None or position == len(self.layer):
            self.layer.append(layer)
        else:
            if position < 0:
                position += len(self.layer)
            assert position >= 0 and position < len(self.layer)
            self.layer.insert(position, layer)

        return self


    def delete_layer(self, position=-1):

        if position == -1 or position == len(self.layer)-1:
            self.layer = self.layer[:-1]
        else:
            if position < 0:
                position += len(self.layer)
            assert position >= 0 and position < len(self.layer)-1

            for i in range(position, len(self.layer)-1):
                self.layer._modules[str(i)] = self.layer._modules[str(i + 1)]
            self.layer = self.layer[:-1]

        return self


    def evaluate_density(self, data, start=0, end=None, param=None):

        data, logj = self.forward(data, start=start, end=end, param=param)
        logq = -self.ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(data.reshape(len(data), self.ndim)**2,  dim=1)/2
        logp = logj + logq

        return logp


    def loss(self, data, start=0, end=None, param=None):
        return -torch.mean(self.evaluate_density(data, start=start, end=end, param=param))


    def sample(self, nsample, start=None, end=0, device=torch.device('cuda'), param=None):

        #device must be the same as the device of the model

        x = torch.randn(nsample, self.ndim, device=device)
        logq = -self.ndim/2.*torch.log(torch.tensor(2.*math.pi)) - torch.sum(x**2,  dim=1)/2
        x, logj = self.inverse(x, start=start, end=end, param=param)
        logp = logj + logq

        return x, logp


    def score(self, data, start=0, end=None, param=None):

        #returns score = dlogp / dx

        data.requires_grad_(True)
        logp = torch.sum(self.evaluate_density(data, start, end, param))
        score = torch.autograd.grad(logp, data)[0]
        data.requires_grad_(False)

        return score


    def Jacobian_gradient(self, latent, start=None, end=0, param=None):

        #returns dlogj / dz, x, logj

        latent.requires_grad_(True)
        data, logj = self.inverse(latent, start, end, param)
        logJ = torch.sum(logj)
        grad = torch.autograd.grad(logJ, latent)[0]
        latent.requires_grad_(False)

        return grad.detach(), data.detach(), logj.detach()


    def grad_wrt_z(self, latent, grad_wrt_x, start=None, end=0, param=None):

        #returns grad_wrt_x * dx / dz

        latent.requires_grad_(True)
        data = self.inverse(latent, start, end, param)[0]
        grad = torch.autograd.grad(data, latent, grad_outputs=grad_wrt_x)[0]
        latent.requires_grad_(False)

        return grad.detach()



class logit(nn.Module):

    #logit transform

    def __init__(self, lambd=1e-5):

        super().__init__()
        self.lambd = lambd


    def forward(self, data, param=None):

        assert torch.min(data) >= 0 and torch.max(data) <= 1

        data = self.lambd + (1 - 2 * self.lambd) * data
        logj = torch.sum(-torch.log(data*(1-data)) + math.log(1-2*self.lambd), axis=1)
        data = torch.log(data) - torch.log1p(-data)
        return data, logj


    def inverse(self, data, param=None):

        data = torch.sigmoid(data)
        logj = torch.sum(-torch.log(data*(1-data)) + math.log(1-2*self.lambd), axis=1)
        data = (data - self.lambd) / (1. - 2 * self.lambd)
        return data, logj



class boundary(nn.Module):

    #transformation that maps bounded parameters to unbounded.
    #logit transform or inverse softplus transform.

    def __init__(self, bounds, lambd=1e-5, beta=1):

        super().__init__()
        self.bounds = bounds
        self.lambd = lambd
        self.beta = beta


    def forward(self, data, param=None):

        logj = torch.zeros(len(data), device=data.device)
        data = data.clone()
        for i in range(data.shape[1]):
            if self.bounds[i] == [None, None]:
                continue
            elif self.bounds[i][0] is None or self.bounds[i][1] is None:
                if self.bounds[i][1] is None:
                    data[:,i] = data[:,i] - self.bounds[i][0] + self.lambd
                else:
                    data[:,i] = - data[:,i] + self.bounds[i][1] + self.lambd
                z = data[:,i].clone()
                assert (z >= self.lambd).all()
                select = z < 20./self.beta
                z[select] = torch.log(torch.exp(self.beta*z[select].double())-1).float() / self.beta
                logj = logj + self.beta * (data[:,i] - z)
                data[:,i] = z
            else:
                temp = self.lambd + (1 - 2 * self.lambd) * (data[:,i] - self.bounds[i][0]) / (self.bounds[i][1] - self.bounds[i][0])
                logj = logj - torch.log(temp*(1-temp)) + math.log((1-2*self.lambd) / (self.bounds[i][1] - self.bounds[i][0]))
                data[:,i] = torch.log(temp) - torch.log1p(-temp)

        return data, logj


    def inverse(self, data, param=None):

        logj = torch.zeros(len(data), device=data.device)
        data = data.clone()
        for i in range(data.shape[1]):
            if self.bounds[i] == [None, None]:
                continue
            elif self.bounds[i][0] is None or self.bounds[i][1] is None:
                x = data[:,i].clone()
                select = x < 20./self.beta
                x[select] = torch.log(torch.exp(self.beta*x[select].double())+1).float() / self.beta
                logj = logj + self.beta * (x - data[:,i])
                if self.bounds[i][1] is None:
                    data[:,i] = x + self.bounds[i][0] - self.lambd
                else:
                    data[:,i] = - x + self.bounds[i][1] + self.lambd
            else:
                temp = torch.sigmoid(data[:,i])
                logj = logj - torch.log(temp*(1-temp)) + math.log((1-2*self.lambd) / (self.bounds[i][1] - self.bounds[i][0]))
                data[:,i] = (temp - self.lambd) / (1. - 2 * self.lambd) * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]

        return data, logj



class whiten(nn.Module):

    #whiten layer

    def __init__(self, ndim_data, scale=True, ndim_latent=None):

        super().__init__()
        if ndim_latent is None:
            ndim_latent = ndim_data
        assert ndim_latent <= ndim_data
        self.ndim_data = ndim_data
        self.ndim_latent = ndim_latent
        self.scale = scale

        self.mean = nn.Parameter(torch.zeros(ndim_data))
        self.D = nn.Parameter(torch.ones(ndim_data))
        self.E = nn.Parameter(torch.eye(ndim_data))
        select = torch.zeros(ndim_data, dtype=torch.bool)
        select[:ndim_latent] = True
        self.register_buffer('select', select)


    def _loss(self, alpha, data, covariance):
        covariance = (1.-alpha) * covariance + alpha * torch.diag(torch.ones(self.ndim_data, device=data.device))
        D, E = torch.symeig(covariance, eigenvectors=True)
        D1 = D[self.select]**(-0.5)
        data = (torch.diag(D1) @ (E.T @ data.T)[self.select]).T
        logj = torch.repeat_interleave(torch.sum(torch.log(D1)), len(data))
        loss = torch.mean(-logj + torch.sum(data**2, dim=1)/2.)
        return loss.item()


    def fit(self, data, weight=None, regularization=0, data_validate=None):

        assert data.ndim == 2 and data.shape[1] == self.ndim_data

        with torch.no_grad():
            if weight is None:
                self.mean[:] = torch.mean(data, dim=0)
            else:
                self.mean[:] = torch.sum(weight.reshape(-1,1)*data, dim=0) / torch.sum(weight)
            data0 = data - self.mean
            if regularization == 'LW':
                if weight is not None:
                    print('LedoitWolf can only be used for unweighted PCA.')
                    if data_validate is not None:
                        regularization = 'validation'
                    else:
                        regularization = 0
                from sklearn.covariance import LedoitWolf
                covariance = torch.as_tensor(LedoitWolf(assume_centered=True).fit(data).covariance_).float().to(data.dtype)
            elif regularization == 'OAS':
                if weight is not None:
                    print('OAS can only be used for unweighted PCA.')
                    if data_validate is not None:
                        regularization = 'validation'
                    else:
                        regularization = 0
                from sklearn.covariance import OAS
                covariance = torch.as_tensor(OAS(assume_centered=True).fit(data).covariance_).float().to(data.dtype)
            elif regularization == 'NERCOME':
                split = int(len(data)*2/3)
                Z = torch.zeros(self.ndim_data, self.ndim_data, device=data.device)
                N = 500
                for i in range(N):
                    order = torch.randperm(len(data))
                    data0 = data0[order]
                    data1 = data0[:split]
                    data2 = data0[split:]
                    if weight is None:
                        covariance1 = data1.T @ data1 / (split-1)
                        covariance2 = data2.T @ data2 / (len(data)-split-1)
                    else:
                        weight = weight[order]
                        weight1 = weight[:split]
                        weight2 = weight[split:]
                        covariance1 = (weight1 * data1.T) @ data1 / (torch.sum(weight1) - torch.sum(weight1**2) / torch.sum(weight1))
                        covariance2 = (weight2 * data2.T) @ data2 / (torch.sum(weight2) - torch.sum(weight2**2) / torch.sum(weight2))
                    U = torch.linalg.eigh(covariance1)[1]
                    Z = Z + U @ torch.diag(torch.diag(U.T @ covariance2 @ U)) @ U.T
                covariance = Z / N
            elif weight is None:
                covariance = data0.T @ data0 / (len(data0)-1)
            else:
                covariance = (weight * data0.T) @ data0 / (torch.sum(weight) - torch.sum(weight**2) / torch.sum(weight))

            if regularization == 'validation':
                from scipy.optimize import minimize_scalar
                res = minimize_scalar(lambda x: self._loss(x, data_validate, covariance), bounds=(0,1), method='Bounded', options={'maxiter':10})
                alpha = res.x
                covariance = (1.-alpha) * covariance + alpha * torch.diag(torch.ones(self.ndim_data, device=data.device))
            elif not isinstance(regularization, str) and regularization > 0:
                alpha = regularization
                covariance = (1.-alpha) * covariance + alpha * torch.diag(torch.ones(self.ndim_data, device=data.device))
            D, E = torch.symeig(covariance, eigenvectors=True)
            self.D[:] = torch.flip(D, dims=(0,))
            self.E[:] = torch.flip(E, dims=(1,))

            return self


    def forward(self, data, param=None):

        assert data.shape[1] == self.ndim_latent
        data0 = data - self.mean

        if self.scale:
            D1 = self.D[self.select]**(-0.5)
            data0 = (torch.diag(D1) @ (self.E.T @ data0.T)[self.select]).T
            logj = torch.repeat_interleave(torch.sum(torch.log(D1)), len(data))
        else:
            data0 = (self.E.T @ data0.T)[self.select].T
            logj = torch.zeros(len(data), device=data.device)

        return data0, logj


    def inverse(self, data, d_dz=None, param=None):

        #d_dz: (len(data), self.ndim_latent, n_z)

        assert data.shape[1] == self.ndim_latent
        if d_dz is not None:
            assert d_dz.shape[0] == data.shape[0] and data.shape[1] == self.ndim_latent and d_dz.shape[1] == self.ndim_latent

        data0 = torch.zeros([data.shape[0], self.ndim_data], device=data.device)
        data0[:, self.select] = data[:]
        if self.scale:
            D1 = self.D**0.5
            D1[~self.select] = 0.
            data0 = (self.E @ torch.diag(D1) @ data0.T).T
            logj = -torch.repeat_interleave(torch.sum(torch.log(D1[self.select])), len(data))
            if d_dz is not None:
                d_dz = torch.einsum('lj,j,ijk->ilk', self.E[:,self.select], D1[self.select], d_dz)
        else:
            data0 = (self.E @ data0.T).T
            logj = torch.zeros(len(data), device=data.device)
            if d_dz is not None:
                d_dz = torch.einsum('lj,ijk->ilk', self.E[:,self.select], d_dz)
        data0 += self.mean

        if d_dz is None:
            return data0, logj
        else:
            return data0, logj, d_dz



def start_timing(device):
    if torch.cuda.is_available() and device != torch.device('cpu'):
        tstart = torch.cuda.Event(enable_timing=True)
        tstart.record()
    else:
        tstart = time.time()
    return tstart



def end_timing(tstart, device):
    if torch.cuda.is_available() and device != torch.device('cpu'):
        tend = torch.cuda.Event(enable_timing=True)
        tend.record()
        torch.cuda.synchronize()
        t = tstart.elapsed_time(tend) / 1000.
    else:
        t = time.time() - tstart
    return t



def _transform_batch_layer(layer, data, logj, index, batchsize, start_index=0, end_index=None, direction='forward', param=None, nocuda=False):

    if torch.cuda.is_available() and not nocuda:
        gpu = index % torch.cuda.device_count()
        device = torch.device('cuda:%d'%gpu)
    else:
        device = torch.device('cpu')

    layer = layer.to(device)

    if end_index is None:
        end_index = len(data)

    i = 0
    while i * batchsize < end_index-start_index:
        start_index0 = start_index + i * batchsize
        end_index0 = min(start_index + (i+1) * batchsize, end_index)
        if direction == 'forward':
            if param is None:
                data1, logj1 = layer.forward(data[start_index0:end_index0].to(device), param=param)
            else:
                data1, logj1 = layer.forward(data[start_index0:end_index0].to(device), param=param[start_index0:end_index0].to(device))
        else:
            if param is None:
                data1, logj1 = layer.inverse(data[start_index0:end_index0].to(device), param=param)
            else:
                data1, logj1 = layer.inverse(data[start_index0:end_index0].to(device), param=param[start_index0:end_index0].to(device))
        data[start_index0:end_index0] = data1.to(data.device)
        logj[start_index0:end_index0] = logj[start_index0:end_index0] + logj1.to(logj.device)
        i += 1

    del data1, logj1, layer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return


def transform_batch_layer(layer, data, batchsize, logj=None, direction='forward', param=None, pool=None, nocuda=False):

    assert direction in ['forward', 'inverse']

    if logj is None:
        logj = torch.zeros(len(data), device=data.device)

    if pool is None:
        _transform_batch_layer(layer, data, logj, 0, batchsize, direction=direction, param=param, nocuda=nocuda)
    else:
        if torch.cuda.is_available() and not nocuda:
            nprocess = torch.cuda.device_count()
        else:
            nprocess = mp.cpu_count()
        param0 = [(layer, data, logj, i, batchsize, len(data)*i//nprocess, len(data)*(i+1)//nprocess, direction, param, nocuda) for i in range(nprocess)]
        pool.starmap(_transform_batch_layer, param0)

    return data, logj



def _transform_batch_model(model, data, logj, index, batchsize, start_index=0, end_index=None, start=0, end=None, param=None, nocuda=False):

    if torch.cuda.is_available() and not nocuda:
        gpu = index % torch.cuda.device_count()
        device = torch.device('cuda:%d'%gpu)
    else:
        device = torch.device('cpu')

    model = model.to(device)

    if end_index is None:
        end_index = len(data)

    i = 0
    while i * batchsize < end_index-start_index:
        start_index0 = start_index + i * batchsize
        end_index0 = min(start_index + (i+1) * batchsize, end_index)
        if param is None:
            data1, logj1 = model.transform(data[start_index0:end_index0].to(device), start=start, end=end, param=param)
        else:
            data1, logj1 = model.transform(data[start_index0:end_index0].to(device), start=start, end=end, param=param[start_index0:end_index0].to(device))
        data[start_index0:end_index0] = data1.to(data.device)
        logj[start_index0:end_index0] = logj[start_index0:end_index0] + logj1.to(logj.device)
        i += 1

    del data1, logj1, model
    if torch.cuda.is_available() and not nocuda:
        torch.cuda.empty_cache()

    return


def transform_batch_model(model, data, batchsize, logj=None, start=0, end=None, param=None, pool=None, nocuda=False):

    if logj is None:
        logj = torch.zeros(len(data), device=data.device)

    if pool is None:
        _transform_batch_model(model, data, logj, 0, batchsize, start=start, end=end, param=param, nocuda=nocuda)
    else:
        if torch.cuda.is_available() and not nocuda:
            nprocess = torch.cuda.device_count()
        else:
            nprocess = mp.cpu_count()
        param0 = [(model, data, logj, i, batchsize, len(data)*i//nprocess, len(data)*(i+1)//nprocess, start, end, param, nocuda) for i in range(nprocess)]
        pool.starmap(_transform_batch_model, param0)

    return data, logj



class SlicedTransport(nn.Module):

    #1 layer of SINF model
    def __init__(self, ndim, K=None, M=200):

        #K: number of slices per iteration. The same K in max K-SWD.
        #M: number of spline knots of rational quadratic spline

        super().__init__()
        self.ndim = ndim
        if K is None:
            self.K = ndim
        else:
            self.K = K
        self.M = M

        ATi = torch.randn(self.ndim, self.K)
        Q, R = torch.linalg.qr(ATi)
        L = torch.sign(torch.diag(R))
        A = (Q * L)

        self.A = nn.Parameter(A)
        self.transform1D = RQspline(self.K, M)


    def fit_A(self, data, sample='gaussian', weight=None, ndata_A=None, MSWD_p=2, MSWD_max_iter=200, pool=None, verbose=True):

        #fit the directions A to apply 1D transform

        if verbose:
            tstart = start_timing(self.A.device)

        if ndata_A is None or ndata_A > len(data):
            ndata_A = len(data)
        if sample != 'gaussian':
            if ndata_A > len(sample):
                ndata_A = len(sample)
            if ndata_A == len(sample):
                sample = sample.to(self.A.device)
            else:
                sample = sample[torch.randperm(len(sample), device=sample.device)[:ndata_A]].to(self.A.device)
        if ndata_A == len(data):
            data = data.to(self.A.device)
            if weight is not None:
                weight = weight.to(self.A.device)
        else:
            order = torch.randperm(len(data), device=data.device)[:ndata_A]
            data = data[order].to(self.A.device)
            if weight is not None:
                weight = weight[order].to(self.A.device)
        if weight is not None:
            weight = weight / torch.sum(weight)
            select = weight > 0
            data = data[select]
            weight = weight[select]

        A, SWD = maxKSWDdirection(data, x2=sample, weight=weight, K=self.K, maxiter=MSWD_max_iter, p=MSWD_p)
        with torch.no_grad():
            SWD, indices = torch.sort(SWD, descending=True)
            A = A[:,indices]
            self.A[:] = torch.linalg.qr(A)[0]

        if verbose:
            t = end_timing(tstart, self.A.device)
            print ('Fit A:', 'Time:', t, 'Wasserstein Distance:', SWD.tolist())
        return self


    def fit_spline(self, data, weight=None, edge_bins=0, derivclip=None, extrapolate='regression', alpha=(0.9,0.99), noise_threshold=0, MSWD_p=2, KDE=True, b_factor=1, batchsize=None, random_knots=False, maxknot=False, verbose=True):

        #fit the 1D transform \Psi

        assert extrapolate in ['endpoint', 'regression']
        assert self.M > 2 * edge_bins
        assert derivclip is None or derivclip >= 1

        with torch.no_grad():
            if verbose:
                tstart = start_timing(self.A.device)

            if noise_threshold > 0:
                SWD = SlicedWasserstein_direction(data, self.A.to(data.device), second='gaussian', weight=weight, p=MSWD_p)
                above_noise = SWD > noise_threshold
            else:
                above_noise = torch.ones(self.A.shape[1], dtype=bool, device=self.A.device)

            data0 = (data @ self.A.to(data.device)).to(self.A.device)
            if weight is not None:
                weight = weight.to(self.A.device)
                weight = weight / torch.sum(weight)
                select = weight > 0
                data0 = data0[select]
                weight = weight[select]

            #build rational quadratic spline transform
            x, y, deriv = estimate_knots_gaussian(data0, M=self.M, above_noise=above_noise, weight=weight, edge_bins=edge_bins, derivclip=derivclip, extrapolate=extrapolate, 
                                                  alpha=alpha, KDE=KDE, b_factor=b_factor, batchsize=batchsize, random_knots=random_knots, maxknot=maxknot)
            self.transform1D.set_param(x, y, deriv)

            if verbose:
                t = end_timing(tstart, self.A.device)
                try:
                    print ('Fit spline:', 'Time:', t, 'Wasserstein Distance:', SWD.tolist())
                except:
                    print ('Fit spline Time:', t)

            return above_noise.any()


    def fit_spline_inverse(self, data, sample, edge_bins=4, derivclip=1, extrapolate='regression', alpha=(0,0), noise_threshold=0, MSWD_p=2, KDE=True, b_factor_data=1, b_factor_sample=1, batchsize=None, verbose=True):

        #fit the 1D transform \Psi
        #inverse method

        assert extrapolate in ['endpoint', 'regression']
        assert self.M > 2 * edge_bins
        assert derivclip is None or derivclip >= 1

        with torch.no_grad():
            if verbose:
                tstart = start_timing(self.A.device)

            if noise_threshold > 0:
                SWD = SlicedWasserstein_direction(data, self.A.to(data.device), second=sample, p=MSWD_p)
                above_noise = SWD > noise_threshold
            else:
                above_noise = torch.ones(self.A.shape[1], dtype=bool, device=self.A.device)

            data0 = (data @ self.A.to(data.device)).to(self.A.device)
            sample0 = (sample @ self.A.to(sample.device)).to(self.A.device)

            #build rational quadratic spline transform
            x, y, deriv = estimate_knots(data0, sample0, M=self.M, above_noise=above_noise, edge_bins=edge_bins, derivclip=derivclip,
                                         extrapolate=extrapolate, alpha=alpha, KDE=KDE, b_factor_data=b_factor_data, b_factor_sample=b_factor_sample, batchsize=batchsize)
            self.transform1D.set_param(x, y, deriv)

            if verbose:
                t = end_timing(tstart, self.A.device)
                try:
                    print ('Fit spline:', 'Time:', t, 'Wasserstein Distance:', SWD.tolist())
                except:
                    print ('Fit spline Time:', t)

            return above_noise.any()


    def transform(self, data, mode='forward', d_dz=None, param=None):

        data0 = data @ self.A
        remaining = data - data0 @ self.A.T
        if mode == 'forward':
            data0, logj = self.transform1D(data0)
        elif mode == 'inverse':
            data0, logj = self.transform1D.inverse(data0)
            if d_dz is not None:
                d_dz0 = torch.einsum('ijk,jl->ilk', d_dz, self.A)
                remaining_d_dz = d_dz - torch.einsum('ijk,lj->ilk', d_dz0, self.A)
                d_dz0 /= torch.exp(logj[:,:,None])
                d_dz = remaining_d_dz + torch.einsum('ijk,lj->ilk', d_dz0, self.A)
        logj = torch.sum(logj, dim=1)
        data = remaining + data0 @ self.A.T

        if d_dz is None:
            return data, logj
        else:
            return data, logj, d_dz


    def forward(self, data, param=None):
        return self.transform(data, mode='forward', param=param)


    def inverse(self, data, d_dz=None, param=None):
        return self.transform(data, mode='inverse', d_dz=d_dz, param=param)




def Shift(data, shift):
    if shift[0] != 0:
        shiftx = shift[0]
        left = data.shape[1] - shiftx
        temp = torch.clone(data[:,:shiftx,:,:])
        data[:,:left,:,:] = torch.clone(data[:,shiftx:,:,:])
        data[:,left:,:,:] = temp
    if shift[1] != 0:
        shifty = shift[1]
        left = data.shape[2] - shifty
        temp = torch.clone(data[:,:,:shifty,:])
        data[:,:,:left,:] = torch.clone(data[:,:,shifty:,:])
        data[:,:,left:,:] = temp
    return data


def UnShift(data, shift):
    if shift[0] != 0:
        shiftx = shift[0]
        left = data.shape[1] - shiftx
        temp = torch.clone(data[:,left:,:,:])
        data[:,shiftx:,:,:] = torch.clone(data[:,:left,:,:])
        data[:,:shiftx,:,:] = temp
    if shift[1] != 0:
        shifty = shift[1]
        left = data.shape[2] - shifty
        temp = torch.clone(data[:,:,left:,:])
        data[:,:,shifty:,:] = torch.clone(data[:,:,:left,:])
        data[:,:,:shifty,:] = temp
    return data


class PatchSlicedTransport(nn.Module):

    #1 layer of patch based sliced transport

    def __init__(self, shape=[28,28,1], kernel=[4,4,1], shift=[0,0], K=None, M=200):

        assert shift[0] >= 0 and shift[0] < shape[0]
        assert shift[1] >= 0 and shift[1] < shape[1]
        assert len(shape) == 3 and len(kernel) == 3 and len(shift) == 2
        assert (kernel[0] <= shape[0]) and (kernel[1] <= shape[1])
        if shape[-1] == 1:
            assert kernel[-1] == 1
        else:
            assert (kernel[-1] == 1) or (kernel[-1] == shape[-1])

        super().__init__()
        self.register_buffer('shape', torch.tensor(shape))
        self.register_buffer('kernel', torch.tensor(kernel))
        self.register_buffer('shift', torch.tensor(shift))

        self.ndim_sub = (self.kernel[0]*self.kernel[1]*self.kernel[2]).item()

        if K is None:
            self.K = self.ndim_sub
        else:
            self.K = K
            assert K <= self.ndim_sub
        self.M = M

        self.Nkernel_H = (self.shape[0] // self.kernel[0]).item()
        self.Nkernel_W = (self.shape[1] // self.kernel[1]).item()
        self.Nkernel_C = (self.shape[2] // self.kernel[2]).item()
        self.Nkernel = self.Nkernel_H * self.Nkernel_W * self.Nkernel_C

        A = torch.zeros(self.Nkernel, self.ndim_sub, self.K)
        for i in range(self.Nkernel):
            ATi = torch.randn(self.ndim_sub, self.K)
            Q, R = torch.linalg.qr(ATi)
            L = torch.sign(torch.diag(R))
            A[i] = (Q * L)

        self.A = nn.Parameter(A)
        self.transform1D = RQspline(self.Nkernel*self.K, M)


    @staticmethod
    def _fit_A_patch(data, sample, A, SWD, dim, index, HWC, kernel, K, ndata_A, max_iter):

        if torch.cuda.is_available():
            gpu = index % torch.cuda.device_count()
            device = torch.device('cuda:%d'%gpu)
            device0 = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
            device0 = torch.device('cpu')
        H, W, C = HWC
        h = index // (W*C)
        w = (index-h*W*C) // C
        c = index - h*W*C - w*C
        if C == 1:
            dim0 = dim[h*kernel[0]:(h+1)*kernel[0], w*kernel[1]:(w+1)*kernel[1], :].reshape(-1).to(device)
        else:
            dim0 = dim[h*kernel[0]:(h+1)*kernel[0], w*kernel[1]:(w+1)*kernel[1], c].reshape(-1).to(device)
        if ndata_A == len(data):
            data0 = data[:, dim0].to(device)
        else:
            data0 = data[torch.randperm(len(data), device=data.device)[:ndata_A]][:, dim0].to(device)
        if sample == 'gaussian':
            sample0 = 'gaussian'
        elif ndata_A == len(sample):
            sample0 = sample[:, dim0].to(device)
        else:
            sample0 = sample[torch.randperm(len(sample), device=sample.device)[:ndata_A]][:, dim0].to(device)
        A0, SWD0 = maxKSWDdirection(data0, sample0, K=K, maxiter=max_iter)
        del data0, sample0, dim0
        with torch.no_grad():
            SWD0, indices = torch.sort(SWD0, descending=True)
            SWD[index] = SWD0.to(SWD.device)
            A0 = A0[:, indices]
            A[index] = torch.linalg.qr(A0)[0].to(A.device)

        del SWD0, indices, A0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def fit_A(self, data, sample='gaussian', ndata_A=None, MSWD_max_iter=200, pool=None, verbose=True):

        #fit the directions to apply 1D transform

        if verbose:
            tstart = start_timing(self.A.device)

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        dim = torch.arange(data.shape[1], device=data.device).reshape(1, *self.shape)
        dim = Shift(dim, self.shift)[0]

        if ndata_A is None:
            ndata_A = len(data)
        elif ndata_A > len(data):
            ndata_A = len(data)
        if sample != 'gaussian' and ndata_A > len(sample):
            ndata_A = len(sample)

        SWD = torch.zeros(self.Nkernel, self.K, device=device)

        HWC = (self.Nkernel_H, self.Nkernel_W, self.Nkernel_C)
        if pool is not None:
            param = [(data, sample, self.A, SWD, dim, index, HWC, self.kernel, self.K, ndata_A, MSWD_max_iter) for index in range(self.Nkernel)]
            pool.starmap(self._fit_A_patch, param)

        else:
            for index in range(self.Nkernel):
                self._fit_A_patch(data, sample, self.A, SWD, dim, index, HWC, self.kernel, self.K, ndata_A, MSWD_max_iter)

        if verbose:
            t = end_timing(tstart, self.A.device)
            print ('Fit A:', 'Time:', t, 'Wasserstein Distance:', SWD.tolist())

        return self


    def construct_A(self):

        dim = torch.arange(torch.prod(self.shape), device=self.A.device).reshape(1, *self.shape)
        dim = Shift(dim, self.shift)[0]
        Ntransform = self.Nkernel*self.K
        A = torch.zeros(torch.prod(self.shape), Ntransform, device=self.A.device)

        for h in range(self.Nkernel_H):
            for w in range(self.Nkernel_W):
                for c in range(self.Nkernel_C):
                    if self.Nkernel_C == 1:
                        dim0 = dim[h*self.kernel[0]:(h+1)*self.kernel[0], w*self.kernel[1]:(w+1)*self.kernel[1], :].reshape(-1)
                    else:
                        dim0 = dim[h*self.kernel[0]:(h+1)*self.kernel[0], w*self.kernel[1]:(w+1)*self.kernel[1], c].reshape(-1)
                    index = h*self.Nkernel_W*self.Nkernel_C + w*self.Nkernel_C + c
                    A[dim0, self.K*index:self.K*(index+1)] = self.A[index]

        return A


    def fit_spline(self, data, edge_bins=0, derivclip=None, extrapolate='regression', alpha=(0.9,0.99), noise_threshold=0, KDE=True, b_factor=1, batchsize=None, verbose=True):

        assert extrapolate in ['endpoint', 'regression']
        assert self.M > 2 * edge_bins
        assert derivclip is None or derivclip >= 1

        with torch.no_grad():
            if verbose:
                tstart = start_timing(self.A.device)

            A = self.construct_A().to(data.device)

            if noise_threshold > 0:
                SWD = SlicedWasserstein_direction(data, A, second='gaussian')
                above_noise = (SWD > noise_threshold).to(self.A.device)
            else:
                above_noise = torch.ones(A.shape[1], dtype=bool, device=self.A.device)

            data0 = (data @ A).to(self.A.device)

            #build rational quadratic spline transform
            x, y, deriv = estimate_knots_gaussian(data0, M=self.M, above_noise=above_noise, edge_bins=edge_bins, derivclip=derivclip,
                                                  extrapolate=extrapolate, alpha=alpha, KDE=KDE, b_factor=b_factor, batchsize=batchsize)
            self.transform1D.set_param(x, y, deriv)

            if verbose:
                t = end_timing(tstart, self.A.device)
                try:
                    print ('Fit spline:', 'Time:', t, 'Wasserstein Distance:', SWD.reshape(self.Nkernel, self.K).tolist())
                except:
                    print ('Fit spline Time:', t)
            return above_noise.any()


    def fit_spline_inverse(self, data, sample, edge_bins=4, derivclip=1, extrapolate='regression', alpha=(0,0), noise_threshold=0, KDE=True, b_factor_data=1, b_factor_sample=1, batchsize=None, verbose=True):

        #fit the 1D transform \Psi
        #inverse method

        assert extrapolate in ['endpoint', 'regression']
        assert self.M > 2 * edge_bins
        assert derivclip is None or derivclip >= 1

        with torch.no_grad():
            if verbose:
                tstart = start_timing(self.A.device)

            A = self.construct_A().to(data.device)

            if noise_threshold > 0:
                SWD = SlicedWasserstein_direction(data, A, second=sample, batchsize=16)
                above_noise = (SWD > noise_threshold).to(self.A.device)
            else:
                above_noise = torch.ones(A.shape[1], dtype=bool, device=self.A.device)
            data0 = (data @ A).to(self.A.device)
            sample0 = (sample @ A).to(self.A.device)

            #build rational quadratic spline transform
            x, y, deriv = estimate_knots(data0, sample0, M=self.M, above_noise=above_noise, edge_bins=edge_bins, derivclip=derivclip,
                                         extrapolate=extrapolate, alpha=alpha, KDE=KDE, b_factor_data=b_factor_data, b_factor_sample=b_factor_sample, batchsize=batchsize)
            self.transform1D.set_param(x, y, deriv)

            if verbose:
                t = end_timing(tstart, self.A.device)
                try:
                    print ('Fit spline:', 'Time:', t, 'Wasserstein Distance:', SWD.reshape(self.Nkernel, self.K).tolist())
                except:
                    print ('Fit spline Time:', t)

            return above_noise.any()


    def transform(self, data, mode='forward', d_dz=None, param=None):

        A = self.construct_A()

        data0 = data @ A
        remaining = data - data0 @ A.T
        if mode == 'forward':
            data0, logj = self.transform1D(data0)
        elif mode == 'inverse':
            data0, logj = self.transform1D.inverse(data0)
            if d_dz is not None:
                d_dz0 = torch.einsum('ijk,jl->ilk', d_dz, A)
                remaining_d_dz = d_dz - torch.einsum('ijk,lj->ilk', d_dz0, A)
                d_dz0 /= torch.exp(logj[:,:,None])
                d_dz = remaining_d_dz + torch.einsum('ijk,lj->ilk', d_dz0, A)
        logj = torch.sum(logj, dim=1)
        data = remaining + data0 @ A.T

        if d_dz is None:
            return data, logj
        else:
            return data, logj, d_dz


    def forward(self, data, param=None):
        return self.transform(data, mode='forward', param=param)


    def inverse(self, data, d_dz=None, param=None):
        return self.transform(data, mode='inverse', d_dz=d_dz, param=param)
