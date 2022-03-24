from .SINF import *

def GIS(data_train, data_validate=None, iteration=None, weight_train=None, weight_validate=None, K=None, M=None, KDE=True, b_factor=1, alpha=None, bounds=None, random_knots=False, maxknot=False,
        edge_bins=None, ndata_A=None, MSWD_max_iter=None, NBfirstlayer=True, Whiten=False, Whiten_with_weights=False, Whiten_reg='OAS', fix_Niteration=True, batchsize=None, nocuda=False, 
        patch=False, shape=[28,28,1], model=None, verbose=True):
    
    '''
    data_train: (ndata_train, ndim).
    data_validate: (ndata_validate, ndim), optional. If provided, its logp will be used to determine the number of iterations.
    iteration: integer, optional. The maximum number of GIS iterations. Required if data_validate is not provided.
    weight_train: (ndata_train, ), optional. The weights of data_train.
    weight_validate: (ndata_train, ), optional. The weights of data_validate.
    K: integer, optional. The number of slices for each iteration. See max K-SWD in the SINF paper. 1 <= K <= ndim.
    M: integer, optional. The number of spline knots for rational quadratic splines.
    KDE: bool. Whether to use KDE for estimating 1D PDF. Recommended True.
    b_factor: positive float number, optional. The multiplicative factor for KDE kernel width.
    alpha: two non-negative float number in the format of (alpha1, alpha2), optional. Regularization parameter. See Equation 13 of SINF paper. alpha1 for interpolation, alpha2 for extrapolation slope. 0 <= alpha1,2 < 1. If not given, very heavy regularization will be used, which could result in slow training and a large number of iterations.
    bounds: sequence, optional. In the format of [[x1_min, x1_max], [x2_min, x2_max], ..., [xd_min, xd_max]]. Represent infinity and negative infinity with None.
    edge_bins: non-negative integer, optional. The number of spline knots at the boundary.
    ndata_A: positive integer, optional. The number of training data used for fitting A (slice axes).
    MSWD_max_iter: positive integer, optional. The maximum number of iterations for optimizing A (slice axes). See Algorithm 1 of SINF paper. Called L_iter in the paper.
    NBfirstlayer: bool, optional. Whether to use Naive Bayes (no rotation) at the first layer.
    Whiten: bool, optional. Whether to whiten the data before applying GIS.
    Whiten_reg: can either be 'LW'(LedoitWolf), 'OAS', 'NERCOME', 'validation', or a float number between 0 and 1. Regularization method/parameter for estimating the sample covariance matrix. LW and OAS can only be used for unweighted data. 
    NERCOME: bool, optional. Whether to use NERCOME algorithm to estimate covariance matrix in whiten layer.
    batchsize: positive integer, optional. The batch size for transforming the data. Does not change the performance. Only saves the memory. Useful when the data is too large and can't fit in the memory.
    nocuda: bool, optional. Whether to use gpu.
    patch: bool, optional. Whether to use patch-based modeling. Only useful for image datasets.
    shape: sequence, optional. The shape of the image datasets, if patch is enabled.
    model: GIS model, optional. Trained GIS model. If provided, new iterations will be added in the model.
    verbose: bool, optional. Whether to print training information.
    '''

    assert data_validate is not None or iteration is not None
 
    #hyperparameters
    ndim = data_train.shape[1]
    if weight_train is None:
        ndata = len(data_train)
    else:
        select = weight_train > 1e-6
        if not select.all():
            weight_train = weight_train[select]
            data_train = data_train[select]
        ndata = (torch.sum(weight_train)**2 / torch.sum(weight_train**2)).item()
    if M is None:
        M = max(min(200, int(ndata**0.5)), 100)
    if alpha is None:
        alpha = (1-0.02*math.log10(ndata), 1-0.001*math.log10(ndata))
    if bounds is not None:
        assert len(bounds) == ndim
        for i in range(ndim):
            assert len(bounds[i]) == 2
    if edge_bins is None:
        edge_bins = max(int(math.log10(ndata))-1, 0)
    if batchsize is None:
        batchsize = len(data_train)
    if not patch:
        if K is None:
            if ndim <= 8 or ndata / float(ndim) < 20:
                K = ndim
            else:
                K = 8
        if ndata_A is None:
            ndata_A = min(len(data_train), int(math.log10(ndim)*1e5))
        if MSWD_max_iter is None:
            MSWD_max_iter = min(round(ndata) // ndim, 200)
    else:
        assert shape[0] > 4 and shape[1] > 4
        K0 = K
        ndata_A0 = ndata_A
        MSWD_max_iter0 = MSWD_max_iter

    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() and not nocuda else "cpu")
    data_train = data_train.to(device)
    if weight_train is not None:
        weight_train = weight_train.to(device)
    if data_validate is not None:
        data_validate = data_validate.to(device)
        if weight_validate is not None:
            weight_validate = weight_validate.to(device)

    #define the model
    if model is None:
        model = SINF(ndim=ndim).requires_grad_(False).to(device)
        logj_train = torch.zeros(len(data_train), device=device)
        if data_validate is not None:
            logj_validate = torch.zeros(len(data_validate), device=device)
            best_logp_validate = -1e10
            best_Nlayer = 0
            wait = 0
            maxwait = 5 
    else:
        t = time.time()
        data_train, logj_train = transform_batch_model(model, data_train, batchsize, logj=None, start=0, end=None, nocuda=nocuda)
        if weight_train is None:
            logp_train = (torch.mean(logj_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_train**2,  dim=1)/2)).item()
        else:
            logp_train = (torch.sum(logj_train*weight_train)/torch.sum(weight_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_train*torch.sum(data_train**2,  dim=1)/2)/torch.sum(weight_train)).item()
    
        if data_validate is not None:
            data_validate, logj_validate = transform_batch_model(model, data_validate, batchsize, logj=None, start=0, end=None, nocuda=nocuda)
            if weight_validate is None:
                logp_validate = (torch.mean(logj_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_validate**2,  dim=1)/2)).item()
            else:
                logp_validate = (torch.sum(logj_validate*weight_validate)/torch.sum(weight_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_validate*torch.sum(data_validate**2,  dim=1)/2)/torch.sum(weight_validate)).item()
            best_logp_validate = logp_validate
            best_Nlayer = len(model.layer)
            wait = 0
            maxwait = 5
            print ('Initial logp:', logp_train, logp_validate, 'time:', time.time()-t, 'iteration:', len(model.layer))
        else:
            print ('Initial logp:', logp_train, 'time:', time.time()-t, 'iteration:', len(model.layer))

    #boundary
    if bounds is not None:
        layer = boundary(bounds=bounds, lambd=0, beta=1).to(device)
        data_train, logj_train = layer(data_train)
        if weight_train is None:
            logp_train = (torch.mean(logj_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_train**2,  dim=1)/2)).item()
        else:
            logp_train = (torch.sum(logj_train*weight_train)/torch.sum(weight_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_train*torch.sum(data_train**2,  dim=1)/2)/torch.sum(weight_train)).item()

        if data_validate is not None:
            data_validate, logj_validate = layer(data_validate)
            if weight_validate is None:
                logp_validate = (torch.mean(logj_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_validate**2,  dim=1)/2)).item()
            else:
                logp_validate = (torch.sum(logj_validate*weight_validate)/torch.sum(weight_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_validate*torch.sum(data_validate**2,  dim=1)/2)/torch.sum(weight_validate)).item()
            best_logp_validate = logp_validate
            best_Nlayer = 1

        model.add_layer(layer)
        if verbose:
            if data_validate is not None:
                print('After boundary transform logp:', logp_train, logp_validate)
            else:
                print('After boundary transform logp:', logp_train)
    
    #Naive Bayes transformation
    if NBfirstlayer:
        layer = SlicedTransport(ndim=ndim, K=ndim, M=M).requires_grad_(False).to(device)
        layer.A[:] = torch.eye(ndim).to(device)
        layer.fit_spline(data=data_train, weight=weight_train, edge_bins=edge_bins, alpha=alpha, KDE=KDE, b_factor=b_factor, batchsize=batchsize, random_knots=random_knots, maxknot=maxknot, verbose=verbose)

        #update the data
        data_train, logj_train = transform_batch_layer(layer, data_train, batchsize, logj=logj_train, direction='forward', nocuda=nocuda)
        if weight_train is None:
            logp_train = (torch.mean(logj_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_train**2,  dim=1)/2)).item()
        else:
            logp_train = (torch.sum(logj_train*weight_train)/torch.sum(weight_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_train*torch.sum(data_train**2,  dim=1)/2)/torch.sum(weight_train)).item()

        model.add_layer(layer)

        if data_validate is not None:
            data_validate, logj_validate = transform_batch_layer(layer, data_validate, batchsize, logj=logj_validate, direction='forward', nocuda=nocuda)
            if weight_validate is None:
                logp_validate = (torch.mean(logj_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_validate**2,  dim=1)/2)).item()
            else:
                logp_validate = (torch.sum(logj_validate*weight_validate)/torch.sum(weight_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_validate*torch.sum(data_validate**2,  dim=1)/2)/torch.sum(weight_validate)).item()
            best_logp_validate = logp_validate
            best_Nlayer = len(model.layer)
        if verbose:
            if data_validate is not None: 
                print ('After Naive Bayes layer logp:', logp_train, logp_validate)
            else:
                print ('After Naive Bayes layer logp:', logp_train)

    #whiten
    if Whiten:
        layer = whiten(ndim_data=ndim, scale=True, ndim_latent=ndim).requires_grad_(False).to(device)
        if Whiten_with_weights:
            layer.fit(data_train, weight_train, Whiten_reg, data_validate)
        else:
            layer.fit(data_train, regularization=Whiten_reg, data_validate=data_validate)

        data_train, logj_train0 = layer(data_train)
        logj_train += logj_train0
        if weight_train is None:
            logp_train = (torch.mean(logj_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_train**2,  dim=1)/2)).item()
        else:
            logp_train = (torch.sum(logj_train*weight_train)/torch.sum(weight_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_train*torch.sum(data_train**2,  dim=1)/2)/torch.sum(weight_train)).item()

        model.add_layer(layer)

        if data_validate is not None:
            data_validate, logj_validate0 = layer(data_validate)
            logj_validate += logj_validate0
            if weight_validate is None:
                logp_validate = (torch.mean(logj_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_validate**2,  dim=1)/2)).item()
            else:
                logp_validate = (torch.sum(logj_validate*weight_validate)/torch.sum(weight_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_validate*torch.sum(data_validate**2,  dim=1)/2)/torch.sum(weight_validate)).item()
            best_logp_validate = logp_validate
            best_Nlayer = len(model.layer)

        if verbose:
            if data_validate is not None:
                print('After whiten logp:', logp_train, logp_validate)
            else:
                print('After whiten logp:', logp_train)


    #GIS iterations
    while True:
        t = time.time()
        if patch:
            #patch layers
            if len(model.layer) % 2 == 0:
                kernel = [4, 4, shape[-1]]
                shift = torch.randint(4, (2,)).tolist()
            else:
                kernel = [2, 2, shape[-1]]
                shift = torch.randint(2, (2,)).tolist()
            #hyperparameter
            ndim = np.prod(kernel)
            if K0 is None:
                if ndim <= 8 or len(data_train) / float(ndim) < 20:
                    K = ndim
                else:
                    K = 8
            elif K0 > ndim:
                K = ndim
            else:
                K = K0
            if ndata_A0 is None:
                ndata_A = min(len(data_train), int(math.log10(ndim)*1e5))
            if MSWD_max_iter0 is None:
                MSWD_max_iter = min(len(data_train) // ndim, 200)
            
            layer = PatchSlicedTransport(shape=shape, kernel_size=kernel, shift=shift, K=K, M=M).requires_grad_(False).to(device)
        else:
            #regular GIS layer
            layer = SlicedTransport(ndim=ndim, K=K, M=M).requires_grad_(False).to(device)
        
        #fit the layer
        if ndim > 1:
            layer.fit_A(data=data_train, weight=weight_train, ndata_A=ndata_A, MSWD_max_iter=MSWD_max_iter, verbose=verbose)

        layer.fit_spline(data=data_train, weight=weight_train, edge_bins=edge_bins, alpha=alpha, KDE=KDE, b_factor=b_factor, batchsize=batchsize, random_knots=random_knots, maxknot=maxknot, verbose=verbose)

        #update the data
        data_train, logj_train = transform_batch_layer(layer, data_train, batchsize, logj=logj_train, direction='forward', nocuda=nocuda)
        if weight_train is None:
            logp_train = (torch.mean(logj_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_train**2,  dim=1)/2)).item()
        else:
            logp_train = (torch.sum(logj_train*weight_train)/torch.sum(weight_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_train*torch.sum(data_train**2,  dim=1)/2)/torch.sum(weight_train)).item()

        model.add_layer(layer)

        if data_validate is not None:
            data_validate, logj_validate = transform_batch_layer(layer, data_validate, batchsize, logj=logj_validate, direction='forward', nocuda=nocuda)
            if weight_validate is None:
                logp_validate = (torch.mean(logj_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_validate**2,  dim=1)/2)).item()
            else:
                logp_validate = (torch.sum(logj_validate*weight_validate)/torch.sum(weight_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.sum(weight_validate*torch.sum(data_validate**2,  dim=1)/2)/torch.sum(weight_validate)).item()
            if logp_validate > best_logp_validate:
                best_logp_validate = logp_validate
                best_Nlayer = len(model.layer)
                wait = 0
            else:
                wait += 1
            if wait >= maxwait:
                model.layer = model.layer[:best_Nlayer]
                break

        if verbose:
            if data_validate is not None: 
                print ('logp:', logp_train, logp_validate, 'time:', time.time()-t, 'iteration:', len(model.layer), 'best:', best_Nlayer)
            else:
                print ('logp:', logp_train, 'time:', time.time()-t, 'iteration:', len(model.layer))

        if iteration is not None and len(model.layer) >= iteration:
            if data_validate is not None:
                model.layer = model.layer[:best_Nlayer]
            break

    if fix_Niteration and iteration is not None:
        while len(model.layer) < iteration:
            layer = SlicedTransport(ndim=ndim, K=K, M=M).requires_grad_(False).to(device)
            model.add_layer(layer)

    return model
