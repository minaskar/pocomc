import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import copy 
import time


def fit(model,
        data,
        context=None,
        validation_data=None,
        validation_context=None,
        validation_split=0.0,
        epochs=20,
        batch_size=100,
        patience=np.inf,
        monitor='val_loss',
        shuffle=True,
        lr=1e-3,
        weight_decay=1e-5,
        clip_grad_norm=1.0,
        l1=None,
        l2=None,
        device='cpu',
        verbose=2):
    """
        Method to fit the normalising flow.
        
    """
        

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
        
    if (validation_data is not None) and (not isinstance(validation_data, torch.Tensor)):
        validation_data = torch.tensor(validation_data, dtype=torch.float32)

    if context is not None:
        use_context = True
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context, dtype=torch.float32)
    else:
        use_context = False

    if (validation_context is not None) and (not isinstance(validation_context, torch.Tensor)):
        validation_context = torch.tensor(validation_context, dtype=torch.float32)
            

    if validation_data is not None:
        
        if use_context:
            train_dl = DataLoader(TensorDataset(data, context), batch_size, shuffle)
            val_dl = DataLoader(TensorDataset(validation_data, validation_context), batch_size, shuffle)
        else:
            train_dl = DataLoader(TensorDataset(data), batch_size, shuffle)
            val_dl = DataLoader(TensorDataset(validation_data), batch_size, shuffle)

        validation = True
    else:
        if validation_split > 0.0 and validation_split < 1.0:
            validation = True
            split = int(data.size()[0] * (1. - validation_split))
            if use_context:
                data, validation_data = data[:split], data[split:]
                context, validation_context = context[:split], context[split:]
                train_dl = DataLoader(TensorDataset(data, context), batch_size, shuffle)
                val_dl = DataLoader(TensorDataset(validation_data, validation_context), batch_size, shuffle)
            else:
                data, validation_data = data[:split], data[split:]
                train_dl = DataLoader(TensorDataset(data), batch_size, shuffle)
                val_dl = DataLoader(TensorDataset(validation_data), batch_size, shuffle)
        else:
            validation = False
            if use_context:
                train_dl = DataLoader(TensorDataset(data, context), batch_size, shuffle)
            else:
                train_dl = DataLoader(TensorDataset(data), batch_size, shuffle)

    history = {} # Collects per-epoch loss
    history['loss'] = []
    history['val_loss'] = []

    if not validation:
        monitor = 'loss'
    best_epoch = 0
    best_loss = np.inf
    best_model = copy.deepcopy(model.state_dict())

    start_time_sec = time.time()

    for epoch in range(epochs):

        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        train_loss = 0.0

        for batch in train_dl:

            optimizer.zero_grad()

            if use_context:
                x = batch[0].to(device)
                y = batch[1].to(device)
                loss = -model.log_prob(x, y).sum()
            else:
                x = batch[0].to(device)
                loss = -model.log_prob(x).sum()

            if l1 is not None:
                loss -= model.log_prior(scale=l1, type='Laplace')

            if l2 is not None:
                loss -= model.log_prior(scale=l2, type='Gaussian')


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

            #train_loss += loss.data.item() * x.size(0)
            train_loss += loss.data.item()
            
        train_loss  = train_loss / len(train_dl.dataset)

        history['loss'].append(train_loss)


        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        if validation:
            val_loss = 0.0

            for batch in val_dl:

                if use_context:
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    loss = -model.log_prob(x, y).sum()
                else:
                    x = batch[0].to(device)
                    loss = -model.log_prob(x).sum()

                if l1 is not None:
                    loss -= model.log_prior(scale=l1, type='Laplace')

                if l2 is not None:
                    loss -= model.log_prior(scale=l2, type='Gaussian')

                #val_loss += loss.data.item() * x.size(0)
                val_loss += loss.data.item()

            val_loss = val_loss / len(val_dl.dataset)

            history['val_loss'].append(val_loss)

        

        if verbose > 1:
            try:
                print('Epoch %3d/%3d, train loss: %5.2f, val loss: %5.2f' % \
                    (epoch+1, epochs, train_loss, val_loss))
            except:
                print('Epoch %3d/%3d, train loss: %5.2f' % \
                    (epoch+1, epochs, train_loss))


        # Monitor loss
        if history[monitor][-1] < best_loss:
            best_loss = history[monitor][-1]
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())


        if epoch - best_epoch >= patience:
            model.load_state_dict(best_model)
            if verbose > 0:
                print('Finished early after %3d epochs' % (best_epoch))
                print('Best loss achieved %5.2f' % (best_loss))
            break
        
    
    # END OF TRAINING LOOP

    if verbose > 0:
        end_time_sec       = time.time()
        total_time_sec     = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / epochs
        print()
        print('Time total:     %5.2f sec' % (total_time_sec))
        print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history


def FlowTrainer(model, data, train_config=None):

    default_train_config = dict(validation_split=0.2,
                                epochs=1000,
                                batch_size=data.shape[0],
                                patience=30,
                                monitor='val_loss',
                                shuffle=True,
                                lr=[1e-2, 1e-3, 1e-4, 1e-5],
                                weight_decay=1e-8,
                                clip_grad_norm=1.0,
                                l1=0.2,
                                l2=None,
                                device='cpu',
                                verbose=0,
                                )

    if train_config is None:
        train_config = default_train_config
    
    for lr in train_config.get('lr', default_train_config['lr']):
        history = fit(model,
                      data,
                      context=None,
                      validation_data=None,
                      validation_context=None,
                      validation_split=train_config.get('validation_split', default_train_config['validation_split']),
                      epochs=train_config.get('epochs', default_train_config['epochs']),
                      batch_size=train_config.get('batch_size', default_train_config['batch_size']),
                      patience=train_config.get('patience', default_train_config['patience']),
                      monitor=train_config.get('monitor', default_train_config['monitor']),
                      shuffle=train_config.get('shuffle', default_train_config['shuffle']),
                      lr=lr,
                      weight_decay=train_config.get('weight_decay', default_train_config['weight_decay']),
                      clip_grad_norm=train_config.get('clip_grad_norm', default_train_config['clip_grad_norm']),
                      l1=train_config.get('l1', default_train_config['l1']),
                      l2=train_config.get('l2', default_train_config['l2']),
                      device=train_config.get('device', default_train_config['device']),
                      verbose=train_config.get('verbose', default_train_config['verbose']))

    return history
