"""
Machine Learning utilities for spectral data processing.
Includes autoencoder models and training functions for CRISM spectral analysis.
"""

import numpy as np
import warnings
from typing import Optional, List, Tuple, Union

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import optim
    from torch.utils.data import DataLoader, random_split
    from numpy.random import randint, choice
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. ML functions will not work. Install with: pip install torch")

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class GumbelSoftmax(nn.Module):
    """
    Gumbel-Softmax activation layer for differentiable sampling.
    
    Args:
        temperature: Temperature parameter for Gumbel-Softmax
        hard: If True, use hard Gumbel-Softmax (one-hot)
    """
    def __init__(self, temperature=1.0, hard=False):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GumbelSoftmax. Install with: pip install torch")
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(self, x):
        return F.gumbel_softmax(x, tau=self.temperature, hard=self.hard)


class SpectralAutoencoder(nn.Module):
    """
    Autoencoder neural network for spectral data dimensionality reduction.
    
    Args:
        encoded_space_dim: Dimension of the encoded space
        in_channels: Number of input spectral bands
        n_layers_encoder: Number of encoder layers
        n_layers_decoder: Number of decoder layers
        out1: List of output dimensions for encoder layers
        out2: List of output dimensions for decoder layers
        act: Activation function
        drops: List of dropout values for each layer
        last: Unused parameter (kept for compatibility)
    """
    def __init__(self, encoded_space_dim, in_channels, n_layers_encoder, n_layers_decoder, 
                 out1, out2, act, drops, last):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for SpectralAutoencoder. Install with: pip install torch")
        super(SpectralAutoencoder, self).__init__()

        self.encoded_space_dim = encoded_space_dim
        self.in_channels = in_channels
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.out1 = out1
        self.out2 = out2
        self.drops = drops
    
        self.model = []
        if self.n_layers_encoder == 1:
            self.model.append(nn.Linear(self.in_channels, self.out1[0]))
            self.model.append(nn.BatchNorm1d(self.out1[0]))
            self.model.append(act)
            self.model.append(nn.Dropout(self.drops[0]))
            self.model.append(nn.Linear(self.out1[0], encoded_space_dim))
            
        elif self.n_layers_encoder == 0:
            self.model.append(nn.Linear(self.in_channels, encoded_space_dim))
            self.model.append(act)
        else:
            for i in range(self.n_layers_encoder):
                if i == self.n_layers_encoder-1:
                    self.model.append(nn.Linear(self.out1[i], self.encoded_space_dim))
                elif i == 0:
                    self.model.append(nn.Linear(self.in_channels, self.out1[i]))
                    self.model.append(nn.BatchNorm1d(self.out1[i]))
                    self.model.append(nn.Dropout(self.drops[i]))
                    self.model.append(act)
                    self.model.append(nn.Linear(self.out1[i], self.out1[i+1]))
                    self.model.append(nn.Dropout(self.drops[i+1]))
                    self.model.append(act)
                else:   
                    self.model.append(nn.Linear(self.out1[i], self.out1[i+1]))
                    self.model.append(nn.BatchNorm1d(self.out1[i+1]))
                    self.model.append(nn.Dropout(self.drops[i+1]))
                    self.model.append(act)
                    
        # Add GumbelSoftmax to nn.Sequential
        self.encoder = nn.Sequential(*self.model, GumbelSoftmax(temperature=1.0, hard=True))
        
        self.model2 = []
        
        if self.n_layers_decoder == 0:
            self.model2.append(nn.Linear(self.encoded_space_dim, self.in_channels))
        elif self.n_layers_decoder == 1:
            self.model2.append(nn.Linear(self.encoded_space_dim, self.out2[0]))
            self.model2.append(act)
            self.model2.append(nn.Linear(self.out2[0], self.in_channels))
        else:
            self.model2.append(nn.Linear(self.encoded_space_dim, self.out2[0]))
            self.model2.append(act)
            for i in range(self.n_layers_decoder-1):
                self.model2.append(nn.Linear(self.out2[i], self.out2[i+1]))
                self.model2.append(act)
            self.model2.append(nn.Linear(self.out2[self.n_layers_decoder-1], self.in_channels))
            self.model2.append(nn.Tanh())
        self.decoder = nn.Sequential(*self.model2)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Alias for backward compatibility
Net = SpectralAutoencoder


def weight_init(model: nn.Module, init_method: str):
    """
    Initialize weights of a neural network model.
    
    Args:
        model: PyTorch model to initialize
        init_method: Initialization method ('kaiming_normal', 'kaiming_uniform', 
                    'xavier_normal', 'xavier_uniform', 'uniform', 'normal', 
                    'ones', 'eye', 'orthogonal')
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for weight_init. Install with: pip install torch")
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if init_method == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight)
            elif init_method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight)
            elif init_method == 'xavier_normal':
                nn.init.xavier_normal_(module.weight)
            elif init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif init_method == 'uniform':
                nn.init.uniform_(module.weight)
            elif init_method == 'normal':
                nn.init.normal_(module.weight)
            elif init_method == 'ones':
                nn.init.ones_(module.weight)
            elif init_method == 'eye':
                nn.init.eye_(module.weight)
            elif init_method == 'orthogonal':
                nn.init.orthogonal_(module.weight)
            else:
                raise ValueError(f"Invalid initialization method: {init_method}")
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


# Available initialization methods
INITS = ['kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform', 
         'uniform', 'normal', 'ones', 'eye', 'orthogonal']


def random_search_autoencoder(dataset, in_channels, criterion, encoded_space_dim,
                             n_encmax, n_decmax, MINenc, MAXenc, MINdec, MAXdec,
                             activations, initializations=INITS,
                             fix_enc=False, fix_dec=False,
                             try_epochs=10, N_try=10, Seed=None,
                             printer='off', bs=100, num_pieces=5, val_split=0.2):
    """
    Random search for optimal autoencoder hyperparameters.
    
    Args:
        dataset: PyTorch dataset
        in_channels: Number of input spectral bands
        criterion: Loss function
        encoded_space_dim: List of possible encoded space dimensions
        n_encmax: Maximum number of encoder layers
        n_decmax: Maximum number of decoder layers
        MINenc: Minimum encoder layer size
        MAXenc: Maximum encoder layer size
        MINdec: Minimum decoder layer size
        MAXdec: Maximum decoder layer size
        activations: List of activation functions to try
        initializations: List of weight initialization methods
        fix_enc: If True, fix encoder layers to n_encmax
        fix_dec: If True, fix decoder layers to n_decmax
        try_epochs: Number of training epochs per trial
        N_try: Number of random trials
        Seed: Random seed
        printer: Print mode ('off' or 'on')
        bs: Batch size
        num_pieces: Unused (kept for compatibility)
        val_split: Validation split ratio
        
    Returns:
        Tuple of best hyperparameters: (LR, Lenc, Ldec, W, enc, ACT, DROPS, inits, L1)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for random_search_autoencoder. Install with: pip install torch")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if Seed is None:
        Seed = torch.seed()
    
    print(f"Using device: {device}")
    
    # Initialize arrays
    rate, W, L1 = [], [], []
    train_l = []
    val_l = []
    Lenc = np.zeros((N_try, n_encmax))
    Ldec = np.zeros((N_try, n_decmax))
    enc = []
    ACT = []
    DROPS = []
    inits = []

    total_len = len(dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len
    
    train_set, val_set = random_split(dataset, [train_len, val_len], 
                                     generator=torch.Generator().manual_seed(Seed))
    TL = DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True)
    VL = DataLoader(val_set, batch_size=bs, shuffle=False, pin_memory=True)
    
    for i in range(N_try):
        losses = []
        validations = []
        
        ENCSPDIM = choice(encoded_space_dim)
        
        print(f'Try {i+1}/{N_try}')
        
        if fix_dec == False:
            n_dec = randint(0, n_decmax+1)
        else:
            n_dec = n_decmax
            
        if fix_enc == False:
            n_enc = randint(0, n_encmax+1)
        else:
            n_enc = n_encmax

        LR = choice(np.array([1, 5])) * choice(np.array([0.00001, 0.0001, 0.001, 0.01, 0.1]))
        
        outLenc = np.sort(choice(np.arange(MINenc*5, MAXenc*5+5, 5, dtype=int), n_enc))[::-1]
        outLdec = np.sort(choice(np.arange(MINdec*5, MAXdec*5+5, 5, dtype=int), n_dec))
        
        dropouts = choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], n_enc+1)
        l1_reg = choice([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
        initials = choice(np.array(initializations))
        inits.append(initials)
        
        act = choice(activations)
        ACT.append(act)
        DROPS.append(dropouts)

        # Generate model
        model = SpectralAutoencoder(ENCSPDIM, in_channels, n_enc, n_dec, 
                                    outLenc, outLdec, act, dropouts, True).to(device)
        weight = choice([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
        weight_init(model, init_method=initials)
        optimizer = choice([optim.Adam(model.parameters(), lr=LR, weight_decay=weight)])

        # Training loop
        for epoch in range(try_epochs):
            model.train()
            LOSS = 0
            for x in TL:
                x = x.float().to(device)
                y = model(x)
                loss = criterion(y, x)
                l1_lambda = l1_reg
                l1 = sum(param.abs().sum() for param in model.parameters())
                loss += l1_lambda * l1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                LOSS += loss.item()
            LOSS /= len(TL)
            losses.append(LOSS)

        # Validation loop
        model.eval()
        VAL_LOSS = 0
        with torch.no_grad():
            for x in VL:
                x = x.float().to(device)
                y = model(x)
                val_loss = criterion(y, x)
                VAL_LOSS += val_loss.item()
        VAL_LOSS /= len(VL)
        val_l.append(VAL_LOSS)

        # Update arrays
        rate.append(LR)
        W.append(weight)
        enc.append(ENCSPDIM)
        L1.append(l1_reg)

        for j in range(n_enc):
            Lenc[i, j] = outLenc[j]
        for j in range(n_dec):
            Ldec[i, j] = outLdec[j]
        
        train_l.append(np.array(losses)[-1])
        print(f'Final loss value = {train_l[i]:.6f}')

    J = np.argmin(np.asarray(val_l))
    
    # Print best results
    hyperparanames = ['LR', 'outC', 'outL', 'n_conv_layers', 'n_lin_layers', 
                     'encoded_space_dim', 'weight_initialization']
    print('\033[4;34;43m' + 'Best results is ' + '\033[0m', J, 
          '\033[4;34;43m' + '. With values: ' + '\033[0m',
          "\n", hyperparanames, "\n", rate[J], Lenc[J], Ldec[J], W[J], 
          enc[J], ACT[J], DROPS[J], inits[J], L1[J])
    
    return rate[J], Lenc[J], Ldec[J], W[J], enc[J], ACT[J], DROPS[J], inits[J], L1[J]


# Alias for backward compatibility
RandomSearch_autoencoder = random_search_autoencoder


def train_autoencoder(model, criterion, train_ds, validation_ds, num_epochs, 
                     patience, weight_decay, bs, device, LR, printer=False):
    """
    Train an autoencoder model.
    
    Args:
        model: PyTorch model
        criterion: Loss function
        train_ds: Training dataset
        validation_ds: Validation dataset
        num_epochs: Number of training epochs
        patience: Patience for learning rate scheduler
        weight_decay: Weight decay for optimizer
        bs: Batch size (currently fixed to 1024)
        device: Device to use ('cuda' or 'cpu')
        LR: Learning rate
        printer: If True, print training progress
        
    Returns:
        Tuple of (train_losses, validation_losses)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for train_autoencoder. Install with: pip install torch")
    
    model.to(device)
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=False, pin_memory=True)
    validation_loader = DataLoader(validation_ds, batch_size=1024, shuffle=False, pin_memory=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           patience=patience, factor=0.5)
    train_losses = []
    validation_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x in train_loader:
            x = x.float().to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        with torch.no_grad():
            model.eval()
            validation_loss = 0
            for x in validation_loader:
                x = x.float().to(device, non_blocking=True)
                outputs = model(x)
                loss = criterion(outputs, x)
                validation_loss += loss.item()
            validation_loss /= len(validation_loader)
            validation_losses.append(validation_loss)
            scheduler.step(validation_loss)
            
            if printer:
                print(f'Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss:.4f} | Validation Loss: {validation_loss:.4f}')
            
    print('Finished Training')
    return train_losses, validation_losses


# Alias for backward compatibility
train_cnn = train_autoencoder


def plot_encoded_space(spectra, model, DIM, plot=True):
    """
    Visualize the encoded space of an autoencoder.
    
    Args:
        spectra: Input spectra array
        model: Trained autoencoder model
        DIM: Number of encoded dimensions to plot
        plot: If True, display the plot
        
    Returns:
        Encoded data array
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for plot_encoded_space. Install with: pip install torch")
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for plot_encoded_space. Install with: pip install matplotlib")
    
    # Extract low-dimensional representations
    encoded_data = model.encoder(torch.tensor(spectra).float()).detach().numpy()
    
    if plot:
        fig, ax = plt.subplots(DIM, DIM)
        for i in range(DIM):
            for j in range(DIM):
                if i != j and i < j:
                    ax[i, j].plot(encoded_data[:, i], encoded_data[:, j], 'k.')
                    ax[i, j].set_xlabel(f'Encoded Dimension {i+1}', fontsize=5)
                    ax[i, j].set_ylabel(f'Encoded Dimension {j+1}', fontsize=5)
                else:
                    ax[i, j].axis('off')
        plt.show()
    return encoded_data


# Alias for backward compatibility
plot_n_encoded = plot_encoded_space

