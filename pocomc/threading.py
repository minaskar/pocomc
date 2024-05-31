import torch

def configure_threads(pytorch_threads=None):
    """Configure the number of threads available.

    This is necessary when using PyTorch on the CPU as by default it will use
    all available threads.

    Notes
    -----
    Uses ``torch.set_num_threads``. If pytorch threads is None but other
    arguments are specified then the value is inferred from them.

    Parameters
    ----------
    pytorch_threads: int, optional
        Maximum number of threads for PyTorch on CPU. If None, pytorch will
        use all available threads.
    """
    if pytorch_threads:
        torch.set_num_threads(pytorch_threads)