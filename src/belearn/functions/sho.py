import torch

def SHO_nn(params, wvec_freq, device='cpu'):
    """
    Computes the response of a Simple Harmonic Oscillator (SHO) model 
    for a given set of parameters and frequencies.

    Args:
        params (torch.Tensor): A tensor of shape (N, 4), where N is the 
                               number of oscillators. Each row contains
                               the parameters [Amp, w_0, Q, phi] for one oscillator.
                               - Amp: Amplitude of the oscillator (real)
                               - w_0: Resonant frequency (real)
                               - Q: Quality factor (real)
                               - phi: Phase offset (real)
        wvec_freq (array-like): A vector of frequencies at which the response 
                                is to be computed.
        device (str, optional): The device to perform the computation on. 
                                Defaults to 'cpu'.

    Returns:
        torch.Tensor: A tensor of complex values representing the response 
                      of the SHO model at the given frequencies.
    """

    # Extract parameters and ensure they are of complex type
    Amp = params[:, 0].type(torch.complex128)
    w_0 = params[:, 1].type(torch.complex128)
    Q = params[:, 2].type(torch.complex128)
    phi = params[:, 3].type(torch.complex128)

    # Convert frequency vector to a torch tensor
    wvec_freq = torch.tensor(wvec_freq)

    # Unsqueeze parameters to match the dimensions required for broadcasting
    Amp = torch.unsqueeze(Amp, 1)
    w_0 = torch.unsqueeze(w_0, 1)
    phi = torch.unsqueeze(phi, 1)
    Q = torch.unsqueeze(Q, 1)

    # Move the frequency vector to the specified device (CPU or GPU)
    wvec_freq = wvec_freq.to(device)

    # Numerator of the SHO response function
    numer = Amp * torch.exp(1.j * phi) * torch.square(w_0)

    # Denominator components
    den_1 = torch.square(wvec_freq)
    den_2 = 1.j * wvec_freq * w_0 / Q
    den_3 = torch.square(w_0)

    # Full denominator of the SHO response function
    den = den_1 - den_2 - den_3

    # Compute the response function
    func = numer / den

    return func
