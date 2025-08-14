__all__ = ['MFCCEncoder','MFCCEncoderConfig','SpectrogramEncoder', 'SpectrogramEncoderConfig']

from ..template_modules import EncoderBase, EncoderBaseConfig
import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional
import torch.nn as nn
import torch

import torchaudio

def compute_mfcc_features_and_derivatives(x: torch.Tensor, samp_rate: int, frame_length_samples:int, frame_shift_samples:int, derivatives:bool=True):
    ''' 
    Compute MFCC features and their first and second derivatives from a signal.
    Batch-parallelized implementation for multi-channel input.
    
    Parameters:
    -----------
    x : torch.Tensor
        Input audio signal with shape (bs, channels, samples) or (bs, samples)
    samp_rate : int
        Sampling rate of the audio in Hz
        
    Returns:
    --------
    torch.Tensor
        MFCC features and derivatives with shape (batch_size, channels, features, time)
    '''
    with torch.no_grad():
        # Save original shape information
        orig_shape = x.shape
        
        # Handle various input shapes
        if x.dim() == 2:
            # Could be (batch, samples) or (channels, samples)
            # We'll treat it as (batch, samples) and add channel dim
            x = x.unsqueeze(1)  # (batch, 1, samples)
        
        batch_size, n_channels, samples = x.shape
        
        # Reshape to process all batch items and channels at once
        x_flat = x.reshape(-1, samples)  # (batch*channels, samples)
        
        # Use original frame length and shift parameters
        frame_length_ms = (x.size(-1) if frame_length_samples==0 else frame_length_samples) / samp_rate * 1000
        frame_shift_ms = 10 if frame_length_samples==0 else (frame_length_samples) / samp_rate * 1000
        
        # Process all channels in parallel
        mfccs_all = torchaudio.compliance.kaldi.mfcc(
            waveform=x_flat,
            sample_frequency=samp_rate,
            use_energy=False,
            frame_length=frame_length_ms,
            frame_shift=frame_shift_ms,
        )  # (batch*channels, time, freq)
        
        # Handle possible empty result
        if mfccs_all.numel() == 0:
            return torch.zeros(batch_size, n_channels, 1, 39 if derivatives else 13, device=x.device)
        
        # Transpose and compute deltas for the whole batch at once
        mfccs_all = mfccs_all.transpose(1, 2)  # (batch*channels, freq, time)
        
        if(derivatives):
            # Compute derivatives in parallel
            deltas_all = torchaudio.functional.compute_deltas(mfccs_all)
            ddeltas_all = torchaudio.functional.compute_deltas(deltas_all)
        
            # Concatenate features
            mfccs_all = torch.cat([mfccs_all, deltas_all, ddeltas_all], dim=1)  # (batch*channels, 3*freq, time)
                
        # Reshape back to (batch, channels, (3*)freqs, time)
        result = mfccs_all.reshape(batch_size, n_channels, -1, samples)
        
        return result

class MFCCEncoder(EncoderBase):
    def __init__(self, hparams_encoder, hparams_input_shape, static_stats_train):
        '''
        calculates MFCC features
        '''
        super().__init__(hparams_encoder, hparams_input_shape, static_stats_train)
        self.sequence_last = hparams_input_shape.sequence_last
        assert(hparams_input_shape.channels2==0)
        self.input_channels = hparams_input_shape.channels
        self.fs = hparams_encoder.fs
        self.frame_length_samples = hparams_encoder.frame_length_samples if hparams_encoder.frame_length_samples>0 else hparams_input_shape.length
        self.frame_shift_samples = hparams_encoder.frame_shift_samples if hparams_encoder.frame_shift_samples>0 else int(0.01*self.fs)

        self.flatten = hparams_encoder.flatten
        self.derivatives = hparams_encoder.derivatives

        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = self.input_channels*(39 if self.derivatives else 13) if self.flatten else self.input_channels
        self.output_shape.channels2 = 0 if self.flatten else hparams_encoder.num_mfccs
        self.output_shape.length = (hparams_input_shape.length - hparams_encoder.frame_length_samples)//hparams_encoder.frame_shift_samples+1
        self.output_shape.sequence_last = False if self.flatten else True
    
    def forward(self, **kwargs):
        seq = kwargs["seq"] #bs,channels,freq,seq
        if(not self.sequence_last):
            seq = torch.movedim(seq,1,-1) #bs,channels,seq

        mfcc_features = compute_mfcc_features_and_derivatives(seq, self.fs, self.frame_length_samples, self.frame_shift_samples, self.derivatives)#(batch, channels, (3*)freqs, time)

        if(self.flatten):
            # For flattened output: (batch, time, channels*features)
            bs, ch, feat, time = mfcc_features.shape
            return mfcc_features.permute(0, 3, 1, 2).reshape(bs, time, ch*feat)
        else:
            # For non-flattened output: (batch, channels, features, time)
            return mfcc_features

    def get_output_shape(self):
        return self.output_shape


@dataclass
class MFCCEncoderConfig(EncoderBaseConfig):
    _target_:str = "clinical_ts.ts.encoder.MFCCEncoder"
    fs:float = 100. #sampling frequency of the input data
    num_mfccs:int = 39 #13 for mfccs only, 39 for mfccs and two derivatives
    frame_length_samples:int = 0 #length of one frame in token default=0: entire sample
    frame_shift_samples:int = 0 #stride between frames in token
    derivatives:bool = True #return 13 MFCCs and 2*13 derivatives or only 13 MFCCs

def compute_spectrogram_features(
    x: torch.Tensor,
    samp_rate: float,
    nperseg: int,
    noverlap: int,
    window: Optional[torch.Tensor] = None,
    pad: bool = True,
    log_transform: bool = True,
    eps: float = 1e-10
):
    """
    Compute spectrogram features from a signal using PyTorch.
    Batch-parallelized implementation for multi-channel input.
    
    Parameters:
    -----------
    x : torch.Tensor
        Input signal with shape (bs, channels, samples) or (bs, samples)
    samp_rate : float
        Sampling rate of the signal in Hz
    nperseg : int
        Length of each segment
    noverlap : int
        Number of points to overlap between segments
    window : torch.Tensor, optional
        Window function to apply to each segment
    pad : bool
        Whether to apply reflection padding
    log_transform : bool
        Whether to apply log transformation to the spectrogram
    eps : float
        Small value to add before log transform to avoid log(0)
        
    Returns:
    --------
    torch.Tensor
        Spectrogram features with shape (batch_size, channels, time, frequency)
    """
    with torch.no_grad():
        # Save original shape information
        orig_shape = x.shape
        
        # Handle various input shapes
        if x.dim() == 2:
            # Could be (batch, samples) or (channels, samples)
            # We'll treat it as (batch, samples) and add channel dim
            x = x.unsqueeze(1)  # (batch, 1, samples)
        
        batch_size, n_channels, samples = x.shape
        
        # Create default window if not provided
        if window is None:
            window = torch.hann_window(nperseg, device=x.device)
        
        # Padding
        if pad:
            pad_length = nperseg // 2
            # Implement reflection padding in PyTorch
            x = torch.nn.functional.pad(
                x.transpose(1, 2),  # (batch, samples, channels)
                (0, 0, pad_length, pad_length),  # Pad samples dimension
                mode='reflect'
            ).transpose(1, 2)  # Back to (batch, channels, padded_samples)
        
        # Reshape to process all batch items and channels at once
        x_flat = x.reshape(-1, x.shape[-1])  # (batch*channels, samples)
        
        # Use torchaudio's spectrogram
        # This returns a ComplexTensor with shape (batch*channels, freq, time)
        spec_complex = torchaudio.functional.spectrogram(
            waveform=x_flat,
            pad=0,  # No additional padding since we did it above
            window=window,
            n_fft=nperseg,
            hop_length=nperseg - noverlap,
            win_length=nperseg,
            power=None,  # Return complex spectrum
            normalized=False,
            center=False,  # No additional padding
            pad_mode='reflect',
            onesided=True
        )
        
        # Convert complex spectrogram to magnitude
        spec_magnitude = torch.abs(spec_complex)
        
        # Apply log transform if requested
        if log_transform:
            spec_magnitude = 20 * torch.log10(spec_magnitude + eps)
        
        # Reshape back to (batch, channels, time, freq)
        result = spec_magnitude.permute(0, 2, 1)  # (batch*channels, time, freq)
        result = result.reshape(batch_size, n_channels, result.shape[1], result.shape[2])
        
        return result


class SpectrogramEncoder(EncoderBase):
    def __init__(self, hparams_encoder, hparams_input_shape, static_stats_train):
        """
        Calculates spectrogram features
        """
        super().__init__(hparams_encoder, hparams_input_shape, static_stats_train)
        self.sequence_last = hparams_input_shape.sequence_last
        assert(hparams_input_shape.channels2 == 0)
        self.input_channels = hparams_input_shape.channels
        self.fs = hparams_encoder.fs
        self.nperseg = hparams_encoder.nperseg
        self.noverlap = hparams_encoder.noverlap
        self.pad = hparams_encoder.pad
        self.log_transform = hparams_encoder.log_transform
        self.eps = hparams_encoder.eps
        self.flatten = hparams_encoder.flatten
        
        # Create window
        self.window = torch.hann_window(self.nperseg)
        
        # Calculate output shape
        n_freq_bins = self.nperseg // 2 + 1
                
        # Calculate time steps to match spectrogram calculation
        hop_length = self.nperseg - self.noverlap
        input_length = hparams_input_shape.length
        padded_length = input_length + (self.nperseg if self.pad else 0)
        time_steps = (padded_length - self.nperseg) // hop_length + 1

        self.output_shape = dataclasses.replace(hparams_input_shape)
        if self.flatten:
            self.output_shape.channels = self.input_channels * n_freq_bins
            self.output_shape.channels2 = 0
        else:
            self.output_shape.channels = self.input_channels
            self.output_shape.channels2 = n_freq_bins
        self.output_shape.length = time_steps
        self.output_shape.sequence_last = False if self.flatten else True
    
    def forward(self, **kwargs):
        seq = kwargs["seq"]  # bs, channels, seq
        if not self.sequence_last:
            seq = torch.movedim(seq, 1, -1)  # bs, channels, seq
        
        # Ensure window is on the same device as input
        window = self.window.to(seq.device)
        
        # Compute spectrogram
        spec = compute_spectrogram_features(
            seq, 
            self.fs, 
            self.nperseg, 
            self.noverlap, 
            window=window,
            pad=self.pad,
            log_transform=self.log_transform,
            eps=self.eps
        )  # (batch, channels, time, freq)
        
        if self.flatten:
            # Flatten channels and frequency dimensions, and transpose to get (batch, time, channels*freq)
            return spec.permute(0, 2, 1, 3).reshape(spec.shape[0], spec.shape[2], -1)
        else:
            # Return (batch, channels, freq, time)
            return spec.permute(0, 1, 3, 2)
    
    def get_output_shape(self):
        return self.output_shape


@dataclass
class SpectrogramEncoderConfig(EncoderBaseConfig):
    _target_: str = "clinical_ts.ts.encoder.SpectrogramEncoder"
    fs: float = 100.0  # sampling frequency of the input data
    nperseg: int = 256  # length of each segment
    noverlap: int = 128  # overlap between segments
    pad: bool = True  # whether to apply reflection padding
    log_transform: bool = True  # whether to apply log transformation
    eps: float = 1e-10  # small constant to add before log transform
    flatten: bool = False  # whether to flatten channels and frequency dimensions