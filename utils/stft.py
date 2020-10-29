from scipy import signal
def stft(wav_data,b_size):
    _, _, x_fft = signal.stft(wav_data, fs=1.0, window='hann', 
                      nperseg=b_size, noverlap=None, nfft=None, 
                      detrend=False, return_onesided=False, 
                      boundary=None, #['even','odd','constant','zeros',None]
                      padded=True, axis=-1)
    return x_fft
