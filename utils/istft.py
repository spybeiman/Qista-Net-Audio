from scipy import signal
def istft(wav_data):
    _, x_ifft = signal.istft(wav_data, fs=1.0,
        window='hann', nperseg=None,
        noverlap=None, nfft=None,
        input_onesided=False, boundary=None,
        time_axis=- 1, freq_axis=- 2)
    return x_ifft
