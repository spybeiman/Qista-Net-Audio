print("\033[H\033[J")
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import scipy.io as sio
from scipy import signal
#import pypesq
import pystoi

from utils.stft import stft
from utils.istft import istft
from utils.SNR import SNR

########## setting area begin ##########
# paths of dataset directory (training set, validation set, test set)
dir_gt = './audio_dataset/Valentini_2017/clean_testset_wav/'
dir_mag = './mag_mat/'
dir_stft = './stft_mat/'
########## setting area end ############

n_output = 256
bs = n_output # block size
bsh = int(bs/2)

something_wrong = False

output_file_name = "Results.txt"

waveform_gt_name = os.listdir(dir_gt)
waveform_no = len(waveform_gt_name)
SNR_time = np.zeros([waveform_no])
stoi_time = np.zeros([waveform_no])

for wave_i in range(waveform_no):
    wave_name = [dir_gt + waveform_gt_name[wave_i]][0]
    if os.path.splitext(wave_name)[-1]=='.WAV' or os.path.splitext(wave_name)[-1]=='.wav':
        wave_file = wave_name
    waveform_gt, samplerate = sf.read(wave_file, dtype='float32') 
    wave_file = []
    len_waveform = waveform_gt.shape[0]
    block_stft_gt = stft(waveform_gt,bs)
    
    mag_file_name = [dir_mag + waveform_gt_name[wave_i] + '_recon_mag.mat'][0]
    block_mag_load = sio.loadmat(mag_file_name)
    block_mag_rec = block_mag_load['mag']
    
    stft_file_name = [dir_stft + waveform_gt_name[wave_i] + '_recon_stft.mat'][0]
    block_stft_load = sio.loadmat(stft_file_name)
    block_stft_rec = block_stft_load['stft']
    block_phase = np.angle(np.transpose(block_stft_rec))
    block_stft_combine = np.transpose(block_mag_rec * np.exp(1j * block_phase))
    
    rec_waveform = istft(block_stft_combine)
    
    SNR_time[wave_i] = SNR(waveform_gt[16:len_waveform-16],rec_waveform[16:len_waveform-16])
    stoi_time[wave_i] = pystoi.stoi(waveform_gt[16:len_waveform-16],rec_waveform.real[16:len_waveform-16],samplerate,extended=False)
    
    print()
    print('wave %d, recon ' %(wave_i+1))
    print(' SNR : %.3f' %(SNR_time[wave_i]))
    print(' stoi : %.4f\n' %(stoi_time[wave_i]))
    

SNR_time_mean = SNR_time.mean()
stoi_time_mean = stoi_time.mean()
    
out1 = 'Avg SNR in time is %.4f dB,\n' %(SNR_time_mean)
out2 = ' stoi is %.4f dB,\n\n' %(stoi_time_mean)
output_data_recon = [out1+out2][0]
print('')
print(output_data_recon)
output_file = open(output_file_name, 'a')
output_file.write(output_data_recon)
output_file.close()
    
