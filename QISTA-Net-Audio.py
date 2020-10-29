print("\033[H\033[J")
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import scipy.io as sio
import time
from scipy import signal
#import pypesq
import pystoi

from utils.stft import stft
from utils.istft import istft
from utils.SNR import SNR

n_output = 256
bs = n_output # block size
bsh = int(bs/2)

something_wrong = False

output_file_name = "Results.txt"
dir_gt = '../../../audio_dataset_20200830/Valentini_2017/clean_testset_wav/'
dir_amp = '../93_amp_mat_48K_256/'
dir_fft = '../112_fft_mat_48K_256/'

waveform_gt_name = os.listdir(dir_gt)
waveform_no = len(waveform_gt_name)
test_pad_length = np.zeros([waveform_no])
SNR_fft = np.zeros([waveform_no])
SNR_time = np.zeros([waveform_no])
#pesq_time = np.zeros([waveform_no])
stoi_time = np.zeros([waveform_no])
SNR_time_trun = np.zeros([waveform_no])
stoi_time_trun = np.zeros([waveform_no])
LSD = np.zeros([waveform_no])
LSD_trun = np.zeros([waveform_no])

for wave_i in range(waveform_no):
    wave_name = [dir_gt + waveform_gt_name[wave_i]][0]
    if os.path.splitext(wave_name)[-1]=='.WAV' or os.path.splitext(wave_name)[-1]=='.wav':
        wave_file = wave_name
    waveform_gt, samplerate = sf.read(wave_file, dtype='float32') 
    if samplerate != 48000:
        print('  sample rate of this file is not 48000: ',end='')
        print(wave_file)
    wave_file = []
    len_waveform = waveform_gt.shape[0]
    block_fft_gt = stft(waveform_gt,bs)
    
    amp_file_name = [dir_amp + waveform_gt_name[wave_i] + '_recon_amp.mat'][0]
    block_amp_load = sio.loadmat(amp_file_name)
    block_amp_rec = block_amp_load['amp']
    
    fft_file_name = [dir_fft + waveform_gt_name[wave_i] + '_recon_fft.mat'][0]
    block_fft_load = sio.loadmat(fft_file_name)
    block_fft_rec = block_fft_load['fft']
    block_phase = np.angle(np.transpose(block_fft_rec))
    block_fft_combine = np.transpose(block_amp_rec * np.exp(1j * block_phase))
    
    SNR_fft[wave_i] = SNR(block_fft_gt,block_fft_combine)
    rec_waveform = istft(block_fft_combine)
    
    SNR_time[wave_i] = SNR(waveform_gt[:len_waveform],rec_waveform[:len_waveform])
    stoi_time[wave_i] = pystoi.stoi(waveform_gt[:len_waveform],rec_waveform.real[:len_waveform],samplerate,extended=False)
    
    mag_hr = np.abs(block_fft_gt)
    mag_sr = np.abs(block_fft_combine)
    power_mag_hr = mag_hr**2
    power_mag_sr = mag_sr**2
    log_power_mag_hr = np.log10(power_mag_hr)
    log_power_mag_sr = np.log10(power_mag_sr)
    log_power_mag_sr[log_power_mag_sr<1e-16] = 1e-16
    square_of_gap = (log_power_mag_hr-log_power_mag_sr)**2
    mean_gap = np.mean(square_of_gap,0)
    sqrt_ = np.sqrt(mean_gap)
    LSD[wave_i] = np.mean(sqrt_,0)
    
#    def compute_log_distortion(x_hr, x_pr):
#        lsd = numpy.mean(numpy.sqrt(numpy.mean((S1-S2)**2, axis=1)), axis=0)
#        return min(lsd, 10.)
    
#    s1 = int(len_waveform*0.005)
#    s2 = int(len_waveform*0.995)
    s1 = 12
    s2 = len_waveform-12
    SNR_time_trun[wave_i] = SNR(waveform_gt[s1:s2],rec_waveform[s1:s2])
    stoi_time_trun[wave_i] = pystoi.stoi(waveform_gt[s1:s2],rec_waveform.real[s1:s2],samplerate,extended=False)
    
    stft_hr = stft(waveform_gt[s1:s2],bs)
    stft_sr = stft(rec_waveform[s1:s2],bs)
    mag_hr = np.abs(stft_hr)
    mag_sr = np.abs(stft_sr)
    power_mag_hr = mag_hr**2
    power_mag_sr = mag_sr**2
    log_power_mag_hr = np.log10(power_mag_hr)
    log_power_mag_sr = np.log10(power_mag_sr)
    log_power_mag_sr[log_power_mag_sr<1e-16] = 1e-16
    square_of_gap = (log_power_mag_hr-log_power_mag_sr)**2
    mean_gap = np.mean(square_of_gap,0)
    sqrt_ = np.sqrt(mean_gap)
    LSD_trun[wave_i] = np.mean(sqrt_,0)
    
    d_name_gt = './20201012_GT_org/%s.wav' % (waveform_gt_name[wave_i][:-4])
    sf.write(d_name_gt, waveform_gt[:len_waveform], samplerate)
    d_name_rec = './20201012_recon_org_with_results/%s_SNR_%.2f_stoi_%.4f.wav' % (waveform_gt_name[wave_i][:-4],SNR_time[wave_i],stoi_time[wave_i])
    sf.write(d_name_rec, rec_waveform.real[:len_waveform], samplerate)
    d_name_rec = './20201012_recon_org/%s_recon.wav' % (waveform_gt_name[wave_i][:-4])
    sf.write(d_name_rec, rec_waveform.real[:len_waveform], samplerate)
    
    l1 = len(waveform_gt)
    l2 = len(rec_waveform.real[:len_waveform])
    print('length = ',l1,l2)
    if l1 != l2:
        print('here error')
        error_occur = True
    
    d_name_gt_trun = './20201012_GT_trun/%s.wav' % (waveform_gt_name[wave_i][:-4])
    sf.write(d_name_gt_trun, waveform_gt[s1:s2], samplerate)
    d_name_rec_trun = './20201012_recon_trun_with_results/%s_SNR_%.2f_stoi_%.4f.wav' % (waveform_gt_name[wave_i][:-4],SNR_time[wave_i],stoi_time[wave_i])
    sf.write(d_name_rec_trun, rec_waveform.real[s1:s2], samplerate)
    d_name_rec_trun = './20201012_recon_trun/%s_recon.wav' % (waveform_gt_name[wave_i][:-4])
    sf.write(d_name_rec_trun, rec_waveform.real[s1:s2], samplerate)
    
    l1 = len(waveform_gt[s1:s2])
    l2 = len(rec_waveform.real[s1:s2])
    print('length = ',l1,l2)
    if l1 != l2:
        print('here error')
        error_occur = True
    
#    mdic = {'waveform':rec_waveform[:len_waveform]}
#    n1 = ['./combine_results/' + waveform_gt_name[wave_i]][0]
#    n2 = '_recon_waveform'
#    n3 = '_SNR_%.2f' %SNR_time[wave_i]
#    n4 = '_SNRfft_%.2f' %SNR_fft[wave_i]
#    n5 = '_stoi_%.4f.mat' %stoi_time[wave_i]
#    name_recon_wave = [n1 + n2 + n3 + n4 + n5][0]
#    sio.savemat(name_recon_wave,mdic)
    
#    plt.figure(figsize=(12,6))
#    plt.subplot(211)
#    plt.plot(np.arange(1,len_waveform+1,1),(waveform_gt),color="b", linewidth=2,linestyle="-",label="ground-truth")
#    plt.xlabel('time',fontsize=16,fontweight='bold')
#    plt.ylabel('amplitude',fontsize=16,fontweight='bold')
#    #plt.xticks(np.arange(0,audio_len,samplerate))
##    plt.yticks(np.arange(-0.12,0.12,0.01))
#    to_title = [waveform_gt_name[wave_i] + ' ' + n3[1:] + n4[:] + n5[:-4]][0]
#    plt.title(to_title,fontsize=28,fontweight='bold')
#    plt.grid()
#    
#    plt.subplot(212)
#    plt.plot(np.arange(1,len_waveform+1,1),rec_waveform[:len_waveform],color="b", linewidth=2,linestyle="-",label="ground-truth")
#    plt.xlabel('time',fontsize=16,fontweight='bold')
#    plt.ylabel('amplitude',fontsize=16,fontweight='bold')
##    plt.yticks(np.arange(-0.12,0.12,0.01))
#    plt.grid()
#    plt_name = ['./results_plot/' + waveform_gt_name[wave_i][:-4] + '.png'][0]
#    plt.savefig(plt_name)
#    plt.show()
#    plt.close()

    print()
    print('wave %d, recon SNR:' %(wave_i+1))
    print('       fft : %.3f' %(SNR_fft[wave_i]))
    print('      time : %.3f' %(SNR_time[wave_i]))
    print('      stoi : %.4f' %(stoi_time[wave_i]))
    print('       LSD : %.4f' %(LSD[wave_i]))
    print('  trun_LSD : %.4f' %(LSD_trun[wave_i]))
    

SNR_time_mean = SNR_time.mean()
SNR_fft_mean = SNR_fft.mean()
stoi_time_mean = stoi_time.mean()
SNR_time_trun_mean = SNR_time_trun.mean()
stoi_time_trun_mean = stoi_time_trun.mean()
LSD_mean = LSD.mean()
LSD_trun_mean = LSD_trun.mean()
    
out1 = "Avg SNR is %.4f dB,\n" % (SNR_time_mean)
out2 = ' SNR in time is %.4f dB,\n' %(SNR_time_mean)
out3 = ' SNR in time-trun is %.4f dB,\n' %(SNR_time_trun_mean)
out4 = ' SNR in stft is %.4f dB,\n' %(SNR_fft_mean)
out5 = ' stoi is %.4f dB,\n' %(stoi_time_mean)
out6 = ' stoi-trun is %.4f dB,\n' %(stoi_time_trun_mean)
out7 = ' LSD is %.4f dB,\n' %(LSD_mean)
out8 = ' LSD-trun is %.4f dB,\n\n' %(LSD_trun_mean)
output_data_recon = [out1+out2+out3+out4+out5+out6+out7+out8][0]
print('')
print(output_data_recon)
output_file = open(output_file_name, 'a')
output_file.write(output_data_recon)
output_file.close()
    




## for read testing
#all_data = os.listdir('./')
#        
#file_0_gt = all_data[3]
#print('ground-truth :', file_0_gt)
#file_0_rec = all_data[4]
#print('reconstruction :', file_0_rec)
#
#audio_1_GT, samplerate = sf.read(file_0_gt, dtype='float32')  
##wav_play(audio_1_GT,3,samplerate)
#audio_1_rec, samplerate = sf.read(file_0_rec, dtype='float32')  
##wav_play(audio_1_rec,3,samplerate)
#
#audio_len = audio_1_GT.shape[0]
#x_axis = np.arange(1,audio_len+1,1)
#n1,n2=0,audio_len # plot range
##n1,n2=14000,14200 # plot range
#plt.figure(figsize=(12,6))
#plt.subplot(131)
#plt.plot(x_axis[n1:n2],(audio_1_GT[n1:n2])*0.1,color="b", linewidth=2,linestyle="-",label="ground-truth")
#plt.xlabel('time',fontsize=16,fontweight='bold')
#plt.ylabel('amplitude',fontsize=16,fontweight='bold')
##plt.xticks(np.arange(0,audio_len,samplerate))
#plt.yticks(np.arange(-0.12,0.12,0.01))
#plt.grid()
#
#plt.subplot(132)
#plt.plot(x_axis[n1:n2],(audio_1_rec[n1:n2])*0.1,color="b", linewidth=2,linestyle="-",label="ground-truth")
#plt.xlabel('time',fontsize=16,fontweight='bold')
#plt.ylabel('amplitude',fontsize=16,fontweight='bold')
#plt.yticks(np.arange(-0.12,0.12,0.01))
#plt.grid()
#
#plt.subplot(133)
#plt.plot(x_axis[n1:n2],(data_test[0,n1:n2])*0.1,color="b", linewidth=2,linestyle="-",label="ground-truth")
#plt.xlabel('time',fontsize=16,fontweight='bold')
#plt.ylabel('amplitude',fontsize=16,fontweight='bold')
#plt.yticks(np.arange(-0.12,0.12,0.01))
#plt.grid()
#
#plt.show()
