print("\033[H\033[J")
import soundfile as sf
import numpy as np
import os
import tensorflow as tf
import scipy.io as sio
import time
from scipy import signal

from utils.stft import stft
from utils.istft import istft
from utils.SNR import SNR

########## setting area begin ##########
is_testing = False # False/True : Train/Test
# paths of dataset directory (training set, validation set, test set)
dir_train = './audio_dataset/Valentini_2017/'
dir_valid = './audio_dataset/Valentini_2017/'
dir_test = './audio_dataset/Valentini_2017/clean_testset_wav/'
if is_testing == True:
    cpkt_model_number = 100
########## setting area end ############

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

q = 0.5
max_layer = 9

SR_ratio = 2
n_output = 256
n_input = int(n_output/SR_ratio)
batch_size = 64
EpochNum = 600

bs = n_output # block size
bsh = int(bs/2)

something_wrong = False

X_input = tf.placeholder(tf.float32, [None, 2, n_input])
X_output = tf.placeholder(tf.float32, [None, 2, n_output])

def add_con2d_weight_bias(w_shape, b_shape, order_no):
    Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_%d' % order_no)
    biases = tf.Variable(tf.random_normal(b_shape, stddev=0.05), name='biases_%d' % order_no)
    return [Weights, biases]

def add_fc(shape1, order_no):
    AA = tf.get_variable(shape=shape1, initializer=tf.contrib.layers.xavier_initializer(), name='FC_%d' % order_no, dtype=tf.float32)
    return AA

def QISTA(input_layers, QY,ATA):
    step_size = tf.Variable(1e-1, dtype=tf.float32)
    alpha = tf.Variable(1e-5, dtype=tf.float32)
    beta = tf.Variable(1.0, dtype=tf.float32)
    
    X_temp_real = tf.reshape(input_layers[-1][:,0,:], shape=[-1,n_output])
    X_temp_imag = tf.reshape(input_layers[-1][:,1,:], shape=[-1,n_output])
    QY_real = tf.reshape(QY[:,0,:], shape=[-1,n_output])
    QY_imag = tf.reshape(QY[:,1,:], shape=[-1,n_output])
    
    x1_ista_real = tf.reshape(tf.add(X_temp_real - tf.scalar_mul(step_size, tf.matmul(X_temp_real, ATA)), tf.scalar_mul(step_size, QY_real)), shape=[-1,1,n_output])
    x1_ista_imag = tf.reshape(tf.add(X_temp_imag - tf.scalar_mul(step_size, tf.matmul(X_temp_imag, ATA)), tf.scalar_mul(step_size, QY_imag)), shape=[-1,1,n_output])
    x1_ista = tf.stack([x1_ista_real,x1_ista_imag],axis=1)

    x2_ista = tf.reshape(x1_ista, shape=[-1, 2, n_output, 1])
    
    [Weights0, bias0] = add_con2d_weight_bias([2, 2, 1, 32], [32], 0)
    [Weights1, bias1] = add_con2d_weight_bias([2, 2, 32, 32], [32], 1)
    [Weights2, bias2] = add_con2d_weight_bias([2, 2, 32, 32], [32], 2)
    [Weights3, bias3] = add_con2d_weight_bias([2, 2, 32, 32], [32], 3)
    [Weights4, bias4] = add_con2d_weight_bias([2, 2, 32, 32], [32], 4)
    [Weights5, bias5] = add_con2d_weight_bias([2, 2, 32, 32], [32], 5)
    [Weights6, bias6] = add_con2d_weight_bias([2, 2, 32, 32], [32], 6)
    [Weights7, bias7] = add_con2d_weight_bias([2, 2, 32, 1], [1], 7)
    
    x3_ista = tf.nn.conv2d(x2_ista, Weights0, strides=[1, 1, 1, 1], padding='SAME')
    x4_ista = tf.nn.relu(tf.nn.conv2d(x3_ista, Weights1, strides=[1, 1, 1, 1], padding='SAME'))
    x40_ista = tf.nn.relu(tf.nn.conv2d(x4_ista, Weights2, strides=[1, 1, 1, 1], padding='SAME'))
    x44_ista = tf.nn.conv2d(x40_ista, Weights3, strides=[1, 1, 1, 1], padding='SAME')
    
    trun_param = alpha / ((0.1 + tf.abs(x44_ista))**(1-q))
    x50_ista = tf.multiply(tf.sign(x44_ista), tf.nn.relu(tf.abs(x44_ista) - trun_param))
    x5_ista = x50_ista - x44_ista
    
    x6_ista = tf.nn.relu(tf.nn.conv2d(x5_ista, Weights4, strides=[1, 1, 1, 1], padding='SAME'))
    x60_ista = tf.nn.relu(tf.nn.conv2d(x6_ista, Weights5, strides=[1, 1, 1, 1], padding='SAME'))
    x66_ista = tf.nn.conv2d(x60_ista, Weights6, strides=[1, 1, 1, 1], padding='SAME')
    x7_ista = tf.nn.conv2d(x66_ista, Weights7, strides=[1, 1, 1, 1], padding='SAME')

    x7_ista = x2_ista + beta * x7_ista
    x8_ista = tf.reshape(x7_ista, shape=[-1, 2, n_output])
    
    x3_ista_sym = tf.nn.relu(tf.nn.conv2d(x3_ista, Weights1, strides=[1, 1, 1, 1], padding='SAME'))
    x30_ista_sym = tf.nn.relu(tf.nn.conv2d(x3_ista_sym, Weights2, strides=[1, 1, 1, 1], padding='SAME'))
    x4_ista_sym = tf.nn.conv2d(x30_ista_sym, Weights3, strides=[1, 1, 1, 1], padding='SAME')
    x60_ista_sym = tf.nn.relu(tf.nn.conv2d(x4_ista_sym, Weights4, strides=[1, 1, 1, 1], padding='SAME'))
    x6_ista_sym = tf.nn.relu(tf.nn.conv2d(x60_ista_sym, Weights5, strides=[1, 1, 1, 1], padding='SAME'))
    x7_ista_sym = tf.nn.conv2d(x6_ista_sym, Weights6, strides=[1, 1, 1, 1], padding='SAME')

    x11_ista = x7_ista_sym - x3_ista

    return [x8_ista, x11_ista]

A_np = np.zeros([n_input,n_output], dtype=np.float32)
for i in range(n_input):
    A_np[i,2*i] = 1
AT_np = A_np.transpose()
AT_tf = tf.convert_to_tensor(AT_np, dtype=tf.float32)

def inference_QISTA(max_layer, X_input, reuse):
    YT_real = X_input[:,0,:]
    YT_imag = X_input[:,1,:]
    AT = add_fc([n_output,n_input], 100)
    ATT = add_fc([n_input,n_output], 101)

    QY_real = tf.reshape(tf.matmul(YT_real, ATT), shape=[-1,1,n_output])
    QY_imag = tf.reshape(tf.matmul(YT_imag, ATT), shape=[-1,1,n_output])
    QY = tf.stack([QY_real,QY_imag],axis=1)
    
    ATA = tf.matmul(AT,ATT)
    
    layers = []
    layers_symetric = []
    layers.append(QY)
    for layer_i in range(max_layer):
        with tf.variable_scope('conv_%d' %layer_i, reuse=reuse):
            [conv1, conv1_sym] = QISTA(layers, QY, ATA)
            layers.append(conv1)
            layers_symetric.append(conv1_sym)
    return [layers, layers_symetric]

[Prediction, Pre_symetric] = inference_QISTA(max_layer, X_input, reuse=False)

def compute_cost(Prediction, X_output):
    cost = tf.reduce_mean(tf.square(Prediction[-1] - X_output))
    cost_sym = 0
    for k in range(max_layer):
        cost_sym += tf.reduce_mean(tf.square(Pre_symetric[k]))
    return [cost, cost_sym]

[cost, cost_sym] = compute_cost(Prediction, X_output)

cost_all = cost + 0.01*cost_sym
optm_all = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost_all)

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)

sess = tf.Session(config=config)
sess.run(init)

model_dir = 'Layer_%d_ratio_%d_Model' % (max_layer, SR_ratio)

if is_testing == False:
    print("Start Training...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    output_file_name_SNR = "Validation_Results_%s.txt" % (model_dir)
else:
    output_file_name = "Testing_Results_%s_Model_%d.txt" % (model_dir, cpkt_model_number)

if is_testing == False:
    for epoch_i in range(EpochNum):
        epoch_using_time_begin = time.time()
        
        ######## load training data begin ########
        data_load_begin = time.time()
        n_sr = [dir_train + 'training_data_sr_stft.mat'][0]
        n_lr = [dir_train + 'training_data_lr_stft.mat'][0]
        block_sr = sio.loadmat(n_sr)
        block_lr = sio.loadmat(n_lr)
        training_data_sr_stft = np.transpose(block_sr['training_data'])
        training_data_lr_stft = np.transpose(block_lr['training_data'])
        print('load training data cost %.2f sec' %(time.time() - data_load_begin))
        train_len = training_data_lr_stft.shape[0]
        total_batch = train_len // batch_size
        
        training_data_sr_real = (training_data_sr_stft.real).reshape(train_len,1,bs)
        training_data_sr_imag = (training_data_sr_stft.imag).reshape(train_len,1,bs)
        training_data_sr = np.append(training_data_sr_real,training_data_sr_imag,axis=1)
        print(training_data_sr.shape)
        del training_data_sr_stft,training_data_sr_real,training_data_sr_imag
        training_data_lr_real = (training_data_lr_stft.real).reshape(train_len,1,bsh)
        training_data_lr_imag = (training_data_lr_stft.imag).reshape(train_len,1,bsh)
        training_data_lr = np.append(training_data_lr_real,training_data_lr_imag,axis=1)
        print(training_data_lr.shape)
        del training_data_lr_stft,training_data_lr_real,training_data_lr_imag
        ######## load training data end ########
        
        randidx_all = np.random.permutation(train_len)
        for batch_i in range(total_batch):
            print('\rtraining epoch {0}, batch {1}/{2}'.format(epoch_i+1,batch_i+1,total_batch),end='')
            randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]
            batch_train_sr = training_data_sr[randidx,:,:]
            batch_train_lr = training_data_lr[randidx,:,:]
            feed_train = {X_input: batch_train_lr, X_output: batch_train_sr}
            sess.run(optm_all, feed_dict=feed_train)
        epoch_using_time = time.time() - epoch_using_time_begin
        print('')
        print('epoch {0} cost {1:<.3f} sec'.format(epoch_i+1,epoch_using_time))

        # validation begin
        
        ######## load validation data begin ########
        data_load_begin = time.time()
        dir_t = '../../audio_dataset_20200830/Valentini_2017/data_generated_by_step_3_256/'
        n_sr = [dir_t + 'validation_data_256_14789_sr_stft.mat'][0]
        n_lr = [dir_t + 'validation_data_256_14789_lr_stft.mat'][0]
        block_sr = sio.loadmat(n_sr)
        block_lr = sio.loadmat(n_lr)
        valid_data_sr_stft = np.transpose(block_sr['validation_data'])
        valid_data_lr_stft = np.transpose(block_lr['validation_data'])
        print('load validation data cost %.2f sec' %(time.time() - data_load_begin))
        valid_len = valid_data_lr_stft.shape[0]
        
        valid_data_sr_real = (valid_data_sr_stft.real).reshape(valid_len,1,bs)
        valid_data_sr_imag = (valid_data_sr_stft.imag).reshape(valid_len,1,bs)
        valid_data_sr = np.append(valid_data_sr_real,valid_data_sr_imag,axis=1)
        print(valid_data_sr.shape)
        del valid_data_sr_stft,valid_data_sr_real,valid_data_sr_imag
        valid_data_lr_real = (valid_data_lr_stft.real).reshape(valid_len,1,bsh)
        valid_data_lr_imag = (valid_data_lr_stft.imag).reshape(valid_len,1,bsh)
        valid_data_lr = np.append(valid_data_lr_real,valid_data_lr_imag,axis=1)
        print(valid_data_lr.shape)
        del valid_data_lr_stft,valid_data_lr_real,valid_data_lr_imag
        ######## load validation data end ########
        
        valid_batch = valid_len // batch_size
        randidx_all_valid = np.random.permutation(valid_len)
        rec_SNR_stft = np.zeros(valid_batch)
        recon_begin = time.time()
        
        for batch_i in range(valid_batch):
            randidx_valid = randidx_all_valid[batch_i*batch_size:(batch_i+1)*batch_size]
            x_stft_sr = valid_data_sr[randidx_valid,:,:]
            x_stft_lr = valid_data_lr[randidx_valid,:,:]
            feed_test = {X_input: x_stft_lr, X_output: x_stft_sr}
            Prediction_value = sess.run(Prediction, feed_dict=feed_test)
            rec_SNR_stft[batch_i] = SNR(x_stft_sr, Prediction_value[-1])
        recon_using_time = time.time() - recon_begin
        SNR_stft_mean = np.mean(rec_SNR_stft)
        
        saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
        output_data_recon = "  Avg SNR in real_imag is "
        input_text = '[Epoch %d, SNR=%.4f]' % (epoch_i+1, SNR_stft_mean)
        output_data_recon = [output_data_recon[0] + input_text + '\n']
        input_text = "cpkt NO. is %d, recover using time %.3f sec\n" % (epoch_i, recon_using_time)
        print('')
        print(output_data_recon[0])
        print(input_text)
        output_file = open(output_file_name_SNR, 'a')
        output_file.write(output_data_recon[0])
        output_file.close()
    print("Training Finished")
else:
    print('Begin Testing')
    test_data_num = 1344
    count_wave_files = 0
    test_pad_length = np.zeros([test_data_num])
    rec_SNR_time = np.zeros([test_data_num])
    rec_SNR_mag = np.zeros([test_data_num])
    saver.restore(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, cpkt_model_number))
    for dialect_region in range(8):
        test_dir = ['../TEST/DR' + str(dialect_region+1)][0]
        speaker_id = os.listdir(test_dir)
        for i in range(len(speaker_id)):
            speaker = [test_dir + '/' + speaker_id[i]][0]
            files = os.listdir(speaker)
            if len(files) != 40:
                print('an error in ', speaker)
            for j in range(len(files)):
                if os.path.splitext(files[j])[-1]=='.WAV':
                    if os.path.splitext(files[j])[0]=='SA1':
                        continue
                    if os.path.splitext(files[j])[0]=='SA2':
                        continue
                    wav_file = [speaker + '/' + files[j]][0]
                    
                    data_temp, samplerate = sf.read(wav_file, dtype='float32')  
                    
                    len_lr = data_temp.shape[0]
                    if len_lr%2 == 1:
                        data_temp = data_temp[:-1]
                    test_pad_length[count_wave_files] = (128-data_temp.shape[0]%128)%128
                    data_temp_lr = data_temp.reshape([int(len_lr/2),2])
                    
                    if samplerate != 16000:
                        print('  sample rate of this file is not 16000: ',end='')
                        print(wav_file)
                    f, t, x_stft = signal.stft(data_temp, fs=1.0, window='hann', 
                      nperseg=256, noverlap=None, nfft=None, 
                      detrend=False, return_onesided=False, 
                      boundary=None, #['even','odd','constant','zeros',None]
                      padded=True, axis=-1)
                    f_lr, t_lr, x_stft_lr = signal.stft(data_temp_lr[:,0], fs=1.0, window='hann', 
                      nperseg=128, noverlap=None, nfft=None, 
                      detrend=False, return_onesided=False, 
                      boundary=None, #['even','odd','constant','zeros',None]
                      padded=True, axis=-1)
                    test_data_block_mag = np.abs(np.transpose(x_stft))
                    test_data_block_ang = np.angle(np.transpose(x_stft))
                    test_data_block_mag_lr = np.abs(np.transpose(x_stft_lr))
                    
                    batch_test_sr = test_data_block_mag
                    batch_test_lr = test_data_block_mag_lr
                    feed_test = {X_input: batch_test_lr, X_output: batch_test_sr}
                    recon_time_begin = time.time()
                    Prediction_value = sess.run(Prediction[-1], feed_dict=feed_test)
                    recon_using_time = time.time() - recon_time_begin
                    temp_mag = Prediction_value.astype(np.float32)
                    
                    if j >= 36: # only final sentence of a speaker
                        d_name = '%s_%s_recon_magnitude_pad_%3d.mat' % (speaker_id[i],files[j],int(test_pad_length[count_wave_files]))
                        sio.savemat(d_name,{'recon':np.transpose(temp_mag)})
                    
                    rec_SNR_mag[count_wave_files] = SNR.SNR(batch_test_sr,temp_mag)
                    
                    # from magnitude and phase, to fft
                    recon_block_stft = np.transpose(temp_mag * np.exp(1j * test_data_block_ang))
                    # from fft to time
                    t_recon, recon_time = signal.istft(recon_block_stft, fs=1.0, window='hann',
                        nperseg=None, noverlap=None, nfft=None,
                        input_onesided=False, boundary=None,
                        time_axis=-1, freq_axis=-2)
                    last_ = int(-1.0*test_pad_length[count_wave_files])
                    if last_ != 0:
                        recon_time = recon_time[:last_] 
                    
                    rec_SNR_time[count_wave_files] = SNR.SNR(data_temp,recon_time)
                    
                    print('audio no. {0:d}/{1:d}, SNR = {2:.3f}, SNR_mag = {3:.3f}, using time {4:.3f} sec'.format(count_wave_files+1,test_data_num,rec_SNR_time[count_wave_files],rec_SNR_mag[count_wave_files],recon_using_time))
                    
                    if j >= 36: # only final sentence of a speaker
                        d_name = '%s_%s_ground_truth.wav' % (speaker_id[i],files[j])
                        sf.write(d_name, data_temp, samplerate)
                        d_name = '%s_%s_recover_SNR_%.2f_SNRamp_%.2f.wav' %(speaker_id[i],files[j],rec_SNR_time[count_wave_files],rec_SNR_mag[count_wave_files])
                        sf.write(d_name, recon_time.real, samplerate)
                         
                    count_wave_files += 1
                
    SNR_time_mean = rec_SNR_time.mean()
    SNR_mag_mean = rec_SNR_mag.mean()

#    rec_pesq = np.zeros(test_data_num)
#    rec_stoi = np.zeros(test_data_num)
    
#    output_data_recon_1 = "Avg SNR is %.2f dB, pesq is %.4f, stoi is %.4f, cpkt NO. is %d \n" % (SNR_mean, pesq_mean, stoi_mean, cpkt_model_number)
#    output_data_recon_2 = "Avg SNR in amp is %.2f dB, stoi in amp is %.4f \n" % (SNR_mag_mean,  stoi_mag_mean)
    output_data_recon = "Avg SNR is %.2f dB, SNR in amp is %.2f dB, cpkt NO. is %d \n" % (SNR_time_mean, SNR_mag_mean, cpkt_model_number)
    print('')
    print(output_data_recon)
    output_file = open(output_file_name, 'a')
    output_file.write(output_data_recon)
    output_file.close()
    
sess.close()




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
