print("\033[H\033[J")
import soundfile as sf
import numpy as np
import os
import tensorflow as tf
import scipy.io as sio
import time

from utils.stft import stft
from utils.SNR import SNR

########## setting area begin ##########
is_testing = True # False/True : Train/Test
# paths of dataset directory (training set, validation set, test set)
dir_train = './audio_dataset/Valentini_2017/'
dir_valid = './audio_dataset/Valentini_2017/'
dir_test = './audio_dataset/Valentini_2017/clean_testset_wav/'
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

model_dir = './model/'

if is_testing == False:
    print("Start Training...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    output_file_name_SNR = 'Validation_Results.txt'
else:
    output_file_name = 'Testing_Results.txt'

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

        ######## load validation data begin ########
        data_load_begin = time.time()
        n_sr = [dir_valid + 'validation_data_sr_stft.mat'][0]
        n_lr = [dir_valid + 'validation_data_lr_stft.mat'][0]
        block_sr = sio.loadmat(n_sr)
        block_lr = sio.loadmat(n_lr)
        valid_data_sr_stft = np.transpose(block_sr['validation_data'])
        valid_data_lr_stft = np.transpose(block_lr['validation_data'])
        print('load validation data cost %.2f sec' %(time.time() - data_load_begin))
        valid_len = valid_data_lr_stft.shape[0]
        
        valid_data_sr_real = (valid_data_sr_stft.real).reshape(valid_len,1,bs)
        valid_data_sr_imag = (valid_data_sr_stft.imag).reshape(valid_len,1,bs)
        valid_data_sr_reim = np.append(valid_data_sr_real,valid_data_sr_imag,axis=1)
        del valid_data_sr_stft,valid_data_sr_real,valid_data_sr_imag
        valid_data_lr_real = (valid_data_lr_stft.real).reshape(valid_len,1,bsh)
        valid_data_lr_imag = (valid_data_lr_stft.imag).reshape(valid_len,1,bsh)
        valid_data_lr_reim = np.append(valid_data_lr_real,valid_data_lr_imag,axis=1)
        del valid_data_lr_stft,valid_data_lr_real,valid_data_lr_imag
        ######## load validation data end ########
        
        valid_batch = valid_len // batch_size
        randidx_all_valid = np.random.permutation(valid_len)
        rec_SNR_reim = np.zeros(valid_batch)
        recon_begin = time.time()
        
        for batch_i in range(valid_batch):
            randidx_valid = randidx_all_valid[batch_i*batch_size:(batch_i+1)*batch_size]
            x_reim_sr = valid_data_sr_reim[randidx_valid,:,:]
            x_reim_lr = valid_data_lr_reim[randidx_valid,:,:]
            feed_test = {X_input: x_reim_lr, X_output: x_reim_sr}
            Prediction_value = sess.run(Prediction[-1], feed_dict=feed_test)
            rec_SNR_reim[batch_i] = SNR(x_reim_sr, Prediction_value)
        recon_using_time = time.time() - recon_begin
        SNR_reim_mean = np.mean(rec_SNR_reim)
        
        saver.save(sess, './%s/Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
        input_text1 = "  Avg SNR in RI image is "
        input_text2 = '[Epoch %d, SNR=%.4f]' % (epoch_i+1, SNR_reim_mean)
        output_data_recon = [input_text1 + input_text2 + '\n'][0]
        input_text = "cpkt NO. is %d, recover using time %.3f sec\n" % (epoch_i, recon_using_time)
        print('')
        print(output_data_recon)
        print(input_text)
        output_file = open(output_file_name_SNR, 'a')
        output_file.write(output_data_recon)
        output_file.close()
    print("Training Finished")
else:
    print('Begin Testing')
    test_data = os.listdir(dir_test)
    test_data_num = len(test_data)
    rec_SNR_reim = np.zeros([test_data_num])
    
    saver.restore(sess, '%sSaved_Model_Qista-Net-n.cpkt' % (model_dir))
    
    for wave_i in range(test_data_num):
        wave_name = [dir_test + test_data[wave_i]][0]
        if os.path.splitext(wave_name)[-1]=='.WAV' or os.path.splitext(wave_name)[-1]=='.wav':
            wave_file = wave_name
        waveform_temp, samplerate = sf.read(wave_file, dtype='float32') 
        wave_file = []
        len_waveform = waveform_temp.shape[0]
        if len_waveform%2 == 1:
            waveform = np.concatenate((waveform_temp,waveform_temp[-1].reshape(1,)),axis=0) # extend by last item
        else:
            waveform = waveform_temp
        len_waveform_lr = int(waveform.shape[0]/2)
        
        waveform_lr_temp = waveform.reshape([len_waveform_lr,2])
        waveform_lr = waveform_lr_temp[:,0]
        
        x_stft_sr = stft(waveform,bs)
        x_stft_lr = stft(waveform_lr,bsh)
        len_sr = x_stft_sr.shape[1]
        len_lr = x_stft_lr.shape[1]
    
        test_block_real_sr = np.transpose(x_stft_sr.real).reshape(len_sr,1,bs)
        test_block_imag_sr = np.transpose(x_stft_sr.imag).reshape(len_sr,1,bs)
        test_block_reim_sr = np.append(test_block_real_sr,test_block_imag_sr,axis=1)
        test_block_real_lr = np.transpose(x_stft_lr.real).reshape(len_lr,1,bsh)
        test_block_imag_lr = np.transpose(x_stft_lr.imag).reshape(len_lr,1,bsh)
        test_block_reim_lr = np.append(test_block_real_lr,test_block_imag_lr,axis=1)
    
        batch_test_sr = test_block_reim_sr
        batch_test_lr = test_block_reim_lr
        feed_test = {X_input: batch_test_lr, X_output: batch_test_sr}
        recon_time_begin = time.time()
        Prediction_value = sess.run(Prediction[-1], feed_dict=feed_test)
        recon_using_time = time.time() - recon_time_begin
        
        recon_reim = Prediction_value.astype(np.float32)
        rec_SNR_reim[wave_i] = SNR(test_block_reim_sr,recon_reim)
        
        recon_real = recon_reim[:,0,:]
        recon_imag = recon_reim[:,1,:]
        recon_stft = np.transpose(recon_real + 1j * recon_imag)
        
        dir_stft = './stft_mat/'
        if not os.path.exists(dir_stft):
            os.makedirs(dir_stft)
        mdic = {'stft':recon_stft}
        name_stft = [dir_stft + test_data[wave_i] + '_recon_stft.mat'][0]
        sio.savemat(name_stft,mdic)
        
        print('wave %d, recon SNR of RI image : %.3f' %(wave_i+1, rec_SNR_reim[wave_i]),end='')
        print('  cost %.2f sec' %(recon_using_time))
        
    SNR_reim_mean = rec_SNR_reim.mean()

    output_data_recon = "Avg SNR of RI image: %.2f dB,\n" % (SNR_reim_mean)
    print('')
    print(output_data_recon)
    output_file = open(output_file_name, 'a')
    output_file.write(output_data_recon)
    output_file.close()
    
sess.close()


