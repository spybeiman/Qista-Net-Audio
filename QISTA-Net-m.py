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
max_layer = 20

SR_ratio = 2
n_output = 256
n_input = int(n_output/SR_ratio)
batch_size = 64
EpochNum = 25000

bs = n_output # block size
bsh = int(bs/2)

something_wrong = False

X_input = tf.placeholder(tf.float32, [None, n_input])
X_output = tf.placeholder(tf.float32, [None, n_output])

def add_fc(shape1, order_no):
    AA = tf.get_variable(shape=shape1, initializer=tf.contrib.layers.xavier_initializer(), name='FC_%d' % order_no, dtype=tf.float32)
    return AA

def QISTA(input_layers, QY,ATA):
    step_size = tf.Variable(1e-1, dtype=tf.float32)
    alpha = tf.Variable(1e-5, dtype=tf.float32)
    
    x1_ista = tf.add(input_layers[-1] - tf.scalar_mul(step_size, tf.matmul(input_layers[-1], ATA)), tf.scalar_mul(step_size, QY))    

    trun_param = alpha / ((0.1 + tf.abs(x1_ista))**(1-q))
    x2_ista = tf.multiply(tf.sign(x1_ista), tf.nn.relu(tf.abs(x1_ista) - trun_param))
    
    return x2_ista

A_np = np.zeros([n_input,n_output], dtype=np.float32)
for i in range(n_input):
    A_np[i,2*i] = 1
AT_np = A_np.transpose()
AT_tf = tf.convert_to_tensor(AT_np, dtype=tf.float32)

def inference_QISTA(max_layer, X_input, reuse):
    YT = X_input
    ATT = add_fc([n_input,n_output], 101)

    QY = tf.matmul(YT, ATT)
    ATA = tf.matmul(AT_tf,ATT)
    
    layers = []
    layers.append(QY)
    for layer_i in range(max_layer):
        with tf.variable_scope('conv_%d' %layer_i, reuse=reuse):
            conv1 = QISTA(layers, QY,ATA)
            layers.append(conv1)
    return layers

Prediction = inference_QISTA(max_layer, X_input, reuse=False)

def compute_cost(Prediction, X_output):
    cost = tf.nn.l2_loss(Prediction[-1] - X_output)
    return cost

cost_all = compute_cost(Prediction, X_output)

optm_all = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost_all)

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

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
        n_sr = [dir_train + 'training_data_sr_mag.mat'][0]
        n_lr = [dir_train + 'training_data_lr_mag.mat'][0]
        block_sr = sio.loadmat(n_sr)
        block_lr = sio.loadmat(n_lr)
        training_data_sr_mag = block_sr['training_data_mag']
        training_data_lr_mag = block_lr['training_data_mag']
        print('load training data cost %.2f sec' %(time.time() - data_load_begin))
        train_len = training_data_lr_mag.shape[0]
        total_batch = train_len // batch_size
        ######## load training data end ########
        
        randidx_all = np.random.permutation(train_len)
        for batch_i in range(total_batch):
            print('\rtraining epoch {0}, batch {1}/{2}'.format(epoch_i+1,batch_i+1,total_batch),end='')
            randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]
            batch_train_sr = training_data_sr_mag[randidx,:]
            batch_train_lr = training_data_lr_mag[randidx,:]
            feed_train = {X_input: batch_train_lr, X_output: batch_train_sr}
            sess.run(optm_all, feed_dict=feed_train)
        epoch_using_time = time.time() - epoch_using_time_begin
        print('')
        print('epoch {0} cost {1:<.3f} sec'.format(epoch_i+1,epoch_using_time))

        ######## load validation data begin ########
        data_load_begin = time.time()
        n_sr = [dir_valid + 'validation_data_sr_mag.mat'][0]
        n_lr = [dir_valid + 'validation_data_lr_mag.mat'][0]
        block_valid_sr = sio.loadmat(n_sr)
        block_valid_lr = sio.loadmat(n_lr)
        valid_data_sr_mag = block_valid_sr['validation_data_mag']
        valid_data_lr_mag = block_valid_lr['validation_data_mag']
        print('load validation data cost %.2f sec' %(time.time() - data_load_begin))
        valid_len = valid_data_lr_mag.shape[0]
        ######## load validation data end ########
        
        valid_batch = valid_len // batch_size
        randidx_all_valid = np.random.permutation(valid_len)
        rec_SNR_mag = np.zeros(valid_batch)
        recon_begin = time.time()
        for batch_i in range(valid_batch):
            randidx_valid = randidx_all_valid[batch_i*batch_size:(batch_i+1)*batch_size]
            x_mag_sr = valid_data_sr_mag[randidx_valid,:]
            x_mag_lr = valid_data_lr_mag[randidx_valid,:]
            feed_test = {X_input: x_mag_lr, X_output: x_mag_sr}
            Prediction_value = sess.run(Prediction, feed_dict=feed_test)
            rec_SNR_mag[batch_i] = SNR(x_mag_sr, Prediction_value[-1])
        recon_using_time = time.time() - recon_begin
        SNR_mag_mean = np.mean(rec_SNR_mag)
           
        saver.save(sess, '%sSaved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
        input_text1 = "  Avg SNR in mag is "
        input_text2 = '[Epoch %d, SNR=%.4f]' % (epoch_i, SNR_mag_mean)
        output_data_recon = [input_text1 + input_text2 + '\n'][0]
        print_text = "cpkt NO. is %d, recover using time %.3f sec\n" % (epoch_i, recon_using_time)
        print('')
        print(output_data_recon)
        print(print_text)
        output_file = open(output_file_name_SNR, 'a')
        output_file.write(output_data_recon)
        output_file.close()
    print("Training Finished")
else:
    print('Begin Testing')
    test_data = os.listdir(dir_test)
    test_data_num = len(test_data)
    rec_SNR_mag = np.zeros([test_data_num])
    saver.restore(sess, '%sSaved_Model_Qista-Net-m.cpkt' % (model_dir))
    
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
            
        test_block_mag_sr = np.abs(np.transpose(x_stft_sr))
        test_block_phase_sr = np.angle(np.transpose(x_stft_sr))
        test_block_mag_lr = np.abs(np.transpose(x_stft_lr))
        
        feed_test = {X_input: test_block_mag_lr, X_output: test_block_mag_sr}
        recon_time_begin = time.time()
        Prediction_value = sess.run(Prediction[-1], feed_dict=feed_test)
        recon_using_time = time.time() - recon_time_begin
        recon_mag = Prediction_value.astype(np.float32)
        
        dir_mag = './mag_mat/'
        if not os.path.exists(dir_mag):
            os.makedirs(dir_mag)
        mdic = {'mag':recon_mag}
        name_mag = [dir_mag + test_data[wave_i] + '_recon_mag.mat'][0]
        sio.savemat(name_mag,mdic)
        
        rec_SNR_mag[wave_i] = SNR(test_block_mag_sr,recon_mag)
        print('wave %d mag recon SNR=%.3f' %(wave_i+1,rec_SNR_mag[wave_i]), end='')
        print('  cost %.2f sec' %(recon_using_time))
        
    SNR_mag_mean = rec_SNR_mag.mean()

    output_data_recon = "Avg SNR-mag is %.2f dB\n" % (SNR_mag_mean)
    print('')
    print(output_data_recon)
    output_file = open(output_file_name, 'a')
    output_file.write(output_data_recon)
    output_file.close()
    
sess.close()



