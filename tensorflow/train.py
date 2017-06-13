import numpy as np
import h5py
import os
import scipy.io
import tensorflow as tf
import math

# Path to 3d tensor. Tensor.shape is (50,50,100)
#tensor_path is training data(rendered data)
#val_path is testing data(real data)
tensor_path = '../pc_real_bias_nolookat_remove_noise/train/'
val_path = '../pc_real_bias_nolookat_remove_noise/val/'

filelist = []
val_list = []
#print(os.listdir(tensor_path))

for dirname in os.listdir(val_path):
    for filename in os.listdir(val_path+'/'+dirname):
        val_list.append(dirname+'/'+filename)
        
for dirname in os.listdir(tensor_path):
    for filename in os.listdir(tensor_path+'/'+dirname):
        filelist.append(dirname+'/'+filename)

print('total training data are %d' %len(filelist))
print('total training data are %d' %len(val_list))
batch_size = 64
num_labels = 6

def read_data(root,path):
    f = h5py.File(root+path)
    data = np.array(f['answer'])
    f.close()
    labels = np.zeros((1,num_labels)) 
    labels[0][int(path[0])] = 1
    return data, labels

# Accuracy function
def get_accuracy(predictions, labels):
  gt = tf.argmax(labels,1)
  prediction = tf.argmax(predictions,1)
  singleacc = tf.reduce_sum(tf.cast(tf.equal(prediction, gt), tf.float32))
  accuracy = 100 * tf.reduce_mean(tf.cast(tf.equal(prediction, gt), tf.float32))
  return  gt,prediction, singleacc, accuracy


def bn(input,depth):
    batch_mean,batch_variance = tf.nn.moments(input, [0,1,2,3], name='moments')
    beta = tf.Variable(tf.constant(0.0, shape=[depth]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[depth]), name='gamma', trainable=True)
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    epsilon = 1e-3
        
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_variance])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_variance)
        
    pred = tf.placeholder(tf.bool)
    pred =  tf.Variable(True,  name='pred')
    mean, var = tf.cond(pred, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_variance)))
    BN = tf.nn.batch_normalization(input,mean,var,beta,gamma,epsilon)
    return BN

def leakyrelu(x, alpha=0., max_value = None):
    '''ReLU.
    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x

graph = tf.Graph()

with graph.as_default():

    predict = tf.Variable(False)
    # Input data.
    with tf.name_scope('data') as scope:
        tf_train_dataset = tf.placeholder(tf.float32, shape=(None, 50, 50, 100, 1), name = "tf_train_dataset")
        tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels), name = "tf_train_labels")

    # Variables.
    with tf.name_scope('conv1') as scope:
	#conv1
        stdv = 1/math.sqrt(5*5*5*1)
        conv1_w = tf.Variable(tf.random_uniform([5, 5, 5, 1, 32], -stdv, stdv ), name = "conv1_w")
        conv1_bias = tf.Variable(tf.random_uniform([32], -stdv, stdv ), name = "conv1_bias")

    with tf.name_scope('conv2') as scope:
	#conv2
        stdv = 1/math.sqrt(3*3*3*32)
        conv2_w = tf.Variable(tf.random_uniform([3, 3, 3, 32, 32], -stdv, stdv ), name = "conv2_w")   
        conv2_bias = tf.Variable(tf.random_uniform([32], -stdv, stdv ), name = "conv2_bias")
        
    with tf.name_scope('fc1') as scope:
    #fc1        
        stdv = 1/math.sqrt(11*11*23*32)
        fc1_w = tf.Variable(tf.random_uniform([11*11*23*32, 128], -stdv, stdv), name = "fc1_w")
        fc1_bias = tf.Variable(tf.random_uniform([128]), name = "fc1_bias")    
        
    with tf.name_scope('fc2') as scope:
	#fc2
        stdv = 1/math.sqrt(128)
        fc2_w = tf.Variable(tf.random_uniform([128, num_labels], -stdv, stdv), name = "fc2_w")
        fc2_bias = tf.Variable(tf.random_uniform([num_labels]), name = "fc2_bias")
        
    #MODEL     
    def model(data):
        # Conv1
        padding = [[0,0],[1,1],[1,1],[1,1],[0,0]]
        padded_input = tf.pad(data,padding,"CONSTANT")
        conv1 = tf.nn.conv3d(padded_input, conv1_w, [1, 2, 2, 2, 1], padding='VALID')
        
        #BN1
        BN1 = bn(conv1,32)
        
        hidden1 = leakyrelu(tf.nn.bias_add(BN1,conv1_bias),0.1)                
        
        dropout1 = tf.nn.dropout(hidden1, 0.8)   #keep probability
                
        # Conv2
        conv2 = tf.nn.conv3d(dropout1, conv2_w, [1, 1, 1, 1, 1],padding='VALID')
        hidden2 = leakyrelu(tf.nn.bias_add(conv2,conv2_bias),0.1)
        #BN2
        BN2 = bn(hidden2,32)
		
        
	#Pool1
        pool1 = tf.nn.max_pool3d(BN2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID')
        dropout2 = tf.nn.dropout(pool1, 0.7)        
        normalize3_flat = tf.reshape(dropout2, [-1, 11*11*23*32])
        
        #FC1
        fc1 = tf.add(tf.matmul(normalize3_flat, fc1_w) , fc1_bias)
        hidden3 = tf.nn.relu(fc1)
        dropout3 = tf.nn.dropout(hidden3, 0.6)

        #FC2                
        res = tf.add(tf.matmul(dropout3, fc2_w) , fc2_bias )
        return res

    
    # Training computation
    local_res = model(tf_train_dataset)#

    with tf.name_scope("cost_function") as scope:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = local_res))
    tf.scalar_summary("cost_function", cross_entropy)
        
    # Optimizer
    train_step = tf.train.MomentumOptimizer(0.01, 0.9).minimize(cross_entropy)
    # Predictions for the training, validation, and test data
    with tf.name_scope("accuracy") as scope:
        gt, prediction, singleacc ,accuracy = get_accuracy(local_res, tf_train_labels)
    tf.scalar_summary("accuracy", accuracy)


    valid_prediction = tf.nn.softmax(model(tf_train_dataset))
    print ('Graph was built')
    
    merged_summary_op = tf.merge_all_summaries()      

# Session
epochs = 3
#each epochs will see all images
steps_per_epoch = int(len(filelist)/batch_size) + 1
if (len(filelist) % batch_size) == 0:
    steps_per_epoch -= 1
#print ('STEPS %d' % steps_per_epoch)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
with tf.Session(graph=graph,config=tf.ConfigProto(gpu_options=gpu_options)) as session:    
    train_writer = tf.train.SummaryWriter('./', session.graph)
    saver = tf.train.Saver(tf.all_variables())
    session.run(tf.initialize_all_variables())
    
    
    
    for epch in xrange(0, epochs):
        np.random.shuffle(filelist)    #shuffle filelist
        print ('EPOCH %d' % epch)
        total_prediction = 0
        for step in range(steps_per_epoch):
            offset = (step * batch_size) % len(filelist)
            
            if (len(filelist) % batch_size) == 0:
                offset = (step * batch_size) % (len(filelist) +1)
                
            # Generate a minibatch.            
            datas = np.zeros((1,50,50,100))
            labels = np.zeros((1,num_labels));
            i = 0;
            while (offset+i) < len(filelist) and (offset+i) < offset + batch_size:
                pc, singlelabel = read_data(tensor_path,filelist[offset+i])
                datas  = np.concatenate((datas,pc[np.newaxis,...]),axis = 0) 
                labels = np.concatenate((labels,singlelabel),axis = 0)
                
                i += 1
            if (offset+i) < offset + batch_size:
                left = batch_size - i
                offset = 0
                for i in range(left):
                    pc, singlelabel = read_data(tensor_path,filelist[offset+i])
                    datas  = np.concatenate((datas,pc[np.newaxis,...]),axis = 0)
                    labels = np.concatenate((labels,singlelabel),axis = 0)  
                
            datas = datas[1:datas.shape[0],:,:,:]
            batch_labels = labels[1:labels.shape[0],:]     #delete first labels
            batch_data = datas.astype('float32')
            batch_data = batch_data[:,:,:,:,None]
            rawoutput,_,train_labels, train_accuracy, train_prediction,_ ,singleaccuracy, summary_str = session.run([local_res,
                                                cross_entropy,gt, 
                                                 accuracy, prediction, train_step, singleacc, merged_summary_op],
                                                feed_dict={tf_train_dataset: batch_data, 
                                                           tf_train_labels: batch_labels})
                        
            total_prediction = int(total_prediction) + int(singleaccuracy)
            train_writer.add_summary(summary_str, step)
	    print("Step %d accuracy: %.1f%%" % (step, train_accuracy))
            
        total_accuracy = 100*total_prediction/ (batch_size * (step + 1))
        print("Epoch %d accuracy: %.1f%%" % (epch, total_accuracy))
        
        #val
        val_pred = 0  
        confusion_matrix = np.zeros((num_labels,num_labels))
        for j in range(len(val_list)):
            pc, singlelabel = read_data(val_path,val_list[j])
            pc = pc[None,:,:,:,None].astype('float32')
            val_prediction, val_label, val_acc,_ = session.run([prediction,gt,singleacc, merged_summary_op],
                                                 feed_dict={tf_train_dataset: pc, tf_train_labels: singlelabel})

            for con_index in range(len(val_prediction)):
                confusion_matrix[val_label[con_index]][val_prediction[con_index]] += 1 
            
            val_pred = val_pred + val_acc
        val_accuracy = 100*val_pred/len(val_list)
        print(confusion_matrix)
        print("Epoch %d val accuracy: %.1f%%" % (epch, val_accuracy))
        
        saver.save(session, 'test.ckpt')
            
            
 
