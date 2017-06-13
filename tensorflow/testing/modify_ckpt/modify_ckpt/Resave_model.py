import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

with tf.Session() as sess:
    checkpoint_path = "./itrichess_withbn.ckpt"
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    print(var_to_shape_map)

    param = {}
    for key in var_to_shape_map:
        try:
            #print(key)
            W = reader.get_tensor(key)            
            v1 = tf.Variable(W, name="v1")
            if key[0:5] == "cond_":
                param[key[7:]] = v1                
            elif key[0:4] == "cond":
                param[key[5:]] = v1
            else:
                param[key] = v1
        except:
            print ("[DEBUG] Failed to get_tensor from key : ",key)

    #init_ops = tf.global_variables_initializer()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(param)
    save_path = saver.save(sess, "./newModel.ckpt")
