#coding:utf-8
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
num_keep_radio = 0.7


"""
Since MTCNN is a Multi-task Network,we should pay attention to the format of training data.The format is:

[path to image][cls_label][bbox_label][landmark_label]

For pos sample,cls_label=1,bbox_label(calculate),landmark_label=[0,0,0,0,0,0,0,0,0,0].

For part sample,cls_label=-1,bbox_label(calculate),landmark_label=[0,0,0,0,0,0,0,0,0,0].

For landmark sample,cls_label=-2,bbox_label=[0,0,0,0],landmark_label(calculate).

For neg sample,cls_label=0,bbox_label=[0,0,0,0],landmark_label=[0,0,0,0,0,0,0,0,0,0].

"""

def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5
    return pos + neg

def dense_to_one_hot(labels_dense,num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels)*num_classes
    #num_sample*num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
#cls_prob:batch*2
#label:batch

def cls_ohem(cls_prob, label):
    zeros = tf.zeros_like(label)
    #label=-1 --> label=0net_factory
    label_filter_invalid = tf.where(tf.less(label,0), zeros, label) # label <0 的 都标注为0
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1]) # reshape 成 N 行，每行2列
    label_int = tf.cast(label_filter_invalid,tf.int32)
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    row = tf.range(num_row)*2
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob+1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob,dtype=tf.float32)
    valid_inds = tf.where(label < zeros,zeros,ones)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    #set 0 to invalid sample
    loss = loss * valid_inds
    loss,_ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)
def bbox_ohem_smooth_L1_loss(bbox_pred,bbox_target,label):
    sigma = tf.constant(1.0)
    threshold = 1.0/(sigma**2)
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
    abs_error = tf.abs(bbox_pred-bbox_target)
    loss_smaller = 0.5*((abs_error*sigma)**2)
    loss_larger = abs_error-0.5/(sigma**2)
    smooth_loss = tf.reduce_sum(tf.where(abs_error<threshold,loss_smaller,loss_larger),axis=1)
    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
    smooth_loss = smooth_loss*valid_inds
    _, k_index = tf.nn.top_k(smooth_loss, k=keep_num)
    smooth_loss_picked = tf.gather(smooth_loss, k_index)
    return tf.reduce_mean(smooth_loss_picked)
def bbox_ohem_orginal(bbox_pred,bbox_target,label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    #pay attention :there is a bug!!!!
    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
    #(batch,)
    square_error = tf.reduce_sum(tf.square(bbox_pred-bbox_target),axis=1)
    #keep_num scalar
    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
    #keep valid index square_error
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)

#label=1 or label=-1 then do regression. 1正样本，-1：part样本
def bbox_ohem(bbox_pred,bbox_target,label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(tf.abs(label), 1),ones_index,zeros_index)
    #(batch,)
    square_error = tf.square(bbox_pred-bbox_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    #keep_num scalar
    num_valid = tf.reduce_sum(valid_inds)
    """
        下面这里被作者注释掉的部分：num_keep_radio，PPT 里有提到  在线困难样本选择
        每个batch中选取loss最大的70%进行反向传播
    """

    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    #keep valid index square_error
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)

# landmark loss TODO
# landmark_label=[0,0,0,0,0,0,0,0,0,0].
def landmark_ohem(landmark_pred,landmark_target,label):
    '''
    :param landmark_pred:
    :param landmark_target:
    :param label:
    :return: mean euclidean loss
    '''
    # landmark sample,cls_label=-2
    # keep label =-2  then do landmark detection
    ones = tf.ones_like(label,dtype=tf.float32)
    zeros = tf.zeros_like(label,dtype=tf.float32)

    '''
    tf.where
        https://www.tensorflow.org/api_docs/python/tf/where
        label, ones, zeros must have same shape.  if tf.equal(label,-2) = True, lable数组对应位置设置为ones对应位置的值也就是1，否则就是zeros对应位置的值0
    e.g. 
        import tensorflow as tf
        import numpy as np
        sess=tf.Session()
        label=np.array([0, 1, -2, 0, -2, 1])
        ones = tf.ones_like(label,dtype=tf.float32)
        zeros = tf.zeros_like(label,dtype=tf.float32)
        valid_inds = tf.where(tf.equal(label,-2),ones,zeros)
        print(sess.run(valid_inds))
        output: [0. 0. 1. 0. 1. 0.]
    '''

    '''
    tf.gather
        import tensorflow as tf

        temp = tf.range(0,10)*10 + tf.constant(1,shape=[10])
        temp2 = tf.gather(temp,[1,5,9])

        with tf.Session() as sess:

            print sess.run(temp)
            print sess.run(temp2)

        输出 
        [ 1 11 21 31 41 51 61 71 81 91] 
        [11 51 91]
    '''
    valid_inds = tf.where(tf.equal(label,-2),ones,zeros)
    square_error = tf.square(landmark_pred-landmark_target) # 两个 1 *10 的向量相减
    square_error = tf.reduce_sum(square_error,axis=1) # 以为landmark 1 * 10的向量，所以这里虽然reduce_sum,仅仅表示一个training sample
    num_valid = tf.reduce_sum(valid_inds) # e.g. [0. 0. 1. 0. 1. 0.] output is: 1+1 =2
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error*valid_inds #  e.g. valid_inds: [0. 0. 1. 0. 1. 0.]
    # based on the https://www.tensorflow.org/api_docs/python/tf/math/top_k,  tf.nn.top_k 函数 返回 values  and index 两个值
    _, k_index = tf.nn.top_k(square_error, k=keep_num) # sample square_error [....] 只有是1的位置值才比较大
    square_error = tf.gather(square_error, k_index) #square_error是一个向量，表示每个sample的square_error, 这里返回还是个向量，只不过过滤了一下，只选中了k_index的位置的square_error
    return tf.reduce_mean(square_error)
    
def cal_accuracy(cls_prob,label):
    pred = tf.argmax(cls_prob,axis=1)
    label_int = tf.cast(label,tf.int64)
    cond = tf.where(tf.greater_equal(label_int,0))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int,picked)
    pred_picked = tf.gather(pred,picked)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return accuracy_op

#construct Pnet
#label:batch
def P_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    #define common param
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005), 
                        padding='valid'):
        net = slim.conv2d(inputs, 10, 3, stride=1,scope='conv1')
        net = slim.max_pool2d(net, kernel_size=[2,2], stride=2, scope='pool1', padding='SAME')
        net = slim.conv2d(net,num_outputs=16,kernel_size=[3,3],stride=1,scope='conv2')
        net = slim.conv2d(net,num_outputs=32,kernel_size=[3,3],stride=1,scope='conv3')
        #batch*H*W*2
        conv4_1 = slim.conv2d(net,num_outputs=2,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.softmax)
        
        #batch*H*W*4
        bbox_pred = slim.conv2d(net,num_outputs=4,kernel_size=[1,1],stride=1,scope='conv4_2',activation_fn=None)
        #batch*H*W*10
        landmark_pred = slim.conv2d(net,num_outputs=10,kernel_size=[1,1],stride=1,scope='conv4_3',activation_fn=None)

        if training:
            #batch*2
            cls_prob = tf.squeeze(conv4_1,[1,2],name='cls_prob')
            cls_loss = cls_ohem(cls_prob,label)
            #batch
            bbox_pred = tf.squeeze(bbox_pred,[1,2],name='bbox_pred')
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            #batch*10
            landmark_pred = tf.squeeze(landmark_pred,[1,2],name="landmark_pred")
            landmark_loss = landmark_ohem(landmark_pred,landmark_target,label)

            accuracy = cal_accuracy(cls_prob,label)
            # forward计算出来的L2_loss, 然后返回回去，让train 那个函数 去做反向更新
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy 
        #test
        else:
            #when test,batch_size = 1
            cls_pro_test = tf.squeeze(conv4_1, axis=0)
            bbox_pred_test = tf.squeeze(bbox_pred,axis=0)
            landmark_pred_test = tf.squeeze(landmark_pred,axis=0)
            return cls_pro_test,bbox_pred_test,landmark_pred_test
        
def R_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3,3], stride=1, scope="conv1")
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        net = slim.conv2d(net,num_outputs=48,kernel_size=[3,3],stride=1,scope="conv2")
        net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope="pool2")
        net = slim.conv2d(net,num_outputs=64,kernel_size=[2,2],stride=1,scope="conv3")
        fc_flatten = slim.flatten(net)
        fc1 = slim.fully_connected(fc_flatten, num_outputs=128,scope="fc1")
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        #batch*10
        landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            landmark_loss = landmark_ohem(landmark_pred,landmark_target,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            return cls_prob,bbox_pred,landmark_pred


# slim lib: https://github.com/tensorflow/models/tree/master/research/slim
def O_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope="conv1")
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv2")
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope="conv4")
        fc_flatten = slim.flatten(net)
        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1")
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        #batch*10
        landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            return cls_prob,bbox_pred,landmark_pred
        
                                                                  
