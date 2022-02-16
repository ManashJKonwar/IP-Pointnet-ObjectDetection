__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__reference__ = "https://github.com/charlesq34/frustum-pointnets/blob/master/train/train.py"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import os
import sys
import shutil
import argparse
import importlib
import numpy as np
import tensorflow as tf
from datetime import datetime
from KITTI_visualize_frustum_data import FrustumDataset, compute_box3d_iou
from utility.KITTI.train_util import get_batch

def log_string(logger_file, out_str):
    logger_file.write(out_str+'\n')
    logger_file.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.compat.v1.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    ''' Main function for training and simple evaluation. '''
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = \
                MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)

            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.compat.v1.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and losses 
            end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
                is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(labels_pl, centers_pl,
                heading_class_label_pl, heading_residual_label_pl,
                size_class_label_pl, size_residual_label_pl, end_points)
            tf.summary.scalar('loss', loss)

            losses = tf.compat.v1.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)

            # Write summaries of bounding box IoU and segmentation accuracies
            iou2ds, iou3ds = tf.py_function(compute_box3d_iou, [\
                end_points['center'], \
                end_points['heading_scores'], end_points['heading_residuals'], \
                end_points['size_scores'], end_points['size_residuals'], \
                centers_pl, \
                heading_class_label_pl, heading_residual_label_pl, \
                size_class_label_pl, size_residual_label_pl], \
                [tf.float32, tf.float32])
            end_points['iou2ds'] = iou2ds 
            end_points['iou3ds'] = iou3ds 
            tf.summary.scalar('iou_2d', tf.reduce_mean(iou2ds))
            tf.summary.scalar('iou_3d', tf.reduce_mean(iou3ds))

            correct = tf.equal(tf.argmax(end_points['mask_logits'], 2), tf.cast(labels_pl, tf.int64))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / \
                float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('segmentation accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate,
                    momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        if FLAGS.restore_model_path is None:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, FLAGS.restore_model_path)

        ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'logits': end_points['mask_logits'],
               'centers_pred': end_points['center'],
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer):
    ''' Training for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops
    '''
    is_training = True
    log_string(str(datetime.now()))
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)/BATCH_SIZE

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0

    # Training with batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = \
            get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['one_hot_vec_pl']: batch_one_hot_vec,
                     ops['labels_pl']: batch_label,
                     ops['centers_pl']: batch_center,
                     ops['heading_class_label_pl']: batch_hclass,
                     ops['heading_residual_label_pl']: batch_hres,
                     ops['size_class_label_pl']: batch_sclass,
                     ops['size_residual_label_pl']: batch_sres,
                     ops['is_training_pl']: is_training,}

        summary, step, _, loss_val, logits_val, centers_pred_val, \
        iou2ds, iou3ds = \
            sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'],
                ops['logits'], ops['centers_pred'],
                ops['end_points']['iou2ds'], ops['end_points']['iou3ds']], 
                feed_dict=feed_dict)

        train_writer.add_summary(summary, step)

        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)
        iou3d_correct_cnt += np.sum(iou3ds>=0.7)

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('segmentation accuracy: %f' % \
                (total_correct / float(total_seen)))
            log_string('box IoU (ground/3D): %f / %f' % \
                (iou2ds_sum / float(BATCH_SIZE*10), iou3ds_sum / float(BATCH_SIZE*10)))
            log_string('box estimation accuracy (IoU=0.7): %f' % \
                (float(iou3d_correct_cnt)/float(BATCH_SIZE*10)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            iou2ds_sum = 0
            iou3ds_sum = 0
            iou3d_correct_cnt = 0
        
        
def eval_one_epoch(sess, ops, test_writer):
    ''' Simple evaluation for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops """
    '''
    global EPOCH_CNT
    is_training = False
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = len(TEST_DATASET)/BATCH_SIZE

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0
   
    # Simple evaluation with batches 
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = \
            get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['one_hot_vec_pl']: batch_one_hot_vec,
                     ops['labels_pl']: batch_label,
                     ops['centers_pl']: batch_center,
                     ops['heading_class_label_pl']: batch_hclass,
                     ops['heading_residual_label_pl']: batch_hres,
                     ops['size_class_label_pl']: batch_sclass,
                     ops['size_residual_label_pl']: batch_sres,
                     ops['is_training_pl']: is_training}

        summary, step, loss_val, logits_val, iou2ds, iou3ds = \
            sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['logits'], 
                ops['end_points']['iou2ds'], ops['end_points']['iou3ds']],
                feed_dict=feed_dict)
        test_writer.add_summary(summary, step)

        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(batch_label==l)
            total_correct_class[l] += (np.sum((preds_val==l) & (batch_label==l)))
        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)
        iou3d_correct_cnt += np.sum(iou3ds>=0.7)

        for i in range(BATCH_SIZE):
            segp = preds_val[i,:]
            segl = batch_label[i,:] 
            part_ious = [0.0 for _ in range(NUM_CLASSES)]
            for l in range(NUM_CLASSES):
                if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0): 
                    part_ious[l] = 1.0 # class not present
                else:
                    part_ious[l] = np.sum((segl==l) & (segp==l)) / \
                        float(np.sum((segl==l) | (segp==l)))

    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval segmentation accuracy: %f'% \
        (total_correct / float(total_seen)))
    log_string('eval segmentation avg class acc: %f' % \
        (np.mean(np.array(total_correct_class) / \
            np.array(total_seen_class,dtype=np.float))))
    log_string('eval box IoU (ground/3D): %f / %f' % \
        (iou2ds_sum / float(num_batches*BATCH_SIZE), iou3ds_sum / \
            float(num_batches*BATCH_SIZE)))
    log_string('eval box estimation accuracy (IoU=0.7): %f' % \
        (float(iou3d_correct_cnt)/float(num_batches*BATCH_SIZE)))
         
    EPOCH_CNT += 1

def train_frustum_pointnet_tf(**kwargs):
    """
    This method trains the frustum pointnet using tensorflow
    Parameters: 
        gpu (int): GPU to use [default: GPU 0]
        model (str): Model name [default: frustum_pointnets_v1]
        log_dir (str): Log dir [default: log]
        num_point (int): Point Number [default: 2048]
        max_epoch (int): Epoch to run [default: 201]
        batch_size (int): Batch Size during training [default: 32]
        learning_rate (float): Initial learning rate [default: 0.001]
        momentum (float): Initial learning rate [default: 0.9]
        optimizer (str): adam or momentum [default: adam]
        decay_step (int): Decay step for lr decay [default: 200000]
        decay_rate (float): Decay rate for lr decay [default: 0.7]
        no_intensity (bool): Only use XYZ for training
        restore_model_path (str): Restore model path e.g. log/model.ckpt [default: None]
    Returns: 
        None
    """
    gpu = kwargs.get('gpu') if 'gpu' in kwargs.keys() else 0
    model_name = kwargs.get('model_name') if 'model_name' in kwargs.keys() else 'frustum_pointnets_v1'
    log_dir = kwargs.get('log_dir') if 'log_dir' in kwargs.keys() else r'logs\KITTI'
    num_point = kwargs.get('num_point') if 'num_point' in kwargs.keys() else 2048
    max_epoch = kwargs.get('max_epoch') if 'max_epoch' in kwargs.keys() else 201
    batch_size = kwargs.get('batch_size') if 'batch_size' in kwargs.keys() else 32
    learning_rate = kwargs.get('learning_rate') if 'learning_rate' in kwargs.keys() else 0.001
    momentum = kwargs.get('momentum') if 'momentum' in kwargs.keys() else 0.9
    optimizer = kwargs.get('optimizer') if 'optimizer' in kwargs.keys() else 'adam'
    decay_step = kwargs.get('decay_step') if 'decay_step' in kwargs.keys() else 200000
    decay_rate = kwargs.get('decay_rate') if 'decay_rate' in kwargs.keys() else 0.7
    no_intensity = kwargs.get('no_intensity') if 'no_intensity' in kwargs.keys() else False
    restore_model_path = kwargs.get('restore_model_path') if 'restore_model_path' in kwargs.keys() else None

    # Set training configurations
    global EPOCH_CNT, BATCH_SIZE, NUM_POINT, MAX_EPOCH, BASE_LEARNING_RATE, GPU_INDEX, MOMENTUM, \
        OPTIMIZER, DECAY_STEP, DECAY_RATE, NUM_CHANNEL, NUM_CLASSES, MODEL, MODEL_FILE, LOG_FOUT, \
        BN_INIT_DECAY, BN_DECAY_DECAY_RATE, BN_DECAY_DECAY_STEP, BN_DECAY_CLIP
    EPOCH_CNT = 0
    BATCH_SIZE = batch_size
    NUM_POINT = num_point
    MAX_EPOCH = max_epoch
    BASE_LEARNING_RATE = learning_rate
    GPU_INDEX = gpu
    MOMENTUM = momentum
    OPTIMIZER = optimizer
    DECAY_STEP = decay_step
    DECAY_RATE = decay_rate
    NUM_CHANNEL = 3 if no_intensity else 4 # point feature channel
    NUM_CLASSES = 2 # segmentation has two classes

    MODEL = importlib.import_module('models.KITTI.%s' %(model_name)) # import network module
    MODEL_FILE = os.path.join(r'models\KITTI', model_name+'.py')
    LOG_DIR = log_dir
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    shutil.copy2(MODEL_FILE, LOG_DIR) # Backup of Model File
    shutil.copy2(r'training\KITTI\train_tf.py', LOG_DIR)
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(kwargs)+'\n')

    BN_INIT_DECAY = 0.5
    BN_DECAY_DECAY_RATE = 0.5
    BN_DECAY_DECAY_STEP = float(DECAY_STEP)
    BN_DECAY_CLIP = 0.99

    # Load Frustum Datasets. Use default data paths.
    TRAIN_DATASET = FrustumDataset(npoints=NUM_POINT, 
                                split='train',
                                rotate_to_center=True, 
                                random_flip=True, 
                                random_shift=True, 
                                one_hot=True)
    TEST_DATASET = FrustumDataset(npoints=NUM_POINT, 
                                split='val',
                                rotate_to_center=True, 
                                one_hot=True)
    
    log_string(logger_file=LOG_FOUT, out_str='pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()