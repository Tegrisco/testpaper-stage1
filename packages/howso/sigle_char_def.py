# uncompyle6 version 3.5.1
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.7.4 (default, Sep  7 2019, 18:27:02) 
# [Clang 10.0.1 (clang-1001.0.46.4)]
# Embedded file name: sigle_char_def.py
# Compiled at: 2019-10-25 09:34:26
# Size of source mod 2**32: 6821 bytes
"""
Created on Sun Feb 24 19:50:47 2019

@author: tbf
"""
import tensorflow.contrib.slim as slim, numpy as np, tensorflow as tf, pickle, os, pandas as pd
from PIL import Image

def init():
    global FLAGS
    tf.app.flags.DEFINE_integer('chi_charset_size', 3755, 'Choose the first `charset_size` characters only.')
    tf.app.flags.DEFINE_integer('eng_charset_size', 52, 'Choose the first `charset_size` characters only.')
    tf.app.flags.DEFINE_integer('num_charset_size', 10, 'Choose the first `charset_size` characters only.')
    tf.app.flags.DEFINE_integer('sign_charset_size', 25, 'Choose the first `charset_size` characters only.')
    tf.app.flags.DEFINE_integer('image_size', 64, 'Needs to provide same value as in training.')
    tf.app.flags.DEFINE_string('chi_checkpoint_dir', '../oral/chi_model/', 'the checkpoint dir')
    tf.app.flags.DEFINE_string('eng_checkpoint_dir', '../oral/eng_model/', 'the checkpoint dir')
    tf.app.flags.DEFINE_string('num_checkpoint_dir', '../oral/num_model/', 'the checkpoint dir')
    tf.app.flags.DEFINE_string('sign_checkpoint_dir', '../oral/sign_model/', 'the checkpoint dir')
    tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
    FLAGS = tf.app.flags.FLAGS


def vals_clr():
    for name in list(FLAGS):
        delattr(FLAGS, name)


def dict_val(chi_dict_val, eng_dict_val, num_dict_val, sign_dict_val):
    global chi_dict
    global eng_dict
    global num_dict
    global sign_dict
    chi_dict, eng_dict, num_dict, sign_dict = (
     chi_dict_val, eng_dict_val, num_dict_val, sign_dict_val)


def build_graph(top_k, type_dict):
    with tf.device('/cpu:0'):
        keep_prob = tf.placeholder(dtype=(tf.float32), shape=[], name='keep_prob')
        images = tf.placeholder(dtype=(tf.float32), shape=[None, 64, 64, 1], name='image_batch')
        labels = tf.placeholder(dtype=(tf.int64), shape=[None], name='label_batch')
        conv_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv1')
        max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding='SAME')
        conv_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv2')
        max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding='SAME')
        conv_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3')
        max_pool_3 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding='SAME')
        flatten = slim.flatten(max_pool_3)
        fc1 = slim.fully_connected((slim.dropout(flatten, keep_prob)), 1024, activation_fn=(tf.nn.tanh), scope='fc1')
        if type_dict == chi_dict:
            logits = slim.fully_connected((slim.dropout(fc1, keep_prob)), (FLAGS.chi_charset_size), activation_fn=None, scope='fc2')
        elif type_dict == eng_dict:
            logits = slim.fully_connected((slim.dropout(fc1, keep_prob)), (FLAGS.eng_charset_size), activation_fn=None, scope='fc2')
        elif type_dict == num_dict:
            logits = slim.fully_connected((slim.dropout(fc1, keep_prob)), (FLAGS.num_charset_size), activation_fn=None, scope='fc2')
        elif type_dict == sign_dict:
            logits = slim.fully_connected((slim.dropout(fc1, keep_prob)), (FLAGS.sign_charset_size), activation_fn=None, scope='fc2')
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
        global_step = tf.get_variable('step', [], initializer=(tf.constant_initializer(0.0)), trainable=False)
        rate = tf.train.exponential_decay(0.0002, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)
        train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step)
        probabilities = tf.nn.softmax(logits)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged_summary_op = tf.summary.merge_all()
        predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
        accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))
    return {'images':images, 
     'labels':labels, 
     'keep_prob':keep_prob, 
     'top_k':top_k, 
     'global_step':global_step, 
     'train_op':train_op, 
     'loss':loss, 
     'accuracy':accuracy, 
     'accuracy_top_k':accuracy_in_top_k, 
     'merged_summary_op':merged_summary_op, 
     'predicted_distribution':probabilities, 
     'predicted_index_top_k':predicted_index_top_k, 
     'predicted_val_top_k':predicted_val_top_k}


def inference(image, type_dict):
    temp_image = Image.open(image).convert('L')
    temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = temp_image.reshape([-1, 64, 64, 1])
    with tf.Session() as (sess):
        graph = build_graph(top_k=3, type_dict=type_dict)
        tf.get_variable_scope().reuse_variables()
        saver = tf.train.Saver()
        if type_dict == chi_dict:
            ckpt = tf.train.latest_checkpoint(FLAGS.chi_checkpoint_dir)
        elif type_dict == eng_dict:
            ckpt = tf.train.latest_checkpoint(FLAGS.eng_checkpoint_dir)
        elif type_dict == num_dict:
            ckpt = tf.train.latest_checkpoint(FLAGS.num_checkpoint_dir)
        elif type_dict == sign_dict:
            ckpt = tf.train.latest_checkpoint(FLAGS.sign_checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
        predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']], feed_dict={graph['images']: temp_image, graph['keep_prob']: 1.0})
    return (predict_val, predict_index)


def output(image, type_dict):
    tf.reset_default_graph()
    dict = pickle.load(open(type_dict, 'br'))
    index_to_char = {value:key for key, value in dict.items()}
    final_predict_val, final_predict_index = inference(image, type_dict)
    return index_to_char[final_predict_index[0][0]]


def recognize(type_dict, img_path, save_path):
    s1 = []
    s2 = []
    df = pd.DataFrame()
    for filename in os.listdir(img_path + '/'):
        image = img_path + '/' + filename
        index = filename[:-4].split('_')[1]
        value = output(image, type_dict)
        if len(value) == 0:
            value = 'None'
        s1.extend([index])
        s2.extend([value])

    df['index'] = s1
    df['value'] = s2
    df.set_index('index', drop=True, inplace=True)
    return df
# okay decompiling sigle_char_def.pyc
