# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 20:33:35 2018

@author: Suraj
"""

import tensorflow as tf
import os
import hiding_net
import revealing_net
import discriminators
import preparation
import data
import cv2
import numpy as np
import pickle
import data
from time import time
import matplotlib.pyplot as plt
import random


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

test_loc = 'ucf/test_all/'
train_loc = 'ucf/train/'


def get_test_data():

    # dirs = os.listdir(test_loc)
    # dirs = sorted(dirs, key=lambda x: int(x))
    # cov = []
    # sec = []
    # for i in range(len(dirs)):
    #     for j in range(len(dirs)):
    #         if i != j:
    #             cov.append(test_loc + dirs[i])
    #             sec.append(test_loc + dirs[j])

    dirs = [test_loc + dir_name for dir_name in os.listdir(test_loc)]
    dirs.extend([train_loc + dir_name for dir_name in os.listdir(train_loc)])
    indices = np.random.choice(len(dirs), 75)
    covers = []
    secrets = []
    for i in range(len(indices)):
        covers.append(dirs[indices[i]])
        index = np.random.choice(len(dirs))
        while dirs[index] == covers[i]:
            index = np.random.choice(len(dirs))
        secrets.append(dirs[index])

    # frames = []
    # for i in range(len(cov)):
    #     n1 = len(os.listdir(cov[i]))
    #     n2 = len(os.listdir(sec[i]))
    #     frames.append(np.min(n1, n2))

    # cover_tensor_data = []
    # secret_tensor_data = []
    # for i in range(len(cov)):
    #     frs = sorted(os.listdir(cov[i]), key=lambda x: int(x.split('.')[0]))
    #     for j in range(frames[i]):
    #         cover_tensor_data.append(cov[i] + '/' + frs[j])
    #         secret_tensor_data.append(sec[i] + '/' + frs[j])

    # return cover_tensor_data, secret_tensor_data
    return covers, secrets

def get_train_data():

    dirs = os.listdir(train_loc)
    dirs = sorted(dirs, key=lambda x: int(x))
    dat = []
    for i in range(len(dirs)):
        dat.append(train_loc + dirs[i])

    return dat

class SingleSizeModel():
    # def get_noise_layer_op(self,tensor,std=.1):
    #     with tf.variable_scope("noise_layer"):
    #         return tensor + tf.random_normal(shape=tf.shape(tensor), mean=0.0, stddev=std, dtype=tf.float32) 

    def __init__(self, beta, log_path, input_shape=(None,8,240,320,3), input_shape1=(None,8,240,320,3)):
        
        self.checkpoint_dir = 'checkpoints_new'
        # self.model_name = 'stegnet_with_disc'
        self.model_name = 'stegnet'
        self.dataset_name = 'ucf'
        self.test_dir_all = 'test_for_video'
        self.log_dir = 'logs'
        self.img_height = 240
        self.img_width = 320
        self.channels = 3
        self.frames_per_batch = 8

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.test_dir_all):
            os.makedirs(self.test_dir_all)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # self.cover_tensor_data_test, self.secret_tensor_data_test = get_test_data()
        # pickle.dump(self.cover_tensor_data_test, open('cov_data_vid.pkl', 'wb'))
        # pickle.dump(self.secret_tensor_data_test, open('sec_data_vid.pkl', 'wb'))
        self.cover_tensor_data_test = pickle.load(open('cov_data_vid.pkl', 'rb'))
        self.secret_tensor_data_test = pickle.load(open('sec_data_vid.pkl', 'rb'))
        self.tensor_data_train = get_train_data()

        self.beta = beta
        self.learning_rate = 0.0001
        self.sess = tf.InteractiveSession()
        
        self.secret_tensor = tf.placeholder(shape=input_shape1,dtype=tf.float32,name="input_secret")
        self.cover_tensor = tf.placeholder(shape=input_shape,dtype=tf.float32,name="input_cover")
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

        self.train_op , self.summary_op, self.loss_op,self.secret_loss_op,self.cover_loss_op = self.prepare_training_graph(self.secret_tensor,self.cover_tensor,self.global_step_tensor)
        # self.train_op , self.summary_op, self.loss_op,self.secret_loss_op,self.cover_loss_op, self.disc_loss_cc, self.disc_loss_sr = self.prepare_training_graph(self.secret_tensor,self.cover_tensor,self.global_step_tensor)

        self.writer = tf.summary.FileWriter(self.log_dir,self.sess.graph)

        # self.hiding_output, self.reveal_output, self.summary_op, self.loss_op,self.secret_loss_op,self.cover_loss_op, self.cover_acc, self.secret_acc = self.prepare_test_graph(self.cover_tensor, self.secret_tensor)
        self.hiding_output, self.reveal_output, self.summary_op, self.loss_op,self.secret_loss_op,self.cover_loss_op = self.prepare_test_graph(self.cover_tensor, self.secret_tensor)

        self.sess.run(tf.global_variables_initializer())
        
        print("OK")


    def prepare_training_graph(self,secret_tensor,cover_tensor,global_step_tensor):
    
        # prep_secret = preparation.prep_network(secret_tensor)
        prep_cover = cover_tensor  #network.cover_prep_network(cover_tensor)
        prep_secret = secret_tensor
        hiding_output = hiding_net.hiding_network(prep_cover, prep_secret)
        reveal_output = revealing_net.revealing_network(hiding_output)
        #noise_add_op = self.get_noise_layer_op(hiding_output_op)

        # loss_op,secret_loss_op,cover_loss_op, cover_acc, secret_acc = self.get_loss_op(secret_tensor,reveal_output,cover_tensor,hiding_output,beta=self.beta)
        loss_op,secret_loss_op,cover_loss_op = self.get_loss_op(secret_tensor,reveal_output,cover_tensor,hiding_output,beta=self.beta)
    
        minimize_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op, global_step=global_step_tensor)

        print(hiding_output.shape)
        print(reveal_output.shape)
        print(secret_tensor.shape)
        print(cover_tensor.shape)

        tf.summary.scalar('loss', loss_op,family='train')
        tf.summary.scalar('reveal_net_loss', secret_loss_op,family='train')
        tf.summary.scalar('cover_net_loss', cover_loss_op,family='train')
        # tf.summary.scalar('secret_acc', secret_acc,family='train')
        # tf.summary.scalar('cover_acc', cover_acc,family='train')



        # tf.summary.image('secret',secret_tensor[:,:,3]*255,max_outputs=1,family='train')
        # tf.summary.image('cover',cover_tensor[:,:,3]*255,max_outputs=1,family='train')
        # tf.summary.image('hidden',hiding_output[:,:,3]*255,max_outputs=1,family='train')
        # #tf.summary.image('hidden_noisy',self.get_tensor_to_img_op(noise_add_op),max_outputs=1,family='train')
        # tf.summary.image('revealed',reveal_output[:,:,3]*255,max_outputs=1,family='train')

        merged_summary_op = tf.summary.merge_all()

        # return minimize_op,merged_summary_op, loss_op,secret_loss_op,cover_loss_op, cover_acc, secret_acc
        return minimize_op,merged_summary_op, loss_op,secret_loss_op,cover_loss_op



    # def prepare_training_graph(self,secret_tensor,cover_tensor,global_step_tensor):
    
    #     # prep_secret = preparation.prep_network(secret_tensor)
    #     prep_cover = cover_tensor  #network.cover_prep_network(cover_tensor)
    #     prep_secret = secret_tensor
    #     hiding_output = hiding_net.hiding_network(prep_cover, prep_secret)
    #     reveal_output = revealing_net.revealing_network(hiding_output)
    #     discriminator_cover_logits = discriminators.disc_container_cover_network(prep_cover)
    #     discriminator_container_logits = discriminators.disc_container_cover_network(hiding_output, reuse=True)
    #     discriminator_secret_logits = discriminators.disc_secret_revealed_network(prep_secret)
    #     discriminator_revealed_logits = discriminators.disc_secret_revealed_network(reveal_output, reuse=True)
    #     #noise_add_op = self.get_noise_layer_op(hiding_output_op)

    #     # loss_op,secret_loss_op,cover_loss_op, cover_acc, secret_acc = self.get_loss_op(secret_tensor,reveal_output,cover_tensor,hiding_output,beta=self.beta)
    #     loss_op,secret_loss_op,cover_loss_op = self.get_loss_op(secret_tensor,reveal_output,cover_tensor,hiding_output,beta=self.beta)
    #     disc_loss_cc = discriminators.discriminator_loss(discriminator_cover_logits, discriminator_container_logits)
    #     disc_loss_sr = discriminators.discriminator_loss(discriminator_secret_logits, discriminator_revealed_logits)

    #     # t_vars = tf.trainable_variables()
    #     # d_vars = [var for var in t_vars if 'disc' in var.name]

    #     # minimize_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op + disc_loss_cc + disc_loss_sr, var_list=d_vars, global_step=global_step_tensor)
    #     minimize_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op + disc_loss_cc + disc_loss_sr, global_step=global_step_tensor)

    #     tf.summary.scalar('loss', loss_op,family='train')
    #     tf.summary.scalar('reveal_net_loss', secret_loss_op,family='train')
    #     tf.summary.scalar('cover_net_loss', cover_loss_op,family='train')
    #     tf.summary.scalar('disc_con_cov_loss', disc_loss_cc,family='train')
    #     tf.summary.scalar('disc_sec_rev_loss', disc_loss_sr,family='train')
    #     # tf.summary.scalar('secret_acc', secret_acc,family='train')
    #     # tf.summary.scalar('cover_acc', cover_acc,family='train')



    #     # tf.summary.image('secret',secret_tensor[:,:,3]*255,max_outputs=1,family='train')
    #     # tf.summary.image('cover',cover_tensor[:,:,3]*255,max_outputs=1,family='train')
    #     # tf.summary.image('hidden',hiding_output[:,:,3]*255,max_outputs=1,family='train')
    #     # #tf.summary.image('hidden_noisy',self.get_tensor_to_img_op(noise_add_op),max_outputs=1,family='train')
    #     # tf.summary.image('revealed',reveal_output[:,:,3]*255,max_outputs=1,family='train')

    #     merged_summary_op = tf.summary.merge_all()

    #     # return minimize_op,merged_summary_op, loss_op,secret_loss_op,cover_loss_op, cover_acc, secret_acc
    #     return minimize_op,merged_summary_op, loss_op,secret_loss_op,cover_loss_op, disc_loss_cc, disc_loss_sr
    
    def prepare_test_graph(self, cover_tensor, secret_tensor):
        with tf.variable_scope("",reuse=True):

            # Image_Data_Class = ImageData(self.img_width, self.img_height, self.channels)
            # cover_tensor = tf.data.Dataset.from_tensor_slices(self.cover_tensor_data_test)
            # secret_tensor = tf.data.Dataset.from_tensor_slices(self.secret_tensor_data_test)

            # gpu_device = '/gpu:0'
            # cover_tensor = cover_tensor.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))
            # secret_tensor = secret_tensor.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))

            # inputs_iterator = inputs.make_one_shot_iterator()

            # self.inputs = inputs_iterator.get_next()

            # prep_secret = preparation.prep_network(secret_tensor)
            prep_secret = secret_tensor
            prep_cover = cover_tensor  #network.cover_prep_network(cover_tensor)
            hiding_output = hiding_net.hiding_network(prep_cover, prep_secret)
            reveal_output = revealing_net.revealing_network(hiding_output)


            # loss_op,secret_loss_op,cover_loss_op, cover_acc, secret_acc = self.get_loss_op(secret_tensor,reveal_output,cover_tensor,hiding_output)
            loss_op,secret_loss_op,cover_loss_op = self.get_loss_op(secret_tensor,reveal_output,cover_tensor,hiding_output)
            
        
            tf.summary.scalar('loss', loss_op,family='test')
            tf.summary.scalar('reveal_net_loss', secret_loss_op,family='test')
            tf.summary.scalar('cover_net_loss', cover_loss_op,family='test')
            # tf.summary.scalar('secret_acc', secret_acc,family='test')
            # tf.summary.scalar('cover_acc', cover_acc,family='test')
            # tf.summary.image('secret',secret_tensor[:,:,3]*255,max_outputs=1,family='test')
            # tf.summary.image('cover',cover_tensor[:,:,3]*255,max_outputs=1,family='test')
            # tf.summary.image('hidden',hiding_output[:,:,3]*255,max_outputs=1,family='test')
            # tf.summary.image('revealed',reveal_output[:,:,3]*255,max_outputs=1,family='test')

            merged_summary_op = tf.summary.merge_all()


            # return hiding_output, reveal_output, merged_summary_op, loss_op, secret_loss_op, cover_loss_op, cover_acc, secret_acc
            return hiding_output, reveal_output, merged_summary_op, loss_op, secret_loss_op, cover_loss_op

    def get_loss_op(self,secret_true,secret_pred,cover_true,cover_pred,beta=1.0):

        with tf.variable_scope("losses"):

            beta = tf.constant(beta, name="beta")
            secret_mse = tf.losses.mean_squared_error(secret_true,secret_pred)
            cover_mse = tf.losses.mean_squared_error(cover_true,cover_pred)

            final_loss = cover_mse + beta * secret_mse

            # secret_acc= tf.equal(tf.argmax(secret_true,1),tf.argmax(secret_pred,1))
            # secret_acc = tf.reduce_mean(tf.cast(secret_acc, tf.float32))

            # cover_acc= tf.equal(tf.argmax(cover_true,1),tf.argmax(cover_pred,1))
            # cover_acc = tf.reduce_mean(tf.cast(cover_acc, tf.float32))

            # return final_loss , secret_mse , cover_mse, cover_acc, secret_acc
            return final_loss , secret_mse , cover_mse


    # def make_chkp(self,saver, path):
    #     global_step = self.sess.run(self.global_step_tensor)
    #     saver.save(self.sess,path,global_step)

    # def load_chkp(self,saver, path):
    #     print("LOADING")
    #     global_step = self.sess.run(self.global_step_tensor)
    #     tf.reset_default_graph()
    #     imported_meta = tf.train.import_meta_graph("./model8_beta0.75/my-model.ckpt-45174.meta")
    #     imported_meta.restore(self.sess, tf.train.latest_checkpoint('./model8_beta0.75/'))
    #     #saver.restore(self.sess, path)
    #     print("LOADED")
        
       
    def train(self):

        epochs = 10000
        
        self.saver = tf.train.Saver(max_to_keep=2)

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            count = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            count = 0
            print(" [!] Load failed...")


        # # For saving generator weights.
        # t_vars = tf.trainable_variables()
        # weights = self.sess.run(t_vars)
        # t_weights_file = open('initial_weights_4100.npy','wb')
        # np.save(t_weights_file, weights)
        # print('saved')
        # input()

        # initial_weights_4100 = np.load('initial_weights_4100.npy')
        # t_vars = tf.trainable_variables()
        # prev_vars = [var for var in t_vars if 'disc' not in var.name]
        # for i in range(len(prev_vars)):
        #     t_vars[i].assign(initial_weights_4100[i])
        # print ('assigned weights successfully.')

        def load_t(base_dir, frame_names, ind):

            fpb = []
            for i in range(self.frames_per_batch):
                frame = base_dir + '/' + frame_names[ind * self.frames_per_batch + i]
                frame = cv2.imread(frame)
                frame = frame / 255.0
                frame = frame.reshape((240,320,3))
                fpb.append(frame)

            return np.reshape(np.array(fpb), (1, self.frames_per_batch, 240, 320, 3))

        def generator():
            for i in range(count, epochs):
                c_path = random.choice(self.tensor_data_train)
                s_path = random.choice(self.tensor_data_train)
                while s_path == c_path:
                    s_path = random.choice(self.tensor_data_train)
                n1 = len(os.listdir(c_path))
                n2 = len(os.listdir(s_path))
                t = int(min(n1, n2) / self.frames_per_batch)
                frs = sorted(os.listdir(c_path), key=lambda x: int(x.split('.')[0]))[:t * self.frames_per_batch]
                for j in range(t):
                    cov_tens = load_t(c_path, frs, j)
                    sec_tens = load_t(s_path, frs, j)
                    yield cov_tens, sec_tens, c_path.split('/')[-1], s_path.split('/')[-1], i, j


        for covers, secrets, c_name, s_name, epoch, batch in generator():
            
            self.sess.run([self.train_op],feed_dict={"input_secret:0":secrets,"input_cover:0":covers})
            summaree, gs, tl, sl, cl, dcl, dsl = self.sess.run([self.summary_op, self.global_step_tensor, self.loss_op,
                                                    self.secret_loss_op,self.cover_loss_op, self.disc_loss_cc, self.disc_loss_sr],
                                                    feed_dict={"input_secret:0":secrets,"input_cover:0":covers})
            self.writer.add_summary(summaree,gs)

            print('\nEpoch: '+str(epoch)+' Batch: '+str(batch)+' Loss: '+str(tl)+' Cover_Loss: '+str(cl)+' Secret_Loss: '+str(sl)+' Disc_cc_Loss: '+str(dcl)+' Disc_sr_Loss: '+str(dsl))

            if np.mod(epoch + 1, 100) == 0: 
                self.save(self.checkpoint_dir, epoch)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}".format(
            self.model_name, self.dataset_name, self.img_height, self.img_width, self.beta)

    @property
    def test_dir(self):
        return "{}_{}_{}_{}_{}_test".format(
            self.model_name, self.dataset_name, self.img_height, self.img_width, self.beta)    


    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    # def test(self, test_cover, test_secret, c_name, s_name):

    #     # self.load_chkp(saver, path)

    #     self.saver = tf.train.Saver(max_to_keep=2)

    #     could_load, checkpoint_counter = self.load(self.checkpoint_dir)

    #     if could_load:
    #         # start_epoch = (int)(checkpoint_counter / self.iteration)
    #         # start_batch_id = checkpoint_counter - start_epoch * self.iteration
    #         count = checkpoint_counter
    #         print(" [*] Load SUCCESS")
    #     else:
    #         # start_epoch = 0
    #         # start_batch_id = 0
    #         count = 1
    #         print(" [!] Load failed...")
    #         exit()

    #     print ('c_name: ' + c_name + ' s_name: ' + s_name)

    #     batch_size = 1
    #     Num = test_cover.shape[0]*test_cover.shape[1]
    #     num_of_batches = test_cover.shape[0] // batch_size
    #     cover_loss = 0
    #     secret_loss = 0
    #     cover_accuracy = 0 
    #     secret_accuracy = 0 

    #     test_dir = os.path.join(self.test_dir_all, self.test_dir)
    #     video_dir = os.path.join(test_dir, 'c_'+c_name+'_s_'+s_name)
    #     cover_dir = os.path.join(video_dir, 'cover')
    #     container_dir = os.path.join(video_dir, 'container')
    #     secret_dir = os.path.join(video_dir, 'secret')
    #     revealed_secret_dir = os.path.join(video_dir, 'revealed_secret')

    #     if not os.path.exists(test_dir):
    #         os.makedirs(test_dir)
    #     if not os.path.exists(video_dir):
    #         os.makedirs(video_dir)
    #         os.makedirs(cover_dir)
    #         os.makedirs(container_dir)
    #         os.makedirs(secret_dir)
    #         os.makedirs(revealed_secret_dir)
    
    #     for i in range(num_of_batches):

    #         print("Frame: "+str(i))

    #         test_cover_input = test_cover[i]
    #         test_secret_input = test_secret[i]

    #         test_cover_input = np.reshape(test_cover_input,(1,8,240,320,3))
    #         test_secret_input = np.reshape(test_secret_input,(1,8,240,320,3))

    #         #covers, secrets = covers, secrets
    #         # hiding_b, reveal_b, summaree, tl, cl, sl, ca, sa= self.sess.run([self.hiding_output, self.reveal_output,  self.summary_op, self.loss_op,self.secret_loss_op,self.cover_loss_op, self.cover_acc, self.secret_acc],feed_dict={"input_secret:0":test_secret_input,"input_cover:0":test_cover_input})
    #         hiding_b, reveal_b, summaree, tl, cl, sl= self.sess.run([self.hiding_output, self.reveal_output,  self.summary_op, self.loss_op,self.secret_loss_op,self.cover_loss_op],feed_dict={"input_secret:0":test_secret_input,"input_cover:0":test_cover_input})
    #         print(hiding_b.shape)
    #         self.writer.add_summary(summaree)

    #         for j in range(8):
    #             im = np.reshape(hiding_b[0][j]*255,(240,320,3))
    #             im1 = np.reshape(reveal_b[0][j]*255,(240,320,3))
    #             cv2.imwrite(container_dir+'/'+str(i*8 + j)+'.png', im)
    #             cv2.imwrite(revealed_secret_dir+'/'+str(i*8 + j)+'.png', im1)
    #             im = np.reshape(test_cover_input[0][j]*255,(240,320,3))
    #             im1 = np.reshape(test_secret_input[0][j]*255,(240,320,3))
    #             cv2.imwrite(cover_dir+'/'+str(i*8 + j)+'.png', im)
    #             cv2.imwrite(secret_dir+'/'+str(i*8 + j)+'.png', im1)
    #             # cover_accuracy = cover_accuracy + ca
    #             # secret_accuracy = secret_accuracy + sa
    #             cover_loss = cover_loss + cl
    #             secret_loss = secret_loss + sl
            
    #     # print('Cover_Loss: '+str(cover_loss/Num)+'\tSecret_Loss: '+str(secret_loss/Num)+'\tCover_Accuracy: '+str(cover_accuracy/Num)+'\tSecret_Accuracy: '+str(secret_accuracy/Num))
    #     print('Cover_Loss: '+str(cover_loss/Num)+'\tSecret_Loss: '+str(secret_loss/Num))

    def test(self):

        # self.load_chkp(saver, path)

        self.saver = tf.train.Saver(max_to_keep=2)

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            # start_epoch = (int)(checkpoint_counter / self.iteration)
            # start_batch_id = checkpoint_counter - start_epoch * self.iteration
            count = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            # start_epoch = 0
            # start_batch_id = 0
            count = 1
            print(" [!] Load failed...")
            exit()

        def load_t(base_dir, frame_names, ind):

            fpb = []
            for i in range(self.frames_per_batch):
                frame = base_dir + '/' + frame_names[ind * self.frames_per_batch + i]
                frame = cv2.imread(frame)
                frame = frame / 255.0
                frame = frame.reshape((240,320,3))
                fpb.append(frame)

            return np.reshape(np.array(fpb), (1, self.frames_per_batch, 240, 320, 3))

        def generator():
            for i in range(len(self.cover_tensor_data_test)):
                c_name = '_'.join(self.cover_tensor_data_test[i].split('/')[-2:])
                s_name = '_'.join(self.secret_tensor_data_test[i].split('/')[-2:])
                n1 = len(os.listdir(self.cover_tensor_data_test[i]))
                n2 = len(os.listdir(self.secret_tensor_data_test[i]))
                t = int(min(n1, n2) / self.frames_per_batch)
                frs = sorted(os.listdir(self.cover_tensor_data_test[i]), key=lambda x: int(x.split('.')[0]))[:t * self.frames_per_batch]
                for j in range(t):
                    cov_tens = load_t(self.cover_tensor_data_test[i], frs, j)
                    sec_tens = load_t(self.secret_tensor_data_test[i], frs, j)
                    yield cov_tens, sec_tens, c_name, s_name



        # test_cover, test_secret, c_name, s_name= data.test_data()

        prev_name = ''
        i = 0
        vid = 0

        start_time = time()
        total_frames = 0
        for test_cover, test_secret, c_name, s_name in generator():

            print ('c_name: ' + c_name + ' s_name: ' + s_name + ' vid: '+ str(vid) + ' i: ' + str(i))
            
            if c_name + s_name != prev_name:
                i = 0
                vid += 1
            # batch_size = 1
            # Num = test_cover.shape[0]*test_cover.shape[1]
            # num_of_batches = test_cover.shape[0] // batch_size
            cover_loss = 0
            secret_loss = 0
            cover_accuracy = 0 
            secret_accuracy = 0 

            test_dir = os.path.join(self.test_dir_all, self.test_dir)
            video_dir = os.path.join(test_dir, 'c_'+c_name+'_s_'+s_name)
            cover_dir = os.path.join(video_dir, 'cover')
            container_dir = os.path.join(video_dir, 'container')
            secret_dir = os.path.join(video_dir, 'secret')
            revealed_secret_dir = os.path.join(video_dir, 'revealed_secret')
            diff_cover_container_dir = os.path.join(video_dir, 'diff_cc')
            diff_secret_revealed_dir = os.path.join(video_dir, 'diff_sr')

            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
                os.makedirs(cover_dir)
                os.makedirs(container_dir)
                os.makedirs(secret_dir)
                os.makedirs(revealed_secret_dir)
                os.makedirs(diff_cover_container_dir)
                os.makedirs(diff_secret_revealed_dir)
        
            # for i in range(num_of_batches):

            # print("Frame: "+str(i))

            # test_cover_input = test_cover[i]
            # test_secret_input = test_secret[i]

            # test_cover_input = np.reshape(test_cover_input,(1,8,240,320,3))
            # test_secret_input = np.reshape(test_secret_input,(1,8,240,320,3))

            #covers, secrets = covers, secrets
            # hiding_b, reveal_b, summaree, tl, cl, sl, ca, sa= self.sess.run([self.hiding_output, self.reveal_output,  self.summary_op, self.loss_op,self.secret_loss_op,self.cover_loss_op, self.cover_acc, self.secret_acc],feed_dict={"input_secret:0":test_secret_input,"input_cover:0":test_cover_input})
            hiding_b, reveal_b = self.sess.run([self.hiding_output, self.reveal_output],feed_dict={"input_secret:0":test_secret, "input_cover:0":test_cover})
            # hiding_b = self.sess.run(self.hiding_output,feed_dict={"input_secret:0":test_secret, "input_cover:0":test_cover})
            # self.writer.add_summary(summaree)

            # print (hiding_b)
            for j in range(8):
                im = np.reshape(hiding_b[0][j] * 255, (240,320,3))
                im1 = np.reshape(reveal_b[0][j] * 255,(240,320,3))
                cv2.imwrite(container_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im)
                cv2.imwrite(revealed_secret_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im1)
                im = np.reshape(test_cover[0][j] * 255,(240,320,3))
                im1 = np.reshape(test_secret[0][j] * 255,(240,320,3))
                cv2.imwrite(cover_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im)
                cv2.imwrite(secret_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im1)
                im = np.reshape(np.absolute(hiding_b[0][j] - test_cover[0][j]) * 255, (240, 320, 3))
                im1 = np.reshape(np.absolute(reveal_b[0][j] - test_secret[0][j]) * 255, (240, 320, 3))
                cv2.imwrite(diff_cover_container_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im)
                cv2.imwrite(diff_secret_revealed_dir+'/'+str(i * self.frames_per_batch + j)+'.jpg', im1)
                # cover_accuracy = cover_accuracy + ca
                # secret_accuracy = secret_accuracy + sa
                # cover_loss = cover_loss + cl
                # secret_loss = secret_loss + sl
            
            i += 1    
            total_frames += self.frames_per_batch
            prev_name = c_name + s_name

            # print('Cover_Loss: '+str(cover_loss/Num)+'\tSecret_Loss: '+str(secret_loss/Num)+'\tCover_Accuracy: '+str(cover_accuracy/Num)+'\tSecret_Accuracy: '+str(secret_accuracy/Num))
            # print('Cover_Loss: '+str(cover_loss)+'\tSecret_Loss: '+str(secret_loss/Num))

        total_time = time() - start_time
        pickle.dump(total_time, open('total_time.pkl', 'wb'))
        time_per_frame = float(total_time) / total_frames 
        pickle.dump(time_per_frame, open('time_per_frame.pkl', 'wb'))
        print ('Total time: '+str(total_time)+' Time Per Frame: '+str(time_per_frame))
        

def show_all_variables():
    model_vars = tf.trainable_variables()
    tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)
  

m = SingleSizeModel(beta=0.75, log_path='log/')
show_all_variables()
# m.train()
# test_cover, test_secret, c_name, s_name= data.test_data()
# m.test(test_cover, test_secret, c_name, s_name)
m.test()