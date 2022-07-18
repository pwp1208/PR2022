from __future__ import absolute_import, division, print_function, unicode_literals

try:
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
    pass
import tensorflow as tf

import os
import time
import sys
import glob
import random
from matplotlib import pyplot as plt
from IPython import display
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Lambda, Add, ReLU, MaxPooling2D
import datetime
import numpy as np
from termcolor import colored, cprint
import sys

from PIL import Image

code_name = sys.argv[0]



########################################### Inputs to be given ###################################################
Test_Dir_Path = './video/REVIDE/'
image_save_path = 'outputs/'
Gen1_Checkpoints_path = 'checkpoint/ckpt_REVIDE/'

BATCH_SIZE = 1
IMG_WIDTH, IMG_HEIGHT = 256, 256
OUTPUT_CHANNELS = 3

##################################################################################################################

test_dataset =sorted(glob.glob(Test_Dir_Path),key = os.path.getmtime)


def load(image_file):
    inputs_batch= []
    gt_batch = []
    
    for i in range(len(image_file)):
        image = tf.io.read_file(image_file[i])
        image = tf.image.decode_png(image)
        w = tf.shape(image)[1]
        w = w // 2
        input_image = image[:, w:, :]
        input_image = tf.image.resize(input_image, [IMG_HEIGHT, IMG_WIDTH],
                                method=tf.image.ResizeMethod.BILINEAR) 
        GT_Enc = image[:, :w, :]
        GT_Enc = tf.image.resize(GT_Enc, [IMG_HEIGHT, IMG_WIDTH],
                                method=tf.image.ResizeMethod.BILINEAR)
        input_image = tf.expand_dims(input_image, axis=0)

        GT_Enc = tf.expand_dims(GT_Enc, axis=0)
        inputs_batch.append(input_image)
        gt_batch.append(GT_Enc)
  
    inputs_batch1 = tf.concat([inputs_batch[k] for k in range(len(inputs_batch))],axis=0)
    GT_Enc_batch1 = tf.concat([gt_batch[k] for k in range(len(gt_batch))],axis=0)

    inputs_batch1 = tf.cast(inputs_batch1, tf.float32)
    GT_Enc_batch1 = tf.cast(GT_Enc_batch1, tf.float32)

    return inputs_batch1, GT_Enc_batch1


def normalize(inputs_batch, GT_Enc_batch):
    inputs_batch = (inputs_batch / 127.5) - 1
    GT_Enc_batch = (GT_Enc_batch / 127.5) - 1
   
    return inputs_batch, GT_Enc_batch

def load_image_train(image_file):
    inputs_batch, GT_Enc_batch = load(image_file)
    inputs_batch, GT_Enc_batch = normalize(inputs_batch, GT_Enc_batch)

    return inputs_batch, GT_Enc_batch


def load_image_test(image_file):
    inputs_batch, GT_Enc_batch = load(image_file)
    inputs_batch, GT_Enc_batch = normalize(inputs_batch, GT_Enc_batch)

    return inputs_batch, GT_Enc_batch


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def Conv_Block(filters, size, stride=1, dilation_rate=1, name='',activation='relu'):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=stride, padding='same', dilation_rate=1,
                             kernel_initializer=initializer, activation=activation, name='Conv_'+name))

    return result 

def DeConv_Block(filters, size, stride=2, name=''):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=stride,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    name='DeConv_'+name))
    result.add(tf.keras.layers.ReLU())
    return result  


def Conv_Activation(filters, size, stride=1, name=''):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=stride, padding='same',
                             kernel_initializer=initializer, activation='tanh',
                            name='Conv_'+name))
    return result  

def batch_norm(tensor):
    return tf.keras.layers.BatchNormalization(axis=3,epsilon=1e-5, momentum=0.1, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(tensor)

def lap_split(img,kernel):
    with tf.name_scope('split'):
        low = tf.nn.conv2d(img, kernel, [1,2,2,1], 'SAME')
        low_upsample = tf.nn.conv2d_transpose(low, kernel*4, tf.shape(img), [1,2,2,1])
        high = img - low_upsample
    return low, high

def LaplacianPyramid(img,kernel,n):
    levels = []
    for i in range(n):
        img, high = lap_split(img, kernel)
        levels.append(high)
    levels.append(img)
    return levels[::-1]


##################################### Generator 1 definition ##################################################
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
class Generator:

    def __init__(self):
        self.inputs = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT,3])
        self.inputs_t2 = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT,3])
        self.inputs2 = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT,3])


        self.name = 'Generator_/'
        self.filters = 16

        self.in_32 = tf.keras.layers.Input(shape=[IMG_WIDTH//8, IMG_HEIGHT//8,self.filters*4])
        self.in_64 = tf.keras.layers.Input(shape=[IMG_WIDTH//4, IMG_HEIGHT//4,self.filters*3])
        self.in_128 = tf.keras.layers.Input(shape=[IMG_WIDTH//2, IMG_HEIGHT//2,self.filters*2])
        self.in_256 = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT,self.filters*1])


    def multiscale_feat(self, inp, filt, name = ''):

        Conv_inp1 = Conv_Block(filt, 3, stride=1, dilation_rate=1, activation=None, name=name+'Dil_Conv1')(inp)
        Conv_inp2 = Conv_Block(filt, 3, stride=1, dilation_rate=2, activation=None, name=name+'Dil_Conv2')(inp)
        Conv_inp3 = Conv_Block(filt, 3, stride=1, dilation_rate=3, activation=None, name=name+'Dil_Conv3')(inp)
        Conv_inp4 = Conv_Block(filt, 3, stride=1, dilation_rate=4, activation=None, name=name+'Dil_Conv4')(inp)

           
        concat_12 = tf.keras.layers.concatenate([Conv_inp1, Conv_inp2], axis=-1)
        Conv_concat_12 = Conv_Block(filt, 1, stride=1, dilation_rate=1, activation=None, name=name+'_concat_12')(concat_12)

        concat_13 = tf.keras.layers.concatenate([Conv_inp1, Conv_inp3], axis=-1)
        Conv_concat_13 = Conv_Block(filt, 1, stride=1, dilation_rate=1, activation=None, name=name+'_concat_13')(concat_13)

        concat_14 = tf.keras.layers.concatenate([Conv_inp1, Conv_inp4], axis=-1)
        Conv_concat_14 = Conv_Block(filt, 1, stride=1, dilation_rate=1, activation=None, name=name+'_concat_14')(concat_14)

        concat_23 = tf.keras.layers.concatenate([Conv_inp2, Conv_inp3], axis=-1)
        Conv_concat_23 = Conv_Block(filt, 1, stride=1, dilation_rate=1, activation=None, name=name+'_concat_23')(concat_23)

        concat_24 = tf.keras.layers.concatenate([Conv_inp2, Conv_inp3], axis=-1)
        Conv_concat_24 = Conv_Block(filt, 1, stride=1, dilation_rate=1, activation=None, name=name+'_concat_24')(concat_24)

        concat_34 = tf.keras.layers.concatenate([Conv_inp3, Conv_inp4], axis=-1)
        Conv_concat_34 = Conv_Block(filt, 1, stride=1, dilation_rate=1, activation=None, name=name+'_concat_34')(concat_34)

        concat_all = tf.keras.layers.concatenate([Conv_concat_12, Conv_concat_13,Conv_concat_14,Conv_concat_23,Conv_concat_24,Conv_concat_34], axis=-1)
        Conv_concat_all = tf.sigmoid(Conv_Block(filt, 1, stride=1, dilation_rate=1, activation=None, name=name+'Dil_Conv5')(concat_all))

        return Conv_concat_all

    def add_sub_block(self, inp, filt, name=''):
        Conv_1 = Conv_Block(filt, 1, stride=1, dilation_rate=1, activation=None, name=name+'conv_1')(inp)
        Conv_3 = Conv_Block(filt, 3, stride=1, dilation_rate=1, activation=None, name=name+'conv_3')(inp)
        Conv_5 = Conv_Block(filt, 5, stride=1, dilation_rate=1, activation=None, name=name+'conv_5')(inp)

        add_1 = Conv_1 + Conv_3
        add_2 = Conv_3 + Conv_5
        add_3 = Conv_5 + Conv_1
        add_cat = tf.keras.layers.concatenate([add_1, add_2, add_3], axis=-1)
        Conv_add = Conv_Block(filt, 1, stride=1, dilation_rate=1, activation=None, name=name+'add_cat')(add_cat)

        sub_1 = Conv_1 - Conv_3
        sub_2 = Conv_3 - Conv_5
        sub_3 = Conv_5 - Conv_1
        sub_cat = tf.keras.layers.concatenate([sub_1, sub_2, sub_3], axis=-1)
        Conv_sub = Conv_Block(filt, 1, stride=1, dilation_rate=1, activation=None, name=name+'sub_cat')(sub_cat)

        all_cat = tf.keras.layers.concatenate([Conv_add, Conv_sub, inp], axis=-1)
        out = Conv_Block(filt, 1, stride=1, dilation_rate=1, activation='relu', name=name+'all_cat')(all_cat)

        return out


    def inception_module(self, inp, filt,name):
        c3 = Conv_Block(filt, 3, stride=1, dilation_rate=1, name=name + '_c3')(inp)    
        c5 = Conv_Block(filt, 5, stride=1, dilation_rate=1, name=name + '_c5')(inp)    
        c7 = Conv_Block(filt, 7, stride=1, dilation_rate=1, name=name + '_c7')(inp) 
        cat_357 = tf.concat([c3, c5, c7], 3)   
        conv_out = Conv_Block(filt, 1, stride=1, dilation_rate=1, name=name + '_conv_cat_357')(cat_357)   
        return conv_out

    def scale1_path(self, inputs1, filt, name = ''):
        scale_1 = self.multiscale_feat(inputs1, filt, name = name +'scale1')
        add_sub_scale_1 = self.add_sub_block(scale_1, filt, name=name +'add_sub_scale_1')

        scale_11 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(add_sub_scale_1)#128*128
        add_sub_scale_11 = self.add_sub_block(scale_11, filt*2, name=name +'add_sub_scale_11')

        scale_12 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(add_sub_scale_11)#64*64
        add_sub_scale_12 = self.add_sub_block(scale_12, filt*3, name=name +'add_sub_scale_12')

        scale_13 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(add_sub_scale_12)#32*32
        add_sub_scale_13 = self.add_sub_block(scale_13, filt*4, name=name +'add_sub_scale_13')

        return add_sub_scale_1, add_sub_scale_11, add_sub_scale_12, add_sub_scale_13

    def scale2_path(self, in_scale2, filt, name = ''):
        scale_2 = self.multiscale_feat(in_scale2, filt, name = name+'scale2')
        add_sub_scale_2 = self.add_sub_block(scale_2, filt, name=name+'add_sub_scale_2')#128*128

        scale_21 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(add_sub_scale_2)#64*64
        add_sub_scale_21 = self.add_sub_block(scale_21, filt*2, name=name+'add_sub_scale_21')

        scale_22 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(add_sub_scale_21)#32*32
        add_sub_scale_22 = self.add_sub_block(scale_22, filt*3, name=name+'add_sub_scale_22')

        return add_sub_scale_2, add_sub_scale_21, add_sub_scale_22


    def scale3_path(self, in_scale3, filt, name = ''):
        scale_3 = self.multiscale_feat(in_scale3, filt, name = name+'scale3')
        add_sub_scale_3 = self.add_sub_block(scale_3, filt, name=name+'add_sub_scale_3')#64*64

        scale_31 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(add_sub_scale_3)#32*32
        add_sub_scale_31 = self.add_sub_block(scale_31, filt*2, name=name+'add_sub_scale_31')

        return add_sub_scale_3, add_sub_scale_31

    def scale4_path(self, in_scale4, filt, name = '' ):
        scale_4 = self.multiscale_feat(in_scale4, filt, name = name+'scale4')
        add_sub_scale_4 = self.add_sub_block(scale_4, filt, name=name+'add_sub_scale_4')#32*32

        return add_sub_scale_4

    def time_modulation(self, inp1, inp2, filt, name):
        avg_features = tf.keras.layers.Average()([inp1, inp2])
        conv_avg = Conv_Block(filt, 3, stride=1, dilation_rate=1, name=name + '_c_avg')(avg_features)

        alpha = tf.sigmoid(tf.keras.layers.GlobalAveragePooling2D()(conv_avg))

        concat_feat = tf.concat([inp1, inp2], 3)
        conv_concat = Conv_Block(filt, 3, stride=1, dilation_rate=1, name=name + '_c_concat')(concat_feat)

        beta = tf.keras.layers.GlobalAveragePooling2D()(conv_concat)

        multiplied_avg = tf.keras.layers.Multiply()([conv_concat, alpha])

        output = multiplied_avg + beta
        return output

    def scale1_modulation(self, inp1, filt, name):
        conv_avg = Conv_Block(filt, 3, stride=1, dilation_rate=1, name=name + '_conv')(inp1)

        alpha = tf.sigmoid(tf.keras.layers.GlobalAveragePooling2D()(conv_avg))

        conv_concat = Conv_Block(filt, 3, stride=1, dilation_rate=1, name=name + '_conv_concat')(inp1)

        beta = tf.keras.layers.GlobalAveragePooling2D()(conv_concat)

        multiplied_avg = tf.keras.layers.Multiply()([conv_concat, alpha])

        output = multiplied_avg + beta
        return output

    def scale2_modulation(self, inp1, inp2, filt, name):
        avg_features = tf.keras.layers.Average()([inp1, inp2])
        conv_avg = Conv_Block(filt, 3, stride=1, dilation_rate=1, name=name + '_c_avg')(avg_features)

        alpha = tf.sigmoid(tf.keras.layers.GlobalAveragePooling2D()(conv_avg))

        concat_feat = tf.concat([inp1, inp2], 3)
        conv_concat = Conv_Block(filt, 3, stride=1, dilation_rate=1, name=name + '_c_concat')(concat_feat)

        beta = tf.keras.layers.GlobalAveragePooling2D()(conv_concat)

        multiplied_avg = tf.keras.layers.Multiply()([conv_concat, alpha])

        output = multiplied_avg + beta
        return output

    def scale3_modulation(self, inp1, inp2, inp3, filt, name):
        avg_features = tf.keras.layers.Average()([inp1, inp2, inp3])
        conv_avg = Conv_Block(filt, 3, stride=1, dilation_rate=1, name=name + '_c_avg')(avg_features)

        alpha = tf.sigmoid(tf.keras.layers.GlobalAveragePooling2D()(conv_avg))

        concat_feat = tf.concat([inp1, inp2, inp3], 3)
        conv_concat = Conv_Block(filt, 3, stride=1, dilation_rate=1, name=name + '_c_concat')(concat_feat)

        beta = tf.keras.layers.GlobalAveragePooling2D()(conv_concat)

        multiplied_avg = tf.keras.layers.Multiply()([conv_concat, alpha])

        output = multiplied_avg + beta
        return output

    
    def generator1_fun(self):
  
        name1 = 'generator1'
        inputs1 = self.inputs*1.0
        inputs2 = self.inputs_t2*1.0
       
        print(self.inputs.shape)

        t1_s11, t1_s12, t1_s13, t1_s14 = self.scale1_path(inputs1, self.filters, name = 'time1')
        t2_s11, t2_s12, t2_s13, t2_s14 = self.scale1_path(inputs2, self.filters, name = 'time2')  

        mod_s11 = self.time_modulation(t1_s11, t2_s11, self.filters, name='time_mod_s11')
        mod_s12 = self.time_modulation(t1_s12, t2_s12, self.filters*2, name='time_mod_s12')
        mod_s13 = self.time_modulation(t1_s13, t2_s13, self.filters*3, name='time_mod_s13')
        mod_s14 = self.time_modulation(t1_s14, t2_s14, self.filters*4, name='time_mod_s14')

        in_scale2_t1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(inputs1)
        in_scale2_t2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(inputs2)

        t1_s21, t1_s22, t1_s23 = self.scale2_path(in_scale2_t1, self.filters, name = 'time1')
        t2_s21, t2_s22, t2_s23 = self.scale2_path(in_scale2_t2, self.filters, name = 'time2') 

        mod_s21 = self.time_modulation(t1_s21, t2_s21, self.filters*2, name='time_mod_s21')
        mod_s22 = self.time_modulation(t1_s22, t2_s22, self.filters*3, name='time_mod_s22')
        mod_s23 = self.time_modulation(t1_s23, t2_s23, self.filters*4, name='time_mod_s23')         


        in_scale3_t1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(in_scale2_t1)
        in_scale3_t2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(in_scale2_t2)

        t1_s31, t1_s32 = self.scale3_path(in_scale3_t1, self.filters, name = 'time1')
        t2_s31, t2_s32 = self.scale3_path(in_scale3_t2, self.filters, name = 'time2') 

        mod_s31 = self.time_modulation(t1_s31, t2_s31, self.filters*3, name='time_mod_s31')
        mod_s32 = self.time_modulation(t1_s32, t2_s32, self.filters*4, name='time_mod_s32') 

        in_scale4_t1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(in_scale3_t1)
        in_scale4_t2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(in_scale3_t2)

        t1_s41 = self.scale4_path(in_scale4_t1, self.filters, name = 'time1')
        t2_s41 = self.scale4_path(in_scale4_t2, self.filters, name = 'time2') 

        mod_s41 = self.time_modulation(t1_s41, t2_s41, self.filters*4, name='time_mod_s41')     
    
        enc_out = tf.concat([mod_s41, mod_s32, mod_s23, mod_s14], 3)
        con_enc_out = Conv_Block(self.filters*4, 1, stride=1, dilation_rate=1, name=name1 + 'enc_out')(enc_out)#32*32 

#########################################################decoder###############################################

        dec_add_sub1 = self.inception_module( con_enc_out, self.filters*4, name=name1+'dec_add_sub1')
        rec_cat1 = tf.concat([dec_add_sub1, self.in_32], 3)
        conv_rec_cat1 = Conv_Block(self.filters*4, 1, stride=1, dilation_rate=1, name=name1 + 'rec_cat1')(rec_cat1)
        dec_out1 = DeConv_Block(self.filters*4, 3, stride=2, name=name1+'deconv1')(conv_rec_cat1)#64*64


        skip1 = self.scale3_modulation(mod_s31, mod_s22, mod_s13, self.filters*3, name = name1 + 'scale3_mod')
        skip_in1 = tf.concat([skip1, dec_out1], 3) #64*64
        dec_add_sub2 = self.inception_module( skip_in1, self.filters*3, name=name1+'dec_add_sub2')
        rec_cat2 = tf.concat([dec_add_sub2, self.in_64], 3)
        conv_rec_cat2 = Conv_Block(self.filters*3, 1, stride=1, dilation_rate=1, name=name1 + 'rec_cat2')(rec_cat2)
        dec_out2 = DeConv_Block(self.filters*3, 3, stride=2, name=name1+'deconv2')(conv_rec_cat2)#128*128

        
        skip2 = self.scale2_modulation(mod_s21, mod_s12, self.filters*2, name = name1+'scale2_mod')
        skip_in2 = tf.concat([skip2, dec_out2], 3) #128*128
        dec_add_sub3 = self.inception_module( skip_in2, self.filters*2, name=name1+'dec_add_sub3')
        rec_cat3 = tf.concat([dec_add_sub3, self.in_128], 3)
        conv_rec_cat3 = Conv_Block(self.filters*2, 1, stride=1, dilation_rate=1, name=name1 + 'rec_cat3')(rec_cat3)
        dec_out3 = DeConv_Block(self.filters*2, 3, stride=2, name=name1+'deconv3')(conv_rec_cat3)#256*256


        skip3 = self.scale1_modulation(mod_s11, self.filters, name = name1+'scale1_mod')
        skip_in3 = tf.concat([dec_out3, skip3], 3)
        dec_add_sub4 = self.inception_module( skip_in3, self.filters, name=name1+'dec_add_sub4')
        rec_cat4 = tf.concat([dec_add_sub4, self.in_256], 3)
        conv_rec_cat4 = Conv_Block(self.filters*1, 1, stride=1, dilation_rate=1, name=name1 + 'rec_cat4')(rec_cat4)
        dec_out4 = DeConv_Block(self.filters, 3, stride=1, name=name1+'deconv4')(conv_rec_cat4)#256*256

        Output = Conv_Activation(3, 1, stride=1, name=name1+ '_ConvAct_last')(dec_out4) 
         
        return tf.keras.Model(inputs=[self.inputs,self.inputs_t2, self.in_32,self.in_64,self.in_128,self.in_256] , outputs=[Output, dec_add_sub1, dec_add_sub2, dec_add_sub3, dec_add_sub4])

    
gen = Generator()
generator1_model = gen.generator1_fun()

print('='*50)
text = print('Total Trainable parameters of Generator 1 are :: {}'.format(generator1_model.count_params()))
print('='*50)

###############################################################################################################

generator1_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

################################ generator1 checkpoints###############################
checkpoint_dir1 = Gen1_Checkpoints_path
checkpoint_prefix1 = os.path.join(checkpoint_dir1, "ckpt")
checkpoint1 = tf.train.Checkpoint(generator1_optimizer=generator1_optimizer,
                                 generator1=generator1_model)

checkpoint1.restore(tf.train.latest_checkpoint(Gen1_Checkpoints_path)).expect_partial()
print(colored('Checkpoint Restored !for gen 1!!!!','cyan'))
print(colored('='*50,'cyan'))


########################## Generate Images  ########################################

def generate_images(generator1_model, test_input, save_filenames, folder = image_save_path, mode='train'):
    with tf.device('/device:cpu:0'):
        
        in1 = tf.expand_dims(test_input,axis=0)
        results1,dec_32,dec_64,dec_128,dec_256 = [], [], [], [], []

        for i in range(test_input.shape[0]):
            if i ==0:

                results_new_128 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(in1[:,i,:,:,:])
                results_new_64 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(results_new_128)
                results_new_32 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(results_new_64)
                result_new_256 = in1[:,i,:,:,:]

                result_256 = tf.concat([result_new_256,result_new_256,result_new_256,result_new_256,result_new_256,tf.expand_dims(result_new_256[:,:,:,0],axis = -1)],axis = 3)
                
                result_128_1 = tf.concat([results_new_128,results_new_128,results_new_128,results_new_128,results_new_128,tf.expand_dims(results_new_128[:,:,:,0],axis = -1)],axis=3)
                result_128 = tf.concat([result_128_1,result_128_1], axis=3)

                result_64_1 = tf.concat([results_new_64,results_new_64,results_new_64,results_new_64,results_new_64,tf.expand_dims(results_new_64[:,:,:,0],axis = -1)],axis=3)
                result_64 = tf.concat([result_64_1,result_64_1,result_64_1],axis =3)

                result_32_1  = tf.concat([results_new_32,results_new_32,results_new_32,results_new_32,results_new_32,tf.expand_dims(results_new_32[:,:,:,0],axis = -1)],axis=3)
                result_32 = tf.concat([result_32_1,result_32_1,result_32_1,result_32_1],axis = 3)

                [output1,dec_add_sub1,dec_add_sub2,dec_add_sub3,dec_add_sub4] = generator1_model([in1[:,i,:,:,:],in1[:,i,:,:,:],result_32,result_64,result_128,result_256], training=True)

            elif i == 1:
                results_new_128 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(results1[-1])
                results_new_64 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(results_new_128)
                results_new_32 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(results_new_64)
                result_new_256 = in1[:,i,:,:,:]

                result_256 = tf.concat([result_new_256,result_new_256,result_new_256,result_new_256,result_new_256,tf.expand_dims(result_new_256[:,:,:,0],axis = -1)],axis = 3)
                
                result_128_1 = tf.concat([results_new_128,results_new_128,results_new_128,results_new_128,results_new_128,tf.expand_dims(results_new_128[:,:,:,0],axis = -1)],axis=3)
                result_128 = tf.concat([result_128_1,result_128_1], axis=3)

                result_64_1 = tf.concat([results_new_64,results_new_64,results_new_64,results_new_64,results_new_64,tf.expand_dims(results_new_64[:,:,:,0],axis = -1)],axis=3)
                result_64 = tf.concat([result_64_1,result_64_1,result_64_1],axis =3)

                result_32_1  = tf.concat([results_new_32,results_new_32,results_new_32,results_new_32,results_new_32,tf.expand_dims(results_new_32[:,:,:,0],axis = -1)],axis=3)
                result_32 = tf.concat([result_32_1,result_32_1,result_32_1,result_32_1],axis = 3)

                [output1,dec_add_sub1,dec_add_sub2,dec_add_sub3,dec_add_sub4] = generator1_model([in1[:,i,:,:,:],in1[:,i-1,:,:,:],result_32,result_64,result_128,result_256], training=True)
         
            else:
                # print(i) 
                results_new_128 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(results1[-1])
                results_new_64 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(results_new_128)
                results_new_32 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(results_new_64)
                result_new_256 = results1[-1]

                result_256 = tf.concat([result_new_256,result_new_256,result_new_256,result_new_256,result_new_256,tf.expand_dims(result_new_256[:,:,:,0],axis = -1)],axis = 3)
                
                result_128_1 = tf.concat([results_new_128,results_new_128,results_new_128,results_new_128,results_new_128,tf.expand_dims(results_new_128[:,:,:,0],axis = -1)],axis=3)
                result_128 = tf.concat([result_128_1,result_128_1], axis=3)

                result_64_1 = tf.concat([results_new_64,results_new_64,results_new_64,results_new_64,results_new_64,tf.expand_dims(results_new_64[:,:,:,0],axis = -1)],axis=3)
                result_64 = tf.concat([result_64_1,result_64_1,result_64_1],axis =3)

                result_32_1  = tf.concat([results_new_32,results_new_32,results_new_32,results_new_32,results_new_32,tf.expand_dims(results_new_32[:,:,:,0],axis = -1)],axis=3)
                result_32 = tf.concat([result_32_1,result_32_1,result_32_1,result_32_1],axis = 3)          

                [output1,dec_add_sub1,dec_add_sub2,dec_add_sub3,dec_add_sub4] = generator1_model([in1[:,i,:,:,:],in1[:,i-2,:,:,:],result_32,result_64,result_128,result_256], training=True)
            
            dec_32.append(dec_add_sub1)
            dec_64.append(dec_add_sub2)
            dec_128.append(dec_add_sub3)
            dec_256.append(dec_add_sub4)
            results1.append(output1)

        outputs = results1[0]        

        for i in range(1,len(results1)):
            outputs = tf.concat([outputs , results1[i]], axis=0)

        try :
            os.mkdir(folder)
        except:
            pass

        for j in range(1,test_input.shape[0]):

            save_name = save_filenames[j].split('\\') 
            out1 = Image.fromarray(np.array((outputs[j]*0.5 + 0.5)*255, dtype='uint8'))
            new_im = Image.new('RGB', (IMG_HEIGHT*1, IMG_WIDTH))
               
            x_offset = 0
            for im in [out1]:
              new_im.paste(im, (x_offset,0))
              x_offset += im.size[0]
       
            new_im.save(folder + '/'+save_name[-1])

log_dir = "logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

n = 0
for i,filename in enumerate(test_dataset):
    filenames1 = sorted(glob.glob(test_dataset[i] +'*.png'),key =  os.path.getmtime)
     
    gen_input, target =load_image_train(filenames1)
    generate_images(generator1_model, gen_input, filenames1, mode='test')
    print('n=',n,end='\r')
    n+=1 
print("TESTED SUCCESSFULLY")

