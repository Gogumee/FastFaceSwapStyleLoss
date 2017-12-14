import os
import sys
import time
import inspect

import tensorflow as tf
import numpy as np
from scipy.io import loadmat

import skimage
import skimage.io
import skimage.transform

import scipy

class VggFace2:
    except_layers = (
        'conv2_1_1x1_proj', 'conv2_1_1x1_proj_bn', 'conv2_1', 'conv2_2', 'conv2_3', 
        'conv3_1_1x1_proj', 'conv3_1_1x1_proj_bn', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
        'conv4_1_1x1_proj', 'conv4_1_1x1_proj_bn', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv4_5', 'conv4_6',
        'conv5_1_1x1_proj', 'conv5_1_1x1_proj_bn', 'conv5_1', 'conv5_2', 'conv5_3'
   )

    def __init__(self, vggFace2_mat_path=None):
        self.mean_pixel = np.array([131.0912,  103.8827,   91.4953])
        
        if vggFace2_mat_path is None:
            path = inspect.getfile(VggFace2)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "./matconvnet/senet/senet50_ft-dag.mat")
            vggFace2_mat_path = path
            print(vggFace2_mat_path)

        print("mat file loading...")
        data = loadmat("./matconvnet/senet/senet50_ft-dag.mat")
        
        print("make data_dict...")
        self.data_dict = {}
        for idx in range(len(data['params'][0])):
            param_name = data['params'][0][idx][0][0]
            param_value = data['params'][0][idx][1]
            self.data_dict[param_name] = param_value
        
        print("make layer info...")
        self.layer_info = {}
        self.save_required_layer = []
        self.layers = data['layers'][0]
        for idx in range(len(data['layers'][0])):
            per_layer_info = {}
            
            layer_name = self.layers[idx]['name'][0]
            layer_type = self.layers[idx]['type'][0]
            
            if layer_name in self.except_layers:
                if len(self.layers[idx][2][0]) > 1:
                    for i, input_name in enumerate(self.layers[idx][2][0]):
                        self.save_required_layer.append(input_name[0])
                else:
                    input_name = self.layers[idx][2][0][0][0]
                    self.save_required_layer.append(input_name)

            output_name = self.layers[idx][3][0][0][0]            
            params_names = []
            if len(self.layers[idx][4]) > 0:
                for i in range(len(self.layers[idx][4][0])):
                    params_names.append(self.layers[idx][4][0][i][0])
            meta_info = {}
            if self.layers[idx][5] == None:
                meta_info = {None}
            else:
                for meta_info_name in self.layers[idx][5].dtype.names:
                    if meta_info_name in ['numChannels', 'opts']:
                        meta_info[meta_info_name] = self.layers[idx][5][meta_info_name][0][0]
                    elif meta_info_name in ['pad', 'dilate', 'size', 'stride', 'method', 'poolSize', 'shape']:
                        meta_info[meta_info_name] = self.layers[idx][5][meta_info_name][0][0][0]
                    elif meta_info_name in ['epsilon', 'hasBias', 'useShortCircuit', 'leak']:
                        meta_info[meta_info_name] = self.layers[idx][5][meta_info_name][0][0][0][0]
                    else:
                        meta_info[meta_info_name] = self.layers[idx][5][meta_info_name][0][0][0][0][0]
            per_layer_info['name'] = layer_name
            per_layer_info['type'] = layer_type
            per_layer_info['output_name'] = output_name
            per_layer_info['params'] = params_names
            per_layer_info['meta'] = meta_info
            self.layer_info[layer_name] = per_layer_info

#     def _undo():
#         out_for_print = out + np.array([131.0912,  103.8827,   91.4953])
#         out_for_print = out_for_print/255.0

    def build(self, input_image, target_layer, isImageNormalized=True):
        """
        :param input_image: rgb image [batch, height, width, 3] values scaled [0,1] - when use skimage
        """
        start_time = time.time()
        print("build model started...")
        if isImageNormalized:
            input_image = input_image * 255.0
        
        input_image = input_image - self.mean_pixel
        
        if not (input_image.get_shape().as_list()[1:] == [224, 224, 3]):
            input_image = tf.image.resize_bicubic(input_image, [224, 224])

        current = input_image
        save_required_params = {}
        for idx, layer in enumerate(self.layers):
            name = self.layers['name'][idx][0]
            layer_type = self.layers['type'][idx][0]
            
            if name in self.except_layers:
                if layer_type == 'dagnn.Axpy':
                    A_name, x_name, y_name = self.get_bottom_name(name)
                    A = save_required_params[A_name]
                    x = save_required_params[x_name]
                    y = save_required_params[y_name]
                    current = A * x + y
                    
                elif layer_type == 'dagnn.Conv':
                    bottom_name = self.get_bottom_name(name)
                    bottom = save_required_params[bottom_name]
                    bottom = self.conv_layer(bottom, name)
                elif layer_type == 'dagnn.BatchNorm':
                    bottom_name = name[:-3]
                    bottom = save_required_params[bottom_name]
                    bottom = self.batch_normalization_layer(bottom, name)
                else:
                    sys.exit(0)
                    
                if self.layer_info[name]['output_name'] in self.save_required_layer:
                    save_required_params[self.layer_info[name]['output_name']] = bottom
            else:
                if layer_type == 'dagnn.Conv':
                    current = self.conv_layer(current, name)
                elif layer_type == 'dagnn.BatchNorm':
                    current = self.batch_normalization_layer(current, name)
                elif layer_type == 'dagnn.ReLU':
                    current = tf.nn.relu(current)
                elif layer_type == 'dagnn.Pooling':
                    current = self.pooling_layer(current, name)
                elif layer_type == 'dagnn.GlobalPooling':
                    current = self.global_average_pooling(current, name)
                elif layer_type == 'dagnn.Sigmoid':
                    current = tf.nn.sigmoid(current)
                elif layer_type == 'dagnn.Reshape':
                    shape_layer = name[:7] + '_1x1_increase_bn'
                    reshape_size = save_required_params[shape_layer].get_shape().as_list()[2]
                else:
                    sys.exit(0)

                if self.layer_info[name]['output_name'] in self.save_required_layer:
                    save_required_params[self.layer_info[name]['output_name']] = current

            if name == target_layer:
                break
        print(("build model finished: %ds" % (time.time() - start_time)))
        return current

    def get_bottom_name(self, name):
        if name == 'conv2_1_1x1_proj':
            return 'pool1_3x3_s2'
        if name == 'conv3_1_1x1_proj':
            return 'conv2_3x'
        if name == 'conv4_1_1x1_proj':
            return 'conv3_4x'
        if name == 'conv5_1_1x1_proj':
            return 'conv4_6x'
        if name == 'conv2_1':
            return ('conv2_1_prob_reshape', 'conv2_1_1x1_increase_bn', 'conv2_1_1x1_proj_bn')
        if name == 'conv2_2':
            return ('conv2_2_prob_reshape', 'conv2_2_1x1_increase_bn', 'conv2_1x')
        if name == 'conv2_3':
            return ('conv2_3_prob_reshape', 'conv2_3_1x1_increase_bn', 'conv2_2x')
        if name == 'conv3_1':
            return ('conv3_1_prob_reshape', 'conv3_1_1x1_increase_bn', 'conv3_1_1x1_proj_bn')
        if name == 'conv3_2':
            return ('conv3_2_prob_reshape', 'conv3_2_1x1_increase_bn', 'conv3_1x')
        if name == 'conv3_3':
            return ('conv3_3_prob_reshape', 'conv3_3_1x1_increase_bn', 'conv3_2x')
        if name == 'conv3_4':
            return ('conv3_4_prob_reshape', 'conv3_4_1x1_increase_bn', 'conv3_3x')
        if name == 'conv4_1':
            return ('conv4_1_prob_reshape', 'conv4_1_1x1_increase_bn', 'conv4_1_1x1_proj_bn')
        if name == 'conv4_2':
            return ('conv4_2_prob_reshape', 'conv4_2_1x1_increase_bn', 'conv4_1x')
        if name == 'conv4_3':
            return ('conv4_3_prob_reshape', 'conv4_3_1x1_increase_bn', 'conv4_2x')
        if name == 'conv4_4':
            return ('conv4_4_prob_reshape', 'conv4_4_1x1_increase_bn', 'conv4_3x')
        if name == 'conv4_5':
            return ('conv4_5_prob_reshape', 'conv4_5_1x1_increase_bn', 'conv4_4x')
        if name == 'conv4_6':
            return ('conv4_6_prob_reshape', 'conv4_6_1x1_increase_bn', 'conv4_5x')
        if name == 'conv5_1':
            return ('conv5_1_prob_reshape', 'conv5_1_1x1_increase_bn', 'conv5_1_1x1_proj_bn')
        if name == 'conv5_2':
            return ('conv5_2_prob_reshape', 'conv5_2_1x1_increase_bn', 'conv5_1x')
        if name == 'conv5_3':
            return ('conv5_3_prob_reshape', 'conv5_3_1x1_increase_bn', 'conv5_2x')
                    
    def global_average_pooling(self, bottom, name):
        with tf.variable_scope(name):
            return tf.reduce_mean(bottom, axis=[1, 2], keep_dims=True)
            
    def pooling_layer(self, bottom, name):
        with tf.variable_scope(name):
            method = self.layer_info[name]['meta']['method']
            if method == 'max':
                padded_input = tf.pad(bottom, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT")
                return tf.nn.max_pool(padded_input, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)
            else: # 'avg'
                return tf.nn.avg_pool(bottom, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID', name=name)
                
    # sigma(k) = sqrt(sigma2(k) + EPSILON)
    def batch_normalization_layer(self, bottom, name):
        with tf.variable_scope(name):
            bn_mult = np.squeeze(self.data_dict[name + '_mult'])
            bn_bias = np.squeeze(self.data_dict[name + '_bias'])
            bn_mu = self.data_dict[name + '_moments'][:, 0]
            bn_sigma = self.data_dict[name + '_moments'][:, 1]
            return tf.nn.batch_normalization(bottom, bn_mu, (tf.square(bn_sigma)-1e-5), bn_bias, bn_mult, 1e-5, name=name)
            
    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            kernel = self.get_conv_kernel(name)
            bias = self.get_conv_bias(name)
            stride = self.get_conv_stride(name)
            
            conv = tf.nn.conv2d(bottom, kernel, stride, padding='SAME')
            return tf.nn.bias_add(conv, bias)

    def get_conv_kernel(self, name):
        # matconvnet: kernels are [width, height, in_channels, out_channels]
        # tensorflow: kernels are [height, width, in_channels, out_channels]
        kernel = self.data_dict[name + '_filter']
        kernel = np.transpose(kernel, (1, 0, 2, 3))
        
        return tf.constant(kernel, name="kernels")

    def get_conv_stride(self, name):
        s = self.layer_info[name]['meta']['stride'][0]
        return [1, s, s, 1]
        
    def get_conv_bias(self, name):
        if self.layer_info[name]['meta']['hasBias']:
            return tf.constant(np.squeeze(self.data_dict[name + '_bias']), name="bias")
        else:
            bias_size = self.layer_info[name]['meta']['size'][3]
            return tf.zeros([bias_size], tf.float32, name="bias")