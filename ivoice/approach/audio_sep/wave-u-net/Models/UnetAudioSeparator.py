import tensorflow as tf

import Models.InterpolationLayer
import Utils
from Utils import LeakyReLU
import numpy as np
import Models.OutputLayer

class UnetAudioSeparator:
    '''
    U-Net separator network for singing voice separation.
    Uses valid convolutions, so it predicts for the centre part of the input - only certain input and output shapes are therefore possible (see getpadding function)
    '''
    #根据模型配置初始化模型，大部分是一些卷积核的大小和上下采样的层数，包括通道数，输入输出的卷积核大小，激活函数的方法等
    def __init__(self, model_config):
        '''
        Initialize U-net
        :param num_layers: Number of down- and upscaling layers in the network 
        '''
        self.num_layers = model_config["num_layers"]
        self.num_initial_filters = model_config["num_initial_filters"]
        self.filter_size = model_config["filter_size"]
        self.merge_filter_size = model_config["merge_filter_size"]
        self.input_filter_size = model_config["input_filter_size"]
        self.output_filter_size = model_config["output_filter_size"]
        self.upsampling = model_config["upsampling"]
        self.output_type = model_config["output_type"]
        self.context = model_config["context"]
        self.padding = "valid" if model_config["context"] else "same"
        self.source_names = model_config["source_names"]
        self.num_channels = 1 if model_config["mono_downmix"] else 2
        self.output_activation = model_config["output_activation"]

    def get_padding(self, shape):
        '''
        计算沿输入和输出的每个轴所需的填充量，以便Unet工作并将给定的形状作为输出形状，保证输入和输出的大小是一致的
        Calculates the required amounts of padding along each axis of the input and output, so that the Unet works and has the given shape as output shape
        :param shape: Desired output shape （描述输出的大小
        :return: Input_shape, output_shape, where each is a list [batch_size, time_steps, channels]

        '''

        if self.context:
            # Check if desired shape is possible as output shape - go from output shape towards lowest-res feature map
            # 检查是否需要的形状是可能的输出形状-从输出形状到最低分辨率的特征图
            rem = float(shape[1]) # Cut off batch size number and channel 切掉batch尺寸和通道的配置

            # Output filter size
            rem = rem - self.output_filter_size + 1 #输出过滤器（卷积核大小）

            # Upsampling blocks 设置上采样块
            for i in range(self.num_layers):
                # 加上下采样部分合并模块的大小
                rem = rem + self.merge_filter_size - 1
                #上采样所关注的特征不断缩小，缩小至原来的1/2
                rem = (rem + 1.) / 2.# out = in + in - 1 <=> in = (out+1)/

            # Round resulting feature map dimensions up to nearest integer（特征图上的数据进行取整）
            x = np.asarray(np.ceil(rem),dtype=np.int64)
            assert(x >= 2)

            # Compute input and output shapes based on lowest-res feature map 上采样得到的是地接特征，接下来计算特征的形状
            output_shape = x
            input_shape = x

            # Extra conv 这里为啥要加上卷积核的大小。。
            input_shape = input_shape + self.filter_size - 1

            # Go from centre feature map through up- and downsampling blocks，上采样之后就是下采样了，从特征图中心开始进行扩散式的下采样
            for i in range(self.num_layers):
                output_shape = 2*output_shape - 1 #Upsampling，下采样即范围扩大了一倍后
                output_shape = output_shape - self.merge_filter_size + 1 # Conv 通过卷积后的特征图变化即为 原特征维度 - 卷积核的大小 + 1 表示特征的维度

                input_shape = 2*input_shape - 1 # Decimation 之前通过上采样获得的inputshape即x的大小这里也是需要进行变化的 这一步我也直接迷惑
                if i < self.num_layers - 1: #最后一层是输出层，没到最后一层都要进行一个简单的卷积，特征的提取实际上是
                    input_shape = input_shape + self.filter_size - 1 # Conv
                else:   #如果是最后一层，（为啥还要加上input_filter_size
                    input_shape = input_shape + self.input_filter_size - 1

            # Output filters
            output_shape = output_shape - self.output_filter_size + 1
            #np对数组进行拼接拼接的三个元素分别是shape[0],[输入的特征向量]，[通道数]
            input_shape = np.concatenate([[shape[0]], [input_shape], [self.num_channels]])
            output_shape = np.concatenate([[shape[0]], [output_shape], [self.num_channels]])
            #将提取的张量进行返回
            return input_shape, output_shape
        else:
            return [shape[0], shape[1], self.num_channels], [shape[0], shape[1], self.num_channels]

    def get_output(self, input, training, return_spectrogram=False, reuse=True):
        '''
        Creates symbolic computation graph of the U-Net for a given input batch
        :param input: Input batch of mixtures, 3D tensor [batch_size, num_samples, num_channels]
        :param reuse: Whether to create new parameter variables or reuse existing ones
        :return: U-Net output: List of source estimates. Each item is a 3D tensor [batch_size, num_out_samples, num_channels]
        :返回值:U-Net输出:源估计列表。每个项目是一个三维的张量[batch_size, num_out_samples, num_channels]
        '''
        with tf.variable_scope("separator", reuse=reuse):
            enc_outputs = list()
            current_layer = input

            # Down-convolution: Repeat strided conv 向下卷积，重复的步长卷积，卷积核的大小在不断的变化，下采样
            for i in range(self.num_layers):
                '''
                current_layer:输入
                filters：卷积核的数目，这里指的是每一层会添加新的卷积核
                self.filter_size： 卷积核的大小
                剩下来的就是步长和激活函数了
                '''
                current_layer = tf.layers.conv1d(current_layer, self.num_initial_filters + (self.num_initial_filters * i), self.filter_size, strides=1, activation=LeakyReLU, padding=self.padding) # out = in - filter + 1
                #卷积结果加入,这个地方是为了方便保存结果之后与上采样中的低阶特征向量进行一个裁剪和链接
                enc_outputs.append(current_layer)
                #下采样舍弃的原则
                current_layer = current_layer[:,::2,:] # Decimate by factor of 2 # out = (in-1)/2 + 1
            #最后一层的输出
            current_layer = tf.layers.conv1d(current_layer, self.num_initial_filters + (self.num_initial_filters * self.num_layers),self.filter_size,activation=LeakyReLU,padding=self.padding) # One more conv here since we need to compute features after last decimation

            # Feature map here shall be X along one dimension

            # Upconvolution 向上卷积
            for i in range(self.num_layers):
                '''
                因为Unet的结构是先进行上卷积然后再进行下卷积的结构，所以在进行下卷积之后就要进行上卷积了
                '''
                #UPSAMPLING
                #在输入的特征的第二维上增加一个填充？没有懂这一步
                current_layer = tf.expand_dims(current_layer, axis=1)
                if self.upsampling == 'learned':#这里是要进行线性插值的
                    # Learned interpolation between two neighbouring time positions by using a convolution filter of width 2, and inserting the responses in the middle of the two respective inputs
                    #调用之前的方法进行线性插值
                    current_layer = Models.InterpolationLayer.learned_interpolation_layer(current_layer, self.padding, i)
                else:
                    #这边应该是另一个插值的方法
                    if self.context:
                        current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2] * 2 - 1], align_corners=True)
                    else:
                        current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2]*2]) # out = in + in - 1
                #删除维度是1的张量
                current_layer = tf.squeeze(current_layer, axis=1)
                # UPSAMPLING FINISHED

                assert(enc_outputs[-i-1].get_shape().as_list()[1] == current_layer.get_shape().as_list()[1] or self.context) #No cropping should be necessary unless we are using context
                #向量的裁剪和链接，链接的对象是之前下采样产生的高阶特征向量
                current_layer = Utils.crop_and_concat(enc_outputs[-i-1], current_layer, match_feature_dim=False)
                #对其进行一维卷积
                current_layer = tf.layers.conv1d(current_layer, self.num_initial_filters + (self.num_initial_filters * (self.num_layers - i - 1)), self.merge_filter_size,
                                                 activation=LeakyReLU,
                                                 padding=self.padding)  # out = in - filter + 1
            #最终的结果与input对齐并且连接
            current_layer = Utils.crop_and_concat(input, current_layer, match_feature_dim=False)

            # Output layer
            # Determine output activation function
            if self.output_activation == "tanh":
                out_activation = tf.tanh
            elif self.output_activation == "linear":
                out_activation = lambda x: Utils.AudioClip(x, training)
            else:
                raise NotImplementedError

            if self.output_type == "direct":
                #这边返回的是输出层的一个
                return Models.OutputLayer.independent_outputs(current_layer, self.source_names, self.num_channels, self.output_filter_size, self.padding, out_activation)
            elif self.output_type == "difference":
                #如果选项是difference则说明需要将输入的向量和最终的输出的向量进行对齐
                cropped_input = Utils.crop(input,current_layer.get_shape().as_list(), match_feature_dim=False)
                return Models.OutputLayer.difference_output(cropped_input, current_layer, self.source_names, self.num_channels, self.output_filter_size, self.padding, out_activation, training)
            else:
                raise NotImplementedError