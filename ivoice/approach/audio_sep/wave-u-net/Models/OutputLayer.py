import tensorflow as tf

import Utils

'''
featureMap就是经过U-Net U型结构处理的特征
sourceName是根据任务来的，如果是分离人声和音乐声，source就为2，如果是分离各种乐器，就是为"bass", "drums", "other", "vocals"这样一个数组
'''
def independent_outputs(featuremap, source_names, num_channels, filter_width, padding, activation):
    outputs = dict()
    for name in source_names:
        # 这里返回的是一个向量吧。。不是很能理解
        # 现在大这里大概是通过不同的声源，对特征进行一个标记
        outputs[name] = tf.layers.conv1d(featuremap, num_channels, filter_width, activation=activation, padding=padding)
    return outputs

def difference_output(input_mix, featuremap, source_names, num_channels, filter_width, padding, activation, training):
    outputs = dict()
    sum_source = 0
    for name in source_names[:-1]:
        #out是一个经过了一维卷积的向量
        out = tf.layers.conv1d(featuremap, num_channels, filter_width, activation=activation, padding=padding)
        outputs[name] = out
        #向量和是对原有音源的重新再现
        sum_source = sum_source + out

    # Compute last source based on the others
    # 计算最后一维的特征并且返回对应的所有的特征输出
    last_source = Utils.crop(input_mix, sum_source.get_shape().as_list()) - sum_source
    last_source = Utils.AudioClip(last_source, training)
    outputs[source_names[-1]] = last_source
    return outputs