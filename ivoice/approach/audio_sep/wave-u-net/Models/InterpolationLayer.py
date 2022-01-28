import tensorflow as tf

'''该类主要是负责实现在插值的方式，在上采样的过程中需要进行论文中所所述的线性插值
'''
def learned_interpolation_layer(input, padding, level):
    '''
    Implements a trainable upsampling layer by interpolation by a factor of two, from N samples to N*2 - 1.
    Interpolation of intermediate feature vectors v_1 and v_2 (of dimensionality F) is performed by
     w \cdot v_1 + (1-w) \cdot v_2, where \cdot is point-wise multiplication, and w an F-dimensional weight vector constrained to [0,1]
    :param input: Input features of shape [batch_size, 1, width, F]
    :param padding:
    :param level:
    :return:
    '''
    assert(padding == "valid" or padding == "same")
    features = input.get_shape().as_list()[3]

    # Construct 2FxF weight matrix, where F is the number of feature channels in the feature map.

    #首先需要构造一个2F*F的的权重矩阵，其中的F是特征map中的特征通道的数量。
    #矩阵是受约束的，由两个对角权值为w和1-w的对角FxF矩阵组成。其中W的值的范围被限制在【0,1】之中。
    # Matrix is constrained, made up out of two diagonal FxF matrices with diagonal weights w and 1-w. w is constrained to be in [0,1] # mioid

    weights = tf.get_variable("interp_" + str(level), shape=[features], dtype=tf.float32)
    # 激活函数范围限定在0-1
    weights_scaled = tf.nn.sigmoid(weights) # Constrain weights to [0,1]
    # 设置权重矩阵的另外一半1-w的对角矩阵
    counter_weights = 1.0 - weights_scaled # Mirrored weights for the features from the other time step
    # 连接两个权重矩阵
    conv_weights = tf.expand_dims(tf.concat([tf.expand_dims(tf.diag(weights_scaled), axis=0), tf.expand_dims(tf.diag(counter_weights), axis=0)], axis=0), axis=0)
    #卷积操作，第一个输入值是需要进行卷积的输入
    # 第二个输入值是卷积核[ filter_height, filter_width, in_channel, out_channels ]，
    # 以上lter_height 为卷积核高度，filter_width 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
    # 第三个输入值是卷积在每一维的步长（第一位和最后一位固定是1）
    # 第四个值是padding表示是否考虑边界，这个在上面提到过，same表示考虑边界，而valid表示不考虑边界，上面是有断言的，这里是true（应该
    intermediate_vals = tf.nn.conv2d(input, conv_weights, strides=[1,1,1,1], padding=padding.upper())
    #高阶的一个转置操作，分别对中间值和输出进行一个转置操作，方便之后的全连接
    intermediate_vals = tf.transpose(intermediate_vals, [2, 0, 1, 3])
    out = tf.transpose(input, [2, 0, 1, 3])
    #返回一个元组并且同时转为一个列表取第一个元素，这个地方就是简单的统计一下数量，N个sample
    num_entries = out.get_shape().as_list()[0]
    #全连接
    out = tf.concat([out, intermediate_vals], axis=0)
    indices = list()

    # Interleave interpolated features with original ones, starting with the first original one
    #分为两种情况，考虑边界值和不考虑边界值
    num_outputs = (2*num_entries - 1) if padding == "valid" else 2*num_entries
    for idx in range(num_outputs):
        if idx % 2 == 0:
            indices.append(idx // 2)
        else:
            indices.append(num_entries + idx//2)
    #插值结果与输出
    out = tf.gather(out, indices)
    #四维重新转置一下
    current_layer = tf.transpose(out, [1, 2, 0, 3])
    return current_layer