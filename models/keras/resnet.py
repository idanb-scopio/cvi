import tensorflow as tf

L1_REG = 5e-7
L2_REG = 4e-4


def conv_op(x, filters, kernel_size, strides, name):
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding='same',
                               name=name,
                               kernel_initializer=tf.keras.initializers.Orthogonal(),
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1_REG, l2=L2_REG))(x)
    return x


def identity_block(x, kernel_size, filters, stage, block):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    f1, f2 = filters

    x_shortcut = x

    x = conv_op(x, filters=f1, kernel_size=kernel_size, strides=(1, 1), name=conv_name_base + '2a')
    x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = conv_op(x, filters=f2, kernel_size=kernel_size, strides=(1, 1), name=conv_name_base + '2b')
    x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2b')(x)

    x = tf.keras.layers.Add()([x, x_shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x


def conv_block(x, kernel_size, filters, stage, block, s=2):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    f1, f2 = filters

    x_shortcut = x
    x = conv_op(x, filters=f1, kernel_size=kernel_size, strides=(s, s), name=conv_name_base + '2a')
    x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = conv_op(x, filters=f1, kernel_size=kernel_size, strides=(1, 1), name=conv_name_base + '2b')
    x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2b')(x)

    x_shortcut = conv_op(x_shortcut, filters=f2, kernel_size=1, strides=(s, s), name=conv_name_base + '1')
    x_shortcut = tf.keras.layers.BatchNormalization(name=bn_name_base + '1')(x_shortcut)

    x = tf.keras.layers.Add()([x, x_shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x


def res_net(inputs, num_classes):

    x = conv_op(inputs, filters=64, kernel_size=5, strides=(1, 1), name='conv_1')
    x = tf.keras.layers.BatchNormalization(name='bn_conv1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block(x, kernel_size=3, filters=(64, 64), stage=2, block='a', s=1)

    x = conv_block(x, kernel_size=3, filters=(80, 80), stage=3, block='a', s=2)
    x = identity_block(x, kernel_size=3, filters=(80, 80), stage=3, block='b')

    filters = (100, 128)
    for i in range(2):
        x = conv_block(x, kernel_size=3, filters=(filters[i], filters[i]), stage=4+i, block='a', s=2)
        x = identity_block(x, kernel_size=3, filters=(filters[i], filters[i]), stage=4+i, block='b')
        x = identity_block(x, kernel_size=3, filters=(filters[i], filters[i]), stage=4+i, block='c')

    x = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    features = tf.keras.layers.Flatten()(x)
    probabilities = tf.keras.layers.Dense(num_classes,
                                          activation='softmax',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1_REG, l2=L2_REG),
                                          name='probabilities')(features)

    return features, probabilities
