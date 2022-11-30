import logging
import tensorflow as tf

L1_REG = 0.0
L2_REG = 1e-4


def conv2d_block(input_tensor, n_filters, kernel_size=3):
    x = input_tensor

    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=n_filters,
                                   kernel_size=(kernel_size, kernel_size),
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=tf.keras.initializers.Orthogonal(),
                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1_REG, l2=L2_REG))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

    return x


def encoder_block(inputs, n_filters, pool_size, dropout=None):
    f = conv2d_block(inputs, n_filters=n_filters)
    p = tf.keras.layers.MaxPooling2D(pool_size)(f)

    if dropout is not None:
        p = tf.keras.layers.Dropout(dropout)(p)

    return f, p


def encoder(inputs, dropout=None):

    f1, p1 = encoder_block(inputs, n_filters=48, pool_size=(2, 2), dropout=dropout)
    f2, p2 = encoder_block(p1, n_filters=64, pool_size=(2, 2), dropout=dropout)
    f3, p3 = encoder_block(p2, n_filters=64, pool_size=(2, 2), dropout=dropout)
    f4, p4 = encoder_block(p3, n_filters=64, pool_size=(2, 2), dropout=dropout)

    return p4, (f1, f2, f3, f4)


def decoder_block(inputs, conv_output, n_filters, dropout=None):

    c = conv2d_block(inputs, n_filters, kernel_size=3)

    u = tf.keras.layers.UpSampling2D(size=(2, 2))(c)
    c = tf.keras.layers.concatenate([u, conv_output])

    if dropout is not None:
        c = tf.keras.layers.Dropout(dropout)(c)

    return c


def decoder(inputs, convs, dropout=None):
    f1, f2, f3, f4 = convs

    c6 = decoder_block(inputs, f4, n_filters=96, dropout=dropout)
    c7 = decoder_block(c6, f3, n_filters=64, dropout=dropout)
    c8 = decoder_block(c7, f2, n_filters=64, dropout=dropout)

    # output block
    c9 = conv2d_block(c8, n_filters=64, kernel_size=3)

    return c9


def det_net_1(inputs, dropout=None):

    if dropout is not None:
        logging.info(f'using dropout value={dropout}')

    encoder_output, convs = encoder(inputs, dropout=dropout)

    outputs = decoder(encoder_output, convs, dropout=dropout)
    density_pred = tf.keras.layers.Conv2D(filters=1,
                                          kernel_size=(1, 1),
                                          use_bias=False,
                                          activation=tf.keras.activations.linear,
                                          kernel_initializer=tf.keras.initializers.Orthogonal(),
                                          padding='same',
                                          name='density_pred')(outputs)

    return density_pred
