import tensorflow as tf


def convlayerBMA(x, filters, size, apply_batchnorm=True):
    x = tf.keras.layers.Conv2D(filters, size, strides=1, padding='same', use_bias=True)(x)

    if apply_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    return x


def upsampleBMA(x, filters, size, basic=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    if basic:
        x = tf.keras.layers.UpSampling2D()(x)
    else:
        x = tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def downsampleBMA(x, filters, size, apply_batchnorm=True, basic=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    if basic:
        x = tf.keras.layers.MaxPool2D()(x)
    else:
        x = tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=True)(x)

    if apply_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def ReturnUNet(x, size=4, layers=7, filters_start=64, double_layers=4):
    """
    default values creates the original generator, filters double from start
    to a max after the number of 'double layers'
    Size is the kernel size
    Layers is the number of layers
    """
    filters_list = []
    for _ in range(double_layers):
        filters_list.append(filters_start)
        filters_start *= 2
    up_filters = []
    batchnorm = False
    down_stack = []
    for _ in range(layers):
        if filters_list:
            filters = filters_list.pop(0)
        up_filters.append(filters)
        x = downsampleBMA(x, filters, size, batchnorm)
        batchnorm = True
        down_stack.append(x)
    x = downsampleBMA(x, filters, size, batchnorm)
    for _ in range(layers):
        filters = up_filters.pop()
        skip = down_stack.pop()
        x = upsampleBMA(x, filters, size)
        x = tf.keras.layers.Concatenate()([x, skip])
    return x


def resize_tensor(x, wanted_distance=1000, acquired_distance=1540):
    output_size = int(x.shape[1]//2*wanted_distance/acquired_distance*2)
    current_size = x.shape[1]
    x = tf.image.resize(x, [output_size, output_size])
    if wanted_distance < acquired_distance:
        x = tf.image.pad_to_bounding_box(x, (current_size - output_size) // 2,
                                         (current_size - output_size) // 2, current_size, current_size)
    else:
        x = x[:, (output_size-current_size)//2:-(output_size-current_size)//2,
            (output_size-current_size)//2:-(output_size-current_size)//2]
    return x


def GeneratorBMA2(top_layers=2, size=4, layers=7, filters_start=64, double_layers=4, add_unet=False, max_filters=64):
    """
    default values creates the original generator, filters double from start
    to a max after the number of 'double layers'
    Size is the kernel size
    Layers is the number of layers
    """
    """
    Back to basic physics
    """
    inputs = x = tf.keras.layers.Input(shape=[256, 256, 5])
    PDOS = tf.expand_dims(inputs[..., 0], axis=-1, name='PDOS')
    fulldrr = tf.expand_dims(inputs[..., 1], axis=-1)
    deep_to_panel = tf.expand_dims(inputs[..., 2], axis=-1)
    iso_to_panel = tf.expand_dims(inputs[..., 3], axis=-1)
    drr_shallow_to_panel = tf.expand_dims(inputs[..., 4], axis=-1)

    fluence_shallow_to_panel = tf.math.exp(-convlayerBMA(drr_shallow_to_panel, filters_start, size)) * PDOS
    # fluence_shallow_to_panel = resize_tensor(fluence_shallow, wanted_distance=1540, acquired_distance=950)
    # fluence_to_iso = resize_tensor(fluence_shallow, wanted_distance=1000, acquired_distance=950)

    # drr_dif_iso_and_shallow = tf.keras.layers.ReLU()(iso_drr - resize_tensor(drr_shallow, wanted_distance=1000, acquired_distance=950))
    fluence_iso_to_panel = tf.math.exp(-convlayerBMA(iso_to_panel, filters_start, size)) * PDOS
    # fluence_iso_to_panel = resize_tensor(fluence_iso, wanted_distance=1540, acquired_distance=1000)
    # fluence_to_deeper = resize_tensor(fluence_iso, wanted_distance=1050, acquired_distance=1000)
    # drr_dif_deeper_and_iso = tf.keras.layers.ReLU()(drr_deep - resize_tensor(iso_drr, wanted_distance=1050, acquired_distance=1000))
    fluence_deep_to_panel = tf.math.exp(-convlayerBMA(deep_to_panel, filters_start, size)) * PDOS
    # fluence_deeper_to_panel = resize_tensor(fluence_deeper, wanted_distance=1540, acquired_distance=1050)

    # drr_dif_panel_and_deeper = tf.keras.layers.ReLU()(fulldrr - resize_tensor(drr_deep, wanted_distance=1540, acquired_distance=1050))
    fluence_panel = tf.math.exp(-convlayerBMA(fulldrr, filters_start, size)) * PDOS

    x = tf.keras.layers.Concatenate() \
        ([fluence_shallow_to_panel, fluence_iso_to_panel, fluence_deep_to_panel, fluence_panel])
    filters_start *= 4
    x = convlayerBMA(x, min([filters_start, max_filters]), size)
    for _ in range(top_layers):
        filters_start *= 2
        x = convlayerBMA(x, min([filters_start, max_filters]), size)
    if add_unet:
        base_out = x
        x = ReturnUNet(x=x, size=size, layers=layers, filters_start=min([filters_start, max_filters]), double_layers=double_layers)
        x = tf.keras.layers.Concatenate()([x, base_out])
    x = tf.keras.layers.Concatenate()([x, fluence_panel])
    x = convlayerBMA(x, min([filters_start, max_filters]), size)
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation=None,
                               padding='same', use_bias=True)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


if __name__ == '__main__':
    pass
