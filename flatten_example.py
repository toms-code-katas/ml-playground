import tensorflow as tf

if __name__ == '__main__':
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=(3, 3, 1)))
    model.add(tf.keras.layers.Flatten())

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    print(model.input_shape)
    print(model.output_shape)

    for layer in model.layers:
        print(f"Layer {layer.name}, input_shape={layer.input_shape}, output_shape={layer.output_shape}")

