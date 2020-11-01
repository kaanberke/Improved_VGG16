import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# plt.imshow(X_train[np.random.randint(X_train.shape[0])], cmap='Greys', interpolation="nearest")
# plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
               'Ankle boot']

img_rows = X_train[0].shape[0]
img_cols = X_test[0].shape[1]

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

input_shape = (img_rows, img_cols, 1)

backbone_input = tf.keras.layers.Input(shape=(47, 47, 1))  # [None, 47, 47, 1]
backbone_conv1 = tf.keras.layers.Conv2D(30, (3, 3), padding="valid", activation="relu")(
    backbone_input)  # [None, 45, 45, 30]
backbone_conv2 = tf.keras.layers.Conv2D(30, (3, 3), padding="valid", activation="relu")(
    backbone_conv1)  # [None, 43, 43, 30]
backbone_conv3 = tf.keras.layers.Conv2D(30, (3, 3), padding="valid", activation="relu", dilation_rate=2)(
    backbone_conv2)  # [None, 39, 39, 30]
backbone_conv4 = tf.keras.layers.Conv2D(40, (3, 3), padding="valid", activation="relu")(
    backbone_conv3)  # [None, 37, 37, 40]

pyramid_lower_conv1 = tf.keras.layers.Conv2D(50, (3, 3), strides=1, padding="valid", activation="relu",
                                             dilation_rate=11)(backbone_conv4)  # [None, 15, 15, 50]
# pyramid_lower_upsampling1 = tf.keras.layers.UpSampling2D()(pyramid_lower_conv1)  # [None, 30, 30, 50]
pyramid_lower_padding = tf.keras.layers.ZeroPadding2D((4, 4))(pyramid_lower_conv1)

pyramid_middle_conv1 = tf.keras.layers.Conv2D(50, (3, 3), strides=1, padding="valid", activation="relu",
                                              dilation_rate=7)(backbone_conv4)  # [None, 23, 23, 50]
pyramid_middle_conc = tf.keras.layers.Concatenate()([pyramid_middle_conv1, pyramid_lower_padding])
# pyramid_middle_upsampling1 = tf.keras.layers.UpSampling2D()(pyramid_middle_conc)
pyramid_middle_padding = tf.keras.layers.ZeroPadding2D((4, 4))(pyramid_middle_conc)

pyramid_upper_conv1 = tf.keras.layers.Conv2D(50, (3, 3), strides=1, padding="valid", activation="relu",
                                             dilation_rate=3)(backbone_conv4)  # [None, 31, 31, 50]
pyramid_upper_conc = tf.keras.layers.Concatenate()([pyramid_upper_conv1, pyramid_middle_padding])

clf_layer_conv1 = tf.keras.layers.Conv2D(150, (1, 1), padding="valid", activation="relu")(pyramid_upper_conc)
clf_layer_conv2 = tf.keras.layers.Conv2D(150, (1, 1), padding="valid", activation="relu")(clf_layer_conv1)
clf_layer_conv3 = tf.keras.layers.Conv2D(5, (1, 1), padding="valid", activation="relu")(clf_layer_conv2)

fc_layer = tf.keras.layers.Flatten()(clf_layer_conv2)
softmax_layer = tf.keras.layers.Dense(10, activation="softmax")(fc_layer)

model = tf.keras.models.Model(inputs=[backbone_input], outputs=[softmax_layer])

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('best.h5', verbose=1, monitor='val_accuracy',
                                                         save_best_only=True, mode='auto')

sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy",
              optimizer=sgd,
              metrics=["accuracy"],
              )
model.summary()

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
)
train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=64,
)
validation_generator = test_datagen.flow(
    X_test, y_test,
    batch_size=64,
)

model.fit(
    train_generator,
    # steps_per_epoch=200,
    epochs=100,
    validation_data=validation_generator,
    # validation_steps=40,
    verbose=1,
    callbacks=[
        checkpoint_callback,
        early_stopping_callback
    ]
)
