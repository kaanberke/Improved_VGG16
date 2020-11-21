# %% Libraries

import tensorflow as tf
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %% Data Preparation

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
)
train_generator = train_datagen.flow_from_directory(
    directory="./data/train/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=4,
    class_mode="categorical",
    shuffle=True,
    # seed=42

)
validation_generator = test_datagen.flow_from_directory(
    directory="./data/test/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=4,
    class_mode="categorical",
    shuffle=True,
    # seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory="./data/val/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    # seed=42
)


#%% Common parts for all models
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


sgd = tf.keras.optimizers.SGD()
learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

#%% VGG-16 Base model
input_layer = tf.keras.layers.Input((224, 224, 3))

model = tf.keras.applications.vgg16.VGG16(include_top=False, input_tensor=input_layer)
for layer in model.layers:
    layer.trainable = False

flat1 = tf.keras.layers.Flatten()(model.layers[-1].output)
fc_layer1 = tf.keras.layers.Dense(1024, activation='relu')(flat1)
fc_layer2 = tf.keras.layers.Dense(1024, activation='relu')(fc_layer1)
output = tf.keras.layers.Dense(2, activation='softmax')(fc_layer2)

model = tf.keras.models.Model(inputs=model.inputs, outputs=output)
model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer=sgd,
              metrics=["accuracy"],
              )

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('block2_model_pyramid.h5', verbose=1, monitor='val_loss',
                                                         save_best_only=True, mode='auto')

history = model.fit(
    train_generator,
    epochs=500,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
        learning_rate_scheduler
    ]
)

# %% Pyramid blocks

# 2nd Block
input_layer = tf.keras.layers.Input((224, 224, 3))

model = tf.keras.applications.vgg16.VGG16(include_top=False, input_tensor=input_layer)

for layer in model.layers:
    layer.trainable = False
block2_conv2 = model.get_layer("block2_conv2")

pyramid_lower_conv1 = tf.keras.layers.Conv2D(50, (3, 3), strides=1, padding="valid", activation="relu",
                                             dilation_rate=11)(block2_conv2.output)
pyramid_lower_padding = tf.keras.layers.ZeroPadding2D((4, 4))(pyramid_lower_conv1)

pyramid_middle_conv1 = tf.keras.layers.Conv2D(50, (3, 3), strides=1, padding="valid", activation="relu",
                                              dilation_rate=7)(block2_conv2.output)
pyramid_middle_conc = tf.keras.layers.Concatenate()(
    [pyramid_middle_conv1, pyramid_lower_padding])
pyramid_middle_padding = tf.keras.layers.ZeroPadding2D((4, 4))(pyramid_middle_conc)

pyramid_upper_conv1 = tf.keras.layers.Conv2D(50, (3, 3), strides=1, padding="valid", activation="relu",
                                             dilation_rate=3)(block2_conv2.output)
pyramid_upper_conc = tf.keras.layers.Concatenate()([pyramid_upper_conv1, pyramid_middle_padding])

clf_layer_conv1 = tf.keras.layers.Conv2D(150, (1, 1), padding="valid", activation="relu")(
    pyramid_upper_conc)
clf_layer_conv2 = tf.keras.layers.Conv2D(150, (1, 1), padding="valid", activation="relu")(
    clf_layer_conv1)
clf_layer_conv3 = tf.keras.layers.Conv2D(5, (1, 1), padding="valid", activation="relu")(
    clf_layer_conv2)

block2_avgpool = tf.keras.layers.GlobalAveragePooling2D()(clf_layer_conv3)
block2_norm = tf.keras.layers.BatchNormalization()(block2_avgpool)
block2_flatten = tf.keras.layers.Flatten()(block2_norm)
block2_fc = tf.keras.layers.Dense(128, activation='relu')(block2_flatten)
block2_output = tf.keras.layers.Dense(2, activation='softmax')(block2_fc)
block2_model = tf.keras.models.Model(inputs=[input_layer], outputs=[block2_output])
block2_model.summary()

block2_model.compile(loss="categorical_crossentropy",
                     optimizer=sgd,
                     metrics=["accuracy"],
                     )

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('block2_model_pyramid.h5', verbose=1, monitor='val_loss',
                                                         save_best_only=True, mode='auto')

history = block2_model.fit(
    train_generator,
    epochs=500,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
        learning_rate_scheduler
    ]
)

#%% 3rd Block
input_layer = tf.keras.layers.Input((224, 224, 3))

model = tf.keras.applications.vgg16.VGG16(include_top=False, input_tensor=input_layer)

for layer in model.layers:
    layer.trainable = False

block3_conv3 = model.get_layer("block3_conv3")

pyramid_lower_conv1 = tf.keras.layers.Conv2D(50, (3, 3), strides=1, padding="valid", activation="relu",
                                             dilation_rate=11)(block3_conv3.output)
pyramid_lower_padding = tf.keras.layers.ZeroPadding2D((4, 4))(pyramid_lower_conv1)

pyramid_middle_conv1 = tf.keras.layers.Conv2D(50, (3, 3), strides=1, padding="valid", activation="relu",
                                              dilation_rate=7)(block3_conv3.output)
pyramid_middle_conc = tf.keras.layers.Concatenate()(
    [pyramid_middle_conv1, pyramid_lower_padding])
pyramid_middle_padding = tf.keras.layers.ZeroPadding2D((4, 4))(pyramid_middle_conc)

pyramid_upper_conv1 = tf.keras.layers.Conv2D(50, (3, 3), strides=1, padding="valid", activation="relu",
                                             dilation_rate=3)(block3_conv3.output)
pyramid_upper_conc = tf.keras.layers.Concatenate()([pyramid_upper_conv1, pyramid_middle_padding])

clf_layer_conv1 = tf.keras.layers.Conv2D(150, (1, 1), padding="valid", activation="relu")(
    pyramid_upper_conc)
clf_layer_conv2 = tf.keras.layers.Conv2D(150, (1, 1), padding="valid", activation="relu")(
    clf_layer_conv1)
clf_layer_conv3 = tf.keras.layers.Conv2D(5, (1, 1), padding="valid", activation="relu")(
    clf_layer_conv2)

block3_avgpool = tf.keras.layers.GlobalAveragePooling2D()(clf_layer_conv3)
block3_norm = tf.keras.layers.BatchNormalization()(block3_avgpool)
block3_flatten = tf.keras.layers.Flatten()(block3_norm)
block3_fc = tf.keras.layers.Dense(128, activation='relu')(block3_flatten)
block3_output = tf.keras.layers.Dense(2, activation='softmax')(block3_fc)
block3_model = tf.keras.models.Model(inputs=[input_layer], outputs=[block3_output])
block3_model.summary()

block3_model.compile(loss="categorical_crossentropy",
                     optimizer=sgd,
                     metrics=["accuracy"],
                     )

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('block2_model_pyramid.h5', verbose=1, monitor='val_loss',
                                                         save_best_only=True, mode='auto')

history = block3_model.fit(
    train_generator,
    epochs=500,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
        learning_rate_scheduler
    ]
)

# 4th Block
input_layer = tf.keras.layers.Input((224, 224, 3))

model = tf.keras.applications.vgg16.VGG16(include_top=False, input_tensor=input_layer)

for layer in model.layers:
    layer.trainable = False

block4_conv2 = model.get_layer("block4_conv3")

pyramid_lower_conv1 = tf.keras.layers.Conv2D(50, (3, 3), strides=1, padding="valid", activation="relu",
                                             dilation_rate=11)(block4_conv2.output)
pyramid_lower_padding = tf.keras.layers.ZeroPadding2D((4, 4))(pyramid_lower_conv1)

pyramid_middle_conv1 = tf.keras.layers.Conv2D(50, (3, 3), strides=1, padding="valid", activation="relu",
                                              dilation_rate=7)(block4_conv2.output)
pyramid_middle_conc = tf.keras.layers.Concatenate()(
    [pyramid_middle_conv1, pyramid_lower_padding])
pyramid_middle_padding = tf.keras.layers.ZeroPadding2D((4, 4))(pyramid_middle_conc)

pyramid_upper_conv1 = tf.keras.layers.Conv2D(50, (3, 3), strides=1, padding="valid", activation="relu",
                                             dilation_rate=3)(block4_conv2.output)
pyramid_upper_conc = tf.keras.layers.Concatenate()([pyramid_upper_conv1, pyramid_middle_padding])

clf_layer_conv1 = tf.keras.layers.Conv2D(150, (1, 1), padding="valid", activation="relu")(
    pyramid_upper_conc)
clf_layer_conv2 = tf.keras.layers.Conv2D(150, (1, 1), padding="valid", activation="relu")(
    clf_layer_conv1)
clf_layer_conv3 = tf.keras.layers.Conv2D(5, (1, 1), padding="valid", activation="relu")(
    clf_layer_conv2)

block4_avgpool = tf.keras.layers.GlobalAvgPool2D()(clf_layer_conv3)
block4_norm = tf.keras.layers.BatchNormalization()(block4_avgpool)
block4_flatten = tf.keras.layers.Flatten()(block4_norm)
block4_fc = tf.keras.layers.Dense(128, activation='relu')(block4_flatten)
block4_output = tf.keras.layers.Dense(2, activation='softmax')(block4_fc)
block4_model = tf.keras.models.Model(inputs=[input_layer], outputs=[block4_output])
block4_model.summary()

block4_model.compile(loss="categorical_crossentropy",
                     optimizer=sgd,
                     metrics=["accuracy"],
                     )

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('block2_model_pyramid.h5', verbose=1, monitor='val_loss',
                                                         save_best_only=True, mode='auto')

history = block4_model.fit(
    train_generator,
    epochs=500,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
        learning_rate_scheduler
    ]
)

# 5th (LAST) Block

input_layer = tf.keras.layers.Input((224, 224, 3))

model = tf.keras.applications.vgg16.VGG16(include_top=False, input_tensor=input_layer)

for layer in model.layers:
    layer.trainable = False

block5_conv3 = model.get_layer("block5_conv3")

pyramid_lower_conv1 = tf.keras.layers.Conv2D(50, (3, 3), strides=1, padding="valid", activation="relu",
                                             dilation_rate=5)(block5_conv3.output)
pyramid_lower_padding = tf.keras.layers.ZeroPadding2D((2, 2))(pyramid_lower_conv1)

pyramid_middle_conv1 = tf.keras.layers.Conv2D(50, (3, 3), strides=1, padding="valid", activation="relu",
                                              dilation_rate=3)(block5_conv3.output)
pyramid_middle_conc = tf.keras.layers.Concatenate()(
    [pyramid_middle_conv1, pyramid_lower_padding])
pyramid_middle_padding = tf.keras.layers.ZeroPadding2D((2, 2))(pyramid_middle_conc)

pyramid_upper_conv1 = tf.keras.layers.Conv2D(50, (3, 3), strides=1, padding="valid", activation="relu",
                                             dilation_rate=1)(block5_conv3.output)
pyramid_upper_conc = tf.keras.layers.Concatenate()([pyramid_upper_conv1, pyramid_middle_padding])

clf_layer_conv1 = tf.keras.layers.Conv2D(150, (1, 1), padding="valid", activation="relu")(
    pyramid_upper_conc)
clf_layer_conv2 = tf.keras.layers.Conv2D(150, (1, 1), padding="valid", activation="relu")(
    clf_layer_conv1)
clf_layer_conv3 = tf.keras.layers.Conv2D(5, (1, 1), padding="valid", activation="relu")(
    clf_layer_conv2)

block5_avgpool = tf.keras.layers.GlobalAvgPool2D()(clf_layer_conv3)
block5_norm = tf.keras.layers.BatchNormalization()(block5_avgpool)
block5_flatten = tf.keras.layers.Flatten()(block5_norm)
block5_fc = tf.keras.layers.Dense(128, activation='relu')(block5_flatten)
block5_output = tf.keras.layers.Dense(2, activation='softmax')(block5_fc)
block5_model = tf.keras.models.Model(inputs=[input_layer], outputs=[block5_output])
block5_model.summary()

block5_model.compile(loss="categorical_crossentropy",
                     optimizer=sgd,
                     metrics=["accuracy"],
                     )

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('block2_model_pyramid.h5', verbose=1, monitor='val_loss',
                                                         save_best_only=True, mode='auto')

history = block5_model.fit(
    train_generator,
    epochs=500,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
        learning_rate_scheduler
    ]
)

# %% Evaluation on
# './data/val/'

models = []

print("vgg16_model.h5 was loaded..")
models.append(tf.keras.models.load_model("vgg16_model.h5"))

for i in range(2, 6):
    file_name = f"block{i}_model_pyramid.h5"
    model = tf.keras.models.load_model(file_name)
    print(file_name, "was loaded..")
    models.append(model)

stackX = None
for model in models:
    probabilities = model.predict_generator(test_generator)
    predicted_class_indices = np.argmax(probabilities, axis=1)
    labels = (test_generator.class_indices)
    labels2 = dict((v, k) for k, v in labels.items())
    # predictions=np.array([labels2[k] for k in predicted_class_indices]).astype(np.int8)
    acc = accuracy_score(test_generator.classes, predicted_class_indices)
    print(f"Model Accuracy: {acc:.3f}")

    predicted_class_indices = np.reshape(predicted_class_indices, (1, -1))
    if stackX is None:
        stackX = predicted_class_indices
    else:
        stackX = np.vstack((stackX, predicted_class_indices))

# Most common
result = []
for i in range(stackX.shape[1]):
    counts = np.bincount(stackX[:, i])
    result.append(np.argmax(counts))

acc = accuracy_score(test_generator.classes, result)
print(f"Most common test accuracy: {acc:.3f}")

# flatten predictions to [rows, members x probabilities]
# stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
model = LogisticRegression(max_iter=100)
model.fit(stackX.T, test_generator.classes)
yhat = model.predict(stackX.T)
acc = accuracy_score(test_generator.classes, yhat)
print(f"Stacked Test Accuracy: {acc:.3f}")
