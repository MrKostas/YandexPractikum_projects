# построение нейронной сети для прогнозирования возраста покупателей по фото

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    AveragePooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet50
import numpy as np
import pandas as pd


# функция загрузки обучающей выборки
def load_train(path):
    labels = pd.read_csv("/datasets/faces/labels.csv")
    train_datagen = ImageDataGenerator(
        vertical_flip=True,
        rotation_range=90,
        width_shift_range=0.3,
        height_shift_range=0.3,
        rescale=1.0 / 255,
        validation_split=0.25,
    )

    train_datagen_flow = train_datagen.flow_from_dataframe(
        dataframe=labels,
        directory="/datasets/faces/final_files/",
        x_col="file_name",
        y_col="real_age",
        target_size=(224, 224),
        batch_size=32,
        class_mode="raw",
        subset="training",
        seed=12345,
    )

    return train_datagen_flow


# функция загрузки валидационной выборки
def load_test(path):
    labels = pd.read_csv("/datasets/faces/labels.csv")
    test_datagen = ImageDataGenerator(validation_split=0.25, rescale=1.0 / 255)

    test_datagen_flow = test_datagen.flow_from_dataframe(
        dataframe=labels,
        directory="/datasets/faces/final_files/",
        x_col="file_name",
        y_col="real_age",
        target_size=(224, 224),
        batch_size=32,
        class_mode="raw",
        subset="validation",
        seed=12345,
    )

    return test_datagen_flow


# функция создания модели нейронной сети
def create_model(input_shape):
    # импортируем архитектуру ResNet
    backbone = ResNet50(
        input_shape=input_shape,
        weights="/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
        include_top=False,
    )
    # инициализация модели
    model = Sequential()
    # добавление слоёв в модель
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation="relu"))
    # подготовка модели к обучению
    optimizer = Adam(0.00005)
    model.compile(
        optimizer=optimizer, loss="mean_squared_error", metrics=["mean_absolute_error"]
    )
    # результат: возвращение настроенной модели
    return model


# функция обучения модели
def train_model(
    model,
    train_datagen_flow,
    test_datagen_flow,
    batch_size=None,
    epochs=20,
    steps_per_epoch=None,
    validation_steps=None,
):

    model.fit(
        train_datagen_flow,
        validation_data=test_datagen_flow,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
        shuffle=True,
    )

    return model
