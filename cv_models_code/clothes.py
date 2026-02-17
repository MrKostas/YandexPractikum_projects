# Необходимые импорты
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.optimizers import Adam
import numpy as np


# Загружаем обучающую выборку
def load_train(path):
    # применяем загрузчик данных
    train_datagen = ImageDataGenerator(validation_split=0.25, rescale=1.0 / 255)
    # вызываем извлечение данных из папки
    train_datagen_flow = train_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode="sparse",
        subset="training",
        seed=12345,
    )
    # результат: возвращение загрузчика данных тренировочной выборки (тестовые данные сервер грузит сам)
    return train_datagen_flow


# Создаём модель
def create_model(input_shape):
    # инициализация модели
    model = Sequential()
    # добавление слоёв в модель
    model.add(
        Conv2D(6, (5, 5), padding="same", activation="relu", input_shape=input_shape)
    )
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=12, activation="softmax"))
    # подготовка модели к обучению с автоматическим подбором параметров для нейронов
    optimizer = Adam()
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["acc"]
    )
    # результат: возвращение настроенной модели
    return model


# Обучаем модель
def train_model(
    model,
    train_data,
    test_data,
    batch_size=None,
    epochs=10,
    steps_per_epoch=None,
    validation_steps=None,
):
    # назначение тренировочных и тестовых данных
    train_datagen_flow = train_data
    test_datagen_flow = test_data
    # обучение модели
    model.fit(
        train_data,
        validation_data=test_data,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
    )
    # результат: возвращение обученной модели
    return model
