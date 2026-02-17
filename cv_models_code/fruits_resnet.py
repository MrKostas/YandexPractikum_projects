# Необходимые импорты
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet50

# Загружаем обучающую выборку
def load_train(path):
    # применяем загрузчик данных
    train_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255)
    # вызываем извлечение данных из папки
    train_datagen_flow = train_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='training',
        seed=12345)
    # результат: возвращение загрузчика данных тренировочной выборки (тестовые данные сервер грузит сам)
    return train_datagen_flow

# Создаём модель
def create_model(input_shape):
    # импортируем архитектуру ResNet
    # инициализируем `костяк`
    backbone = ResNet50(input_shape=(150, 150, 3),
                        # чтобы код выполнялся быстрее, веса модели `ResNet50` уже загружены на сервер
                        # прочитаем их, указав в аргументе `weights` путь
                        weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                        # указываем, что верхушку убираем, делать её будем заново
                        include_top=False)
    # инициализация модели
    model = Sequential()
    # добавление слоёв в модель
    # добавляем `костяк`
    model.add(backbone)
    # добавляем глобальный пуллинг
    model.add(GlobalAveragePooling2D())
    # добавляем полносвязный слой с 12 нейронами
    model.add(Dense(12, activation='softmax')) 
    # подготовка модели к обучениюс автоматическим подбором параметров для нейронов
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])
    # результат: возвращение настроенной модели
    return model

# Обучаем модель
def train_model(model, train_data, test_data, batch_size=None, epochs=10,
               steps_per_epoch=None, validation_steps=None):
    # назначение тренировочных и тестовых данных
    train_datagen_flow = train_data
    test_datagen_flow = test_data
    # обучение модели
    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)
    # результат: возвращение обученной модели
    return model