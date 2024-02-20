import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definir la arquitectura de la red neuronal
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid') # La clasificación es binaria
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Preprocesamiento de imágenes y carga de datos
train_datagen = ImageDataGenerator(rescale=1./255) # Se ajustan los valores de cada píxel para que estén en el rango [0, 1]
train_generator = train_datagen.flow_from_directory(
    'dataset_directory',  # Ruta al directorio con imágenes de entrenamiento
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')  # La clasificación es binaria

# Entrenamiento del modelo
model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10
)

# Guardar el modelo entrenado
model.save('model.h5')

# Cargar el modelo entrenado
loaded_model = tf.keras.models.load_model('model.h5')