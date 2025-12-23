import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# ==============================
#  AYARLAR
# ==============================
DATA_DIR = "asl_dataset_mask"   # Maskeli dataset
IMG_SIZE = 96
BATCH_SIZE = 32
VAL_SPLIT = 0.2
EPOCHS = 15

# ==============================
#  VERİ YÜKLEYİCİ
# ==============================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VAL_SPLIT,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=(0.8, 1.2)
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

print("Sınıflar:", train_gen.class_indices)
num_classes = train_gen.num_classes

# ==============================
#  MobileNetV2 Modeli
# ==============================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
preds = Dense(num_classes, activation="softmax")(x)

model = Model(base_model.input, preds)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
#  CALLBACKS
# ==============================
checkpoint = ModelCheckpoint(
    "asl_cnn_best.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early = EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

# ==============================
#  AŞAMA 1: Üst katmanları eğit
# ==============================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, early]
)

# ==============================
#  AŞAMA 2: Fine-tuning
# ==============================
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, early]
)

model.save("asl_cnn.h5")
print("Model kaydedildi.")

with open("class_indices.json", "w", encoding="utf-8") as f:
    json.dump(train_gen.class_indices, f, indent=2)

print("class_indices.json oluşturuldu.")
