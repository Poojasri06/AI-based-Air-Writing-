import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ==============================
# GLOBAL CONFIG
# ==============================
IMG_SIZE = 28
EPOCHS = 15
BATCH_SIZE = 32

BASE_DATASET_PATH = "data"
MODEL_SAVE_DIR = "models"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ==============================
# LANGUAGE CONFIG
# ==============================
LANGUAGES = {
    "english": {
        "num_classes": 26  # A-Z letters
    }
}

# ==============================
# MODEL ARCHITECTURE
# ==============================
def build_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ==============================
# TRAIN LOOP (LANGUAGE-WISE)
# ==============================
for language, config in LANGUAGES.items():
    print(f"\n🚀 Training model for: {language.upper()}")

    dataset_path = os.path.join(BASE_DATASET_PATH, language)
    model_save_path = os.path.join(MODEL_SAVE_DIR, f"{language}_cnn.h5")

    num_classes = config["num_classes"]

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        image_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )

    train_ds = train_ds.map(
        lambda x, y: (x / 255.0, tf.one_hot(y, num_classes))
    )

    model = build_model(num_classes)
    model.fit(train_ds, epochs=EPOCHS)

    model.save(model_save_path)
    print(f"✅ {language.upper()} model saved at: {model_save_path}")

print("\n🎉 All language models trained successfully!")
