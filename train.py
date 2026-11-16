
import os
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow

from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras import layers, models

# ======================================
# 1. CONFIGURATION
# ======================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get(
    "BANGLALEKHA_IMAGES",
    "/kaggle/working/banglalekha_data/"
    "BanglaLekha-Isolated/Images"
))

MODEL_DIR = ROOT / "models"
MLFLOW_DIR = ROOT / "artifacts" / "mlflow"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 64
BATCH_SIZE = 64

tf.keras.utils.set_random_seed(42)

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}"
)

mlflow.set_experiment(
    "SICIP_Bangla_Handwritten_Recognition"
)

# ======================================
# 2. LOAD DATASET
# ======================================

def load_dataframe():
    records = []

    for class_id in range(1, 85):
        folder = DATA_DIR / str(class_id)

        for path in folder.glob("*.png"):
            records.append({
                "path": str(path),
                "label": class_id - 1,
                "form": path.stem.rsplit("_", 1)[0]
            })

    df = pd.DataFrame(records)

    if df.empty or df["label"].nunique() != 84:
        raise ValueError(
            "Dataset not found or contains fewer than 84 classes."
        )

    return df


def split_data(df):
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )

    train_idx, temp_idx = next(
        splitter.split(df, groups=df["form"])
    )

    train = df.iloc[train_idx]
    temp = df.iloc[temp_idx]

    splitter2 = GroupShuffleSplit(
        n_splits=1, test_size=0.5, random_state=42
    )

    val_idx, test_idx = next(
        splitter2.split(temp, groups=temp["form"])
    )

    train = train.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    val = temp.iloc[val_idx].reset_index(drop=True)
    test = temp.iloc[test_idx].reset_index(drop=True)

    return train, val, test


# ======================================
# 3. IMAGE PREPROCESSING
# ======================================

def preprocess(path):
    img = Image.open(path).convert("L")
    arr = np.asarray(img)

    if np.median(arr) < 127:
        img = ImageOps.invert(img)

    arr = np.asarray(img)
    ys, xs = np.where(arr < 235)

    if len(xs):
        img = img.crop((
            xs.min(), ys.min(),
            xs.max() + 1, ys.max() + 1
        ))

    img.thumbnail(
        (56, 56),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new("L", (64, 64), 255)

    canvas.paste(
        img,
        ((64 - img.width) // 2,
         (64 - img.height) // 2)
    )

    return np.asarray(
        canvas, dtype=np.float32
    ) / 255.0


def make_dataset(df, training=False):

    def load(path, label):
        def read(p):
            return preprocess(
                p.decode("utf-8")
            )

        image = tf.numpy_function(
            read, [path], tf.float32
        )

        image.set_shape((64, 64))
        image = image[..., None]

        return image, tf.cast(label, tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((
        df["path"].values,
        df["label"].values.astype(np.int32)
    ))

    if training:
        ds = ds.shuffle(
            len(df), seed=42,
            reshuffle_each_iteration=True
        )

    return (
        ds.map(
            load,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


# ======================================
# 4. IMPROVED CNN
# ======================================

def build_model():

    model = models.Sequential([
        layers.Input(shape=(64, 64, 1)),

        layers.RandomRotation(
            0.04, fill_mode="constant", fill_value=1.0
        ),
        layers.RandomTranslation(
            0.05, 0.05,
            fill_mode="constant", fill_value=1.0
        ),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(84, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ======================================
# 5. MLFLOW LOGGING
# ======================================

class MLflowLogger(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        for name, value in (logs or {}).items():
            mlflow.log_metric(
                name, float(value), step=epoch + 1
            )


# ======================================
# 6. TRAIN TWO MLFLOW RUNS
# ======================================

def main():

    print("Loading BanglaLekha dataset...")
    df = load_dataframe()

    train_df, val_df, test_df = split_data(df)

    train_ds = make_dataset(train_df, True)
    val_ds = make_dataset(val_df)
    test_ds = make_dataset(test_df)

    print("Training images:", len(train_df))
    print("Validation images:", len(val_df))
    print("Testing images:", len(test_df))

    model = build_model()

    experiments = [
        ("Run_01_Initial_Training", 0, 3, 0.0005),
        ("Run_02_Fine_Tuning", 3, 6, 0.0002)
    ]

    results = []

    for name, start_epoch, end_epoch, lr in experiments:

        model.optimizer.learning_rate.assign(lr)

        checkpoint_path = MODEL_DIR / f"{name}.keras"

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True
        )

        with mlflow.start_run(run_name=name) as run:

            mlflow.log_params({
                "model": "Improved_CNN",
                "training_stage": name,
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
                "learning_rate": lr,
                "batch_size": BATCH_SIZE,
                "image_size": IMG_SIZE,
                "classes": 84
            })

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                initial_epoch=start_epoch,
                epochs=end_epoch,
                callbacks=[MLflowLogger(), checkpoint],
                verbose=1
            )

            best_acc = max(
                history.history["val_accuracy"]
            )

            mlflow.log_metric(
                "best_val_accuracy", float(best_acc)
            )

            mlflow.log_artifact(
                str(checkpoint_path),
                artifact_path="trained_model"
            )

            results.append((
                best_acc, checkpoint_path, run.info.run_id
            ))

            print(
                f"\n{name}: {best_acc * 100:.2f}%"
            )

    # Select best checkpoint from both runs
    best_acc, best_path, best_run_id = max(
        results, key=lambda item: item[0]
    )

    final_path = MODEL_DIR / "model.keras"
    shutil.copy2(best_path, final_path)

    # Evaluate best model
    final_model = tf.keras.models.load_model(final_path)

    test_loss, test_acc = final_model.evaluate(
        test_ds, verbose=1
    )

    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_metric("test_accuracy", float(test_acc))
        mlflow.log_metric("test_loss", float(test_loss))

    print("\n===== FINAL RESULTS =====")
    print(f"Best validation accuracy: {best_acc * 100:.2f}%")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    print("Model saved:", final_path)
    print("MLflow runs completed:", len(results))


if __name__ == "__main__":
    main()
