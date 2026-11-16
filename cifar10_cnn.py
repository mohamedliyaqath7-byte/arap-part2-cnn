"""
cifar10_cnn.py  -  ARAP Project Part 2

Builds, trains and evaluates a CNN on CIFAR-10, then makes it lighter
with TensorFlow Lite quantization and measures the effect.

Run in Google Colab (GPU) or on your laptop:
    python cifar10_cnn.py

Everything it produces is saved in the folder "outputs":
    model_summary.txt      all layers of the model          (output 1)
    metrics.json           accuracy, speed, size numbers    (output 2 + optimisation)
    training_curves.png    accuracy and loss graphs
    confusion_matrix.png   where the model gets confused
    misclassified.png      error analysis pictures
    cifar10_cnn.keras      the trained model
    cifar10_cnn.tflite     the quantized (lighter) model
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
EPOCHS = 40
BATCH_SIZE = 64
OUT_DIR = "outputs"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

os.makedirs(OUT_DIR, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)


# ----------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------
def load_data():
    """Load CIFAR-10 and keep 5000 training images for validation."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()
    x_val, y_val = x_train[-5000:], y_train[-5000:]
    x_train, y_train = x_train[:-5000], y_train[:-5000]
    print("train", x_train.shape, "val", x_val.shape, "test", x_test.shape)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def augment(image, label):
    """Random flip and shift so the model sees more variety (less overfitting)."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, [32, 32, 3])
    return image, label


def make_dataset(x, y, training):
    ds = tf.data.Dataset.from_tensor_slices((x.astype("float32"), y))
    if training:
        ds = ds.shuffle(10000, seed=SEED).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------------
# 2. Model (the layers)
# ----------------------------------------------------------------------
def conv_block(x, filters, dropout):
    """Two Conv layers + BatchNorm + ReLU, then MaxPool and Dropout."""
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    return layers.Dropout(dropout)(x)


def build_model():
    inputs = keras.Input(shape=(32, 32, 3), name="image_32x32_rgb")
    x = layers.Rescaling(1.0 / 255, name="scale_0_to_1")(inputs)  # raw pixels 0-255 in
    x = conv_block(x, 32, 0.2)
    x = conv_block(x, 64, 0.3)
    x = conv_block(x, 128, 0.4)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax", name="class_probabilities")(x)
    return keras.Model(inputs, outputs, name="cifar10_cnn")


# ----------------------------------------------------------------------
# 3. Evaluation helpers
# ----------------------------------------------------------------------
def save_curves(history):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history.history["accuracy"], label="train")
    ax[0].plot(history.history["val_accuracy"], label="validation")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history.history["loss"], label="train")
    ax[1].plot(history.history["val_loss"], label="validation")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/training_curves.png", dpi=150)
    plt.close(fig)


def save_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(10)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (test set)")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close(fig)
    return cm


def save_misclassified(x, y_true, y_pred, probs, n=16):
    wrong = np.where(y_true != y_pred)[0]
    # show the most confident mistakes first
    wrong = wrong[np.argsort(-probs[wrong, y_pred[wrong]])][:n]
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, idx in zip(axes.ravel(), wrong):
        ax.imshow(x[idx]); ax.axis("off")
        ax.set_title(f"true: {CLASS_NAMES[y_true[idx]]}\npred: {CLASS_NAMES[y_pred[idx]]}", fontsize=8)
    fig.suptitle("Most confident mistakes (error analysis)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/misclassified.png", dpi=150)
    plt.close(fig)


def keras_speed(model, x, n=300):
    """Average time (seconds) to classify ONE image with the Keras model."""
    model(x[:1].astype("float32"), training=False)  # warm-up
    start = time.perf_counter()
    for i in range(n):
        model(x[i:i + 1].astype("float32"), training=False)
    return (time.perf_counter() - start) / n


def tflite_eval(path, x, y, n=2000):
    """Accuracy and average time per image of the TFLite model."""
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    correct, start = 0, time.perf_counter()
    for i in range(n):
        interp.set_tensor(inp["index"], x[i:i + 1].astype("float32"))
        interp.invoke()
        correct += int(np.argmax(interp.get_tensor(out["index"])[0]) == y[i])
    return correct / n, (time.perf_counter() - start) / n


# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------
def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    train_ds = make_dataset(x_train, y_train, training=True)
    val_ds = make_dataset(x_val, y_val, training=False)

    model = build_model()
    model.summary()                                           # output 1: all layers
    with open(f"{OUT_DIR}/model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ]
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    train_minutes = (time.time() - start) / 60

    # ---- output 2: accuracy on images the model has never seen ----
    probs = model.predict(x_test.astype("float32"), batch_size=256, verbose=0)
    y_pred = probs.argmax(axis=1)
    test_acc = float((y_pred == y_test).mean())
    print(f"\nTEST ACCURACY: {test_acc * 100:.2f} %")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))

    save_curves(history)
    cm = save_confusion(y_test, y_pred)
    save_misclassified(x_test, y_test, y_pred, probs)
    per_class = {CLASS_NAMES[i]: float(cm[i, i] / cm[i].sum()) for i in range(10)}

    model.save(f"{OUT_DIR}/cifar10_cnn.keras")
    keras_size_kb = os.path.getsize(f"{OUT_DIR}/cifar10_cnn.keras") / 1024
    keras_time = keras_speed(model, x_test)

    # ---- optimisation: post-training quantization (dynamic range) ----
    metrics = {
        "test_accuracy": test_acc,
        "per_class_accuracy": per_class,
        "parameters": int(model.count_params()),
        "epochs_run": len(history.history["loss"]),
        "training_minutes": round(train_minutes, 1),
        "keras_size_kb": round(keras_size_kb, 1),
        "keras_ms_per_image": round(keras_time * 1000, 2),
        "keras_fps": round(1 / keras_time, 1),
    }
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(f"{OUT_DIR}/cifar10_cnn.tflite", "wb") as f:
            f.write(tflite_model)
        acc_q, t_q = tflite_eval(f"{OUT_DIR}/cifar10_cnn.tflite", x_test, y_test)
        metrics.update({
            "tflite_size_kb": round(len(tflite_model) / 1024, 1),
            "tflite_accuracy_first_2000_test_images": acc_q,
            "tflite_ms_per_image": round(t_q * 1000, 2),
            "tflite_fps": round(1 / t_q, 1),
            "size_reduction_x": round(keras_size_kb / (len(tflite_model) / 1024), 2),
        })
    except Exception as err:  # keep the main results even if conversion fails
        print("TFLite conversion failed:", err)
        metrics["tflite_error"] = str(err)

    with open(f"{OUT_DIR}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
