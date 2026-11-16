"""
predict_images.py  -  ARAP Project Part 2

Uses the trained CNN to classify pictures from a folder and saves a picture
with the top-3 guesses for each image.

Output 3 (your own images):
    python predict_images.py my_images
Output 4 (image taken by the e-puck camera in Webots):
    python predict_images.py webots_images

Put .jpg, .png, .webp or .avif files in the folder first. The script shrinks every image
to 32 x 32 pixels, the size the CNN was trained on.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow import keras

try:                                           # lets Pillow open .avif files if the plugin is installed
    import pillow_avif  # noqa: F401
except ImportError:
    pass

MODEL_PATH = "outputs/cifar10_cnn.keras"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def load_image(path):
    """Open an image, make it square, and shrink it to 32 x 32 RGB."""
    img = Image.open(path).convert("RGB")
    side = min(img.size)                       # centre crop so shapes are not stretched
    left, top = (img.width - side) // 2, (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img, np.asarray(img.resize((32, 32), Image.LANCZOS), dtype="float32")


def main(folder):
    files = sorted(f for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.avif") for f in glob.glob(os.path.join(folder, ext)))
    if not files:
        sys.exit(f"No .jpg/.jpeg/.png/.webp/.avif images found in '{folder}'.")

    model = keras.models.load_model(MODEL_PATH)
    fig, axes = plt.subplots(1, len(files), figsize=(4 * len(files), 4.5), squeeze=False)
    for ax, path in zip(axes[0], files):
        original, small = load_image(path)
        probs = model.predict(small[None, ...], verbose=0)[0]
        top3 = np.argsort(-probs)[:3]
        print(f"\n{os.path.basename(path)}")
        for rank, idx in enumerate(top3, 1):
            print(f"  {rank}. {CLASS_NAMES[idx]:<11} {probs[idx] * 100:5.1f} %")
        ax.imshow(original); ax.axis("off")
        ax.set_title("\n".join(f"{CLASS_NAMES[i]}: {probs[i] * 100:.1f}%" for i in top3), fontsize=10)
    fig.tight_layout()
    out = f"outputs/predictions_{os.path.basename(os.path.normpath(folder))}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.2)
    print("\nSaved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "my_images")
