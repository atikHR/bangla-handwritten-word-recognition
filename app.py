
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from ocr_utils import decode_word

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Bangla Handwritten Word Recognition",
    page_icon="✍️",
    layout="centered"
)


# =====================================
# LOAD MODEL AND LABELS
# =====================================

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(
        ROOT / "models" / "model.keras"
    )

    with open(
        ROOT / "labels.json",
        encoding="utf-8"
    ) as f:
        labels = json.load(f)

    return model, labels


# =====================================
# PREPROCESS CHARACTER
# =====================================

def prepare_character(region):

    img = Image.fromarray(region).convert("L")

    arr = np.asarray(img)
    ys, xs = np.where(arr < 235)

    if len(xs) == 0:
        return np.ones((64, 64), dtype=np.float32)

    img = img.crop((
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1
    ))

    img.thumbnail(
        (56, 56),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new("L", (64, 64), 255)

    canvas.paste(
        img,
        (
            (64 - img.width) // 2,
            (64 - img.height) // 2
        )
    )

    return np.asarray(
        canvas, dtype=np.float32
    ) / 255.0


# =====================================
# WORD SEGMENTATION
# =====================================

def segment_characters(image):

    gray = np.asarray(image.convert("L"))

    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    ys, xs = np.where(binary > 0)

    if len(xs) == 0:
        return [], gray

    pad = 10

    x1 = max(0, int(xs.min()) - pad)
    x2 = min(gray.shape[1], int(xs.max()) + pad + 1)

    y1 = max(0, int(ys.min()) - pad)
    y2 = min(gray.shape[0], int(ys.max()) + pad + 1)

    cropped = gray[y1:y2, x1:x2]

    _, binary = cv2.threshold(
        cropped, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Detect Bangla headline (matra)
    width = cropped.shape[1]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(20, width // 3), 1)
    )

    headline = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    body = cv2.subtract(binary, headline)

    # Vertical projection
    active = (
        np.sum(body > 0, axis=0) >= 2
    ).astype(np.uint8)

    active = cv2.morphologyEx(
        active.reshape(1, -1),
        cv2.MORPH_CLOSE,
        np.ones((1, 7), np.uint8)
    ).flatten()

    # Signed integers avoid boundary detection error
    changes = np.diff(
        np.pad(active.astype(np.int16), (1, 1))
    )

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    min_width = max(6, width // 50)

    regions = [
        cropped[:, int(start):int(end)]
        for start, end in zip(starts, ends)
        if end - start >= min_width
    ]

    return regions, cropped


# =====================================
# STREAMLIT USER INTERFACE
# =====================================

st.title("✍️ Bangla Handwritten Word Recognition")

st.write(
    "Draw a Bangla word using your mouse "
    "or upload an image."
)

st.info(
    "Prototype: best suited to short words "
    "supported by the built-in dictionary."
)

stroke_width = st.slider(
    "Pen thickness",
    min_value=3,
    max_value=15,
    value=6
)

st.subheader("Draw Your Word")

canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=240,
    width=600,
    drawing_mode="freedraw",
    return_image_data=True,
    key="bangla_canvas"
)

st.subheader("Or Upload an Image")

uploaded_file = st.file_uploader(
    "Choose a handwritten word image",
    type=["png", "jpg", "jpeg"]
)

if st.button("🔍 Recognize Word", type="primary"):

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

    elif canvas_result.image_data is not None:

        rgba = Image.fromarray(
            np.asarray(
                canvas_result.image_data,
                dtype=np.uint8
            )
        ).convert("RGBA")

        white = Image.new(
            "RGBA", rgba.size,
            (255, 255, 255, 255)
        )

        white.alpha_composite(rgba)
        image = white.convert("RGB")

    else:
        image = None

    if image is None:
        st.warning("Please draw or upload a word.")

    else:
        regions, cropped = segment_characters(image)

        if len(regions) == 0:
            st.warning(
                "No handwriting detected. Please try again."
            )

        elif len(regions) > 8:
            st.warning(
                "Too many character regions detected. "
                "Try drawing more clearly."
            )

        else:
            model, labels = load_assets()

            processed = np.stack([
                prepare_character(region)
                for region in regions
            ])

            X = processed[..., np.newaxis]

            probabilities = model.predict(
                X, verbose=0
            )

            raw_word, final_word, top_words = decode_word(
                probabilities, labels
            )

            st.divider()

            st.subheader("Recognition Result")

            st.image(
                cropped,
                caption="Detected Handwriting"
            )

            st.write(
                "Detected regions:",
                len(regions)
            )

            st.write(
                "Raw CNN prediction:",
                raw_word
            )

            st.success(
                f"Predicted Complete Word: {final_word}"
            )

            if top_words:
                with st.expander("Other Possible Words"):
                    for word, score in top_words:
                        st.write(
                            f"{word} — Score: {score:.3f}"
                        )

st.divider()

st.caption(
    "SICIP Assignment 04 | BRAC University"
)
