# Bangla Handwritten Word Recognition System

**SICIP BRAC University — Assignment 04**

**Student Name:** [Md. Atik Hasan Rahat]  
**Student ID:** [1000128329]

## 1. Project Overview

This project implements a Bangla handwriting recognition prototype
using a Convolutional Neural Network (CNN), TensorFlow, Streamlit,
and MLflow.

The system allows users to:
- Draw Bangla handwriting on a Streamlit canvas.
- Upload an image containing handwritten Bangla text.
- Segment character-like regions from the input.
- Predict character probabilities using a trained CNN.
- Display a complete Bangla word using a limited dictionary-based decoder.

The application is a prototype rather than a general-purpose
Bangla word recognition system.

## 2. Dataset

**Dataset:** BanglaLekha-Isolated

**Source:** https://data.mendeley.com/datasets/hf6sf8zrkc/2

The dataset contains approximately 166,105 images organized
into 84 isolated Bangla character classes.

The dataset was divided into training, validation, and test
subsets using a group-based splitting procedure intended to
reduce leakage between related handwriting samples.

The character mapping is stored in `labels.json`.

## 3. Model Architecture

The project uses a CNN classifier with:

- Convolutional layers
- Batch normalization
- Max-pooling
- Data augmentation
- Dense classification layers
- Dropout regularization
- Softmax output over 84 classes

Input images are converted to grayscale, resized while preserving
their aspect ratio, and represented as 64 × 64 pixel images.

The training implementation is provided in `train.py`.

## 4. Training and Results

Model training was tracked using MLflow.

| Training Stage | Best Validation Accuracy |
|---|---:|
| Run 01 — Initial Training | 76.58% |
| Run 02 — Fine-Tuning | 90.98% |

**Final model evaluation:**

- Test accuracy: **90.65%**
- Test loss: **0.3549**

Run 02 continued training the model from Run 01 with a modified
learning rate. These are two tracked training stages, not two
independently initialized models.

The reported accuracy measures **isolated character classification**,
not unrestricted handwritten word recognition.

## 5. MLflow Tracking

The project includes two recorded MLflow runs:

1. `Run_01_Initial_Training`
2. `Run_02_Fine_Tuning`

MLflow was used to record training stages and their associated
metrics.

Evidence of the experiment is available in:

`screenshots/mlflow_experiment.png`

The MLflow database and generated tracking artifacts are excluded
from the source-only submission ZIP.

## 6. Streamlit Application

The Streamlit interface supports:

- Handwriting input using an interactive drawing canvas
- Handwritten image uploads
- Image preprocessing
- Character-region segmentation
- CNN-based classification
- Limited dictionary-based word decoding

### Run Locally

Install Python 3.13 and the project dependencies:

```bash
pip install -r requirements.txt
```

Place the separately supplied trained model at:

```text
models/model.keras
```

Start the application:

```bash
streamlit run app.py
```

Open the address shown by Streamlit, normally:

```text
http://localhost:8501
```

**Important:** The trained model is not distributed in the
source-only submission ZIP. It must be supplied separately.

## 7. Docker Support

Build the Docker image:

```bash
docker build -t bangla-ocr .
```

Run the application from the project directory on Linux or macOS:

```bash
docker run --rm -p 8501:8501 \
  -v "$(pwd)/models:/app/models:ro" \
  bangla-ocr
```

The model file must exist at `models/model.keras` on the host.

Access the application at:

```text
http://localhost:8501
```

The Docker configuration is provided for reproducibility;
successful Docker image building and execution have not yet
been independently verified.

## 8. Project Structure

```text
bangla-ocr-assignment/
├── app.py
├── ocr_utils.py
├── train.py
├── labels.json
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
├── screenshots/
│   ├── mlflow_experiment.png
│   └── streamlit_app.png
└── models/
    └── model.keras    # Supplied separately
```

## 9. Recognition Limitations

BanglaLekha-Isolated is a character-level dataset, not a dataset
of handwritten complete words.

Consequently:

- The model recognizes isolated character classes.
- Character segmentation can fail for connected handwriting.
- The decoder uses a small predefined Bangla word dictionary.
- Dependent vowel signs and complex joined writing are not
  comprehensively supported.
- Arbitrary handwritten words cannot be recognized reliably.
- No general-purpose word-level accuracy has been established.
- The manually assembled character-label mapping requires
  independent verification against an authoritative class mapping.

The Streamlit application demonstrates an end-to-end recognition
pipeline for limited supported words. Future improvements would
require suitable word-level training data and a sequence-based
recognition model.

## 10. Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- OpenCV
- Scikit-learn
- Streamlit
- MLflow
- Docker

## 11. Submission Notes

The source-code submission should include the scripts,
configuration files, README, and required screenshots.

The large dataset, trained model weights, MLflow database,
and temporary training files should not be included in the
source-only ZIP.

The trained model can be retained separately for demonstration.
