# 🚗 Vehicle Image Classifier

A Convolutional Neural Network (CNN) that classifies images into 20 vehicle categories, wrapped in an interactive Streamlit web app.

**🔗 Live aApp:** [vehicle-classification-app.streamlit.app](https://vehicle-classification-app-v3vcvgkm6sizzh7x6gjoln.streamlit.app/)

---

## 📋 Overview

This project trains a CNN from scratch on the [Vehicles Image Dataset](https://www.kaggle.com/datasets/mmohaiminulislam/vehicles-image-dataset) from Kaggle, then serves the trained model through a Streamlit interface where users can upload an image or paste an image URL to get a live prediction.

**How it works:**
1. The CNN scans the image in small patches (filters) to detect simple patterns first — edges, colors, textures.
2. Each layer stacks on the previous one, combining simple patterns into more complex shapes (wheels, windows, wings) the deeper it goes.
3. At the end, it takes all the learned features and votes on which of the 20 vehicle classes the image most likely belongs to.

## 🏷️ Classes

The model recognizes 20 categories:

`airplane` · `ambulance` · `bicycle` · `boat` · `bus` · `car` · `fire_truck` · `helicopter` · `hovercraft` · `jet_ski` · `kayak` · `motorcycle` · `rickshaw` · `scooter` · `segway` · `skateboard` · `tractor` · `truck` · `unicycle` · `van`

## 🖥️ App Preview

- Upload an image file (JPG/PNG) **or** paste an image URL
- View the top predicted class with a confidence score
- See a horizontal bar chart of the top 5 predictions
- Low-confidence predictions (< 60%) are flagged as "not a recognized vehicle" instead of a false-confident guess

## 🧠 Model Architecture

Built with `tf.keras.Sequential`, trained on 128×128 RGB images:

| Block | Layers |
|-------|--------|
| Input | `Input(128, 128, 3)` |
| Augmentation | `RandomFlip`, `RandomRotation(0.1)`, `RandomZoom(0.1)`, `Rescaling(1./255)` |
| Block 1 | `Conv2D(32)` → `BatchNorm` → `MaxPooling2D` → `Dropout(0.2)` |
| Block 2 | `Conv2D(64)` → `BatchNorm` → `MaxPooling2D` → `Dropout(0.25)` |
| Block 3 | `Conv2D(128)` → `BatchNorm` → `MaxPooling2D` → `Dropout(0.3)` |
| Block 4 | `Conv2D(256)` → `BatchNorm` → `MaxPooling2D` → `Dropout(0.35)` |
| Head | `GlobalAveragePooling2D` → `Dense(128, relu)` → `Dropout(0.5)` → `Dense(20, softmax)` |

**Training configuration:**
- Optimizer: Adam (`lr=1e-3`)
- Loss: Sparse Categorical Crossentropy
- Epochs: 80
- Batch size: 32
- Data split: 80% train / 10% validation / 10% test (via `split-folders`)

**Results (final epoch on test set):**
- Test Accuracy: **~47.1%**
- Test Loss: **~2.22**

> ⚠️ Training accuracy reached ~74% while validation/test accuracy plateaued around 45–48%, indicating some overfitting despite dropout and augmentation. There's room to improve via transfer learning (e.g. MobileNetV2/EfficientNet), more aggressive augmentation, class balancing, or early stopping — see [Future Improvements](#-future-improvements).

## 📁 Project Structure

```
.
├── test_images/                       # Sample test images
├── CNN_Vehicle_Classification.ipynb   # Data prep, model training & evaluation
├── app.py                             # Streamlit web app
├── cars_cnn.keras                     # Trained model weights
├── requirements.txt                   # Python dependencies
└── README.md
```

## ⚙️ Setup & Installation

1. **Clone the repository**
   ```bash
   git clone <https://github.com/dawoodofficial04/vehicle-classification-app.git>
   cd vehicle-classification-app
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

   The app will open in your browser at `http://localhost:8501`.

## 🏋️ Retraining the Model

The full training pipeline is in `CNN_Vehicle_Classification.ipynb`:

1. Downloads the dataset from Kaggle via `kagglehub`
2. Splits it 80/10/10 into train/val/test using `split-folders`
3. Cleans up non-RGB images (converts 4-channel/grayscale images to RGB)
4. Loads images with `tf.keras.utils.image_dataset_from_directory` at 128×128
5. Builds and trains the CNN described above
6. Evaluates on the test set and saves the model as `cars_cnn.keras`

To retrain, open the notebook, run all cells, and replace `cars_cnn.keras` with the newly saved model.

## 🛠️ Tech Stack

- **Model:** TensorFlow / Keras
- **App:** Streamlit
- **Image handling:** Pillow, `tf.keras.preprocessing.image`
- **Data prep:** kagglehub, split-folders
- **Deployment:** Streamlit Community Cloud

## 🚧 Future Improvements

- Use transfer learning (MobileNetV2, ResNet, EfficientNet) to boost accuracy beyond the current ~47%
- Add early stopping / learning rate scheduling to reduce overfitting
- Expand and rebalance the dataset across classes
- Add a confusion matrix / per-class metrics to the notebook
- Cache predictions and add batch (multi-image) upload support to the app

## 📄 License

Add your license of choice here (e.g. MIT).