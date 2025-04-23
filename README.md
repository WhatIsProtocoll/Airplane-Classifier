# Airplane Classifier

This project uses deep learning to detect and classify airplanes in real-world images. It leverages YOLOv8 for object detection and a custom-trained DenseNet201 model for classification into 16 commercial airplane families.

---

## Project Structure

```
├── data/                # Folder with training and test images
├── app.py               # Streamlit app for running inference with YOLO + classifier
├── model.ipynb          # Notebook used for training the model
├── requirements.txt     # Python dependencies
├── packages.txt         # System dependencies (for OpenCV image handling and Streamlit compatibility)

```

---

## Model Information

- **Architecture**: DenseNet201 (ImageNet weights) + custom classifier head
- **Input shape**: 224x224x3
- **Classes**: 16 airplane families
- **Augmentations**: random flip, crop, brightness, Gaussian noise, downsampling
- **Performance**:  
  - ~83% validation accuracy  
  - 79% final test accuracy 
- **Loss**: `categorical_crossentropy`, trained with SGD and early stopping

---

## Class Labels

```
['A320', 'A330', 'A340', 'A350', 'A380', 'ATR-72', 'Boeing_737', 'Boeing_737_MAX',
 'Boeing_747', 'Boeing_757', 'Boeing_767', 'Boeing_777', 'Boeing_787', 
 'CRJ-700', 'Dash_8', 'Embraer_E-Jet']
```

---

## Training Your Own Model

To train the model:

1. Run `model.ipynb` in Google Colab or locally (GPU recommended).
2. Save and export the `.keras` model for use in the app.

---

## How to Run the App

Install the required packages:

```bash
pip install -r requirements.txt
```

Then launch the Streamlit app:

```bash
streamlit run app.py
```

The app allows users to upload an airport image. It uses YOLOv8 to detect airplanes, crops them, and classifies them using the trained DenseNet model.

---

## Live Image Inference

- YOLOv8 detects airplanes in the image.
- The cropped airplane images are passed to the classifier.
- The predicted airplane family name and confidence score are shown directly in the app.


## Authors

- **William Bour Hutchinson**
- **Sander Conn Ødegaard**  

---

