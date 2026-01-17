# Vehicle Image Classification using Deep Learning

A production-ready image classification system that identifies vehicle types
from images using a fine-tuned **MobileNetV2** deep learning model.

The project covers the full ML lifecycle:
- Data loading & preprocessing
- Transfer learning & fine-tuning
- Model evaluation
- Inference via CLI and Web App (Streamlit)

---

## Vehicle Classes
- bicycle
- boat
- bus
- car
- helicopter
- minibus
- motorcycle
- taxi
- train
- truck

---

## Model Architecture
- Base model: **MobileNetV2 (ImageNet pretrained)**
- Input size: **224 × 224**
- Transfer Learning:
  - Phase 1: Feature extraction (base frozen)
  - Phase 2: Fine-tuning top 30% of layers
- Regularization:
  - Data augmentation
  - Dropout
  - L2 weight decay

---

## Performance
- Validation Accuracy: **~89.6%**
- Dataset size: **36,006 images**
- Evaluation includes:
  - Classification report
  - Confusion matrix
  - Accuracy & loss curves

---

## Web App Demo (Streamlit)

Upload an image and instantly get:
- Predicted vehicle type
- Confidence score

### Run the app locally:
```bash
streamlit run src/inference/app.py
