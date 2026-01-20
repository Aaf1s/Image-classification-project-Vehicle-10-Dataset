# Vehicle Image Classification using Deep Learning

A production-ready image classification system that identifies vehicle types
from images using a fine-tuned MobileNetV2 deep learning model.

The project covers the full ML lifecycle:
- Data loading and preprocessing
- Transfer learning and fine-tuning
- Model evaluation
- Inference via CLI and Web App (Streamlit)

 <img width="633" height="772" alt="image" src="https://github.com/user-attachments/assets/5f54848b-8cba-4cb6-86a1-79e432e6a647" />
 

--------------------------------------------------

VEHICLE CLASSES
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

--------------------------------------------------

MODEL ARCHITECTURE
- Base model: MobileNetV2 (ImageNet pretrained)
- Input size: 224 x 224
- Transfer Learning:
  - Phase 1: Feature extraction (base frozen)
  - Phase 2: Fine-tuning top 30 percent of layers
- Regularization:
  - Data augmentation
  - Dropout
  - L2 weight decay

--------------------------------------------------

PERFORMANCE
- Validation Accuracy: approximately 89.6 percent
- Dataset size: 36,006 images
- Evaluation includes:
  - Classification report
  - Confusion matrix
  - Accuracy and loss curves

--------------------------------------------------

HOW YOU CAN RUN THIS PROJECT (STEP-BY-STEP)

1. Clone the repository
git clone https://github.com/Aaf1s/Image-classification-project-Vehicle-10-Dataset.git
cd Image-classification-project-Vehicle-10-Dataset

2. Create and activate a virtual environment
python -m venv venv

Windows:
venv\Scripts\activate

macOS or Linux:
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Run the Streamlit web application
streamlit run src/inference/app.py

5. Open the browser
The app will open automatically, or visit:
http://localhost:8501

Upload any vehicle image to get:
- Predicted vehicle type
- Confidence score

--------------------------------------------------

OPTIONAL: MODEL TRAINING
Training is NOT required for inference.
A trained model is already included.

To retrain the model:
python src/training/train.py

Saved artifacts:
- models/vehicle10_mobilenet_best.keras
- models/vehicle10_mobilenet_last.keras
- models/training_history.json

--------------------------------------------------

OPTIONAL: Model Evaluation

python src/evaluation/evaluate_model.py

--------------------------------------------------

OPTIONAL: CLI Image Prediction

python src/inference/predict_image.py

--------------------------------------------------

PROJECT STRUCTURE

```text
src/
  training/
  evaluation/
  inference/
  utils/

models/

README.md
requirements.txt
```


--------------------------------------------------

NOTES
- Fully runnable on CPU
- No notebooks required
- Clean, modular ML pipeline
- Both CLI and Web-based inference supported
