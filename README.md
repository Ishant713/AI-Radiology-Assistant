# AI Assistant for Radiology Images (Educational Demo)

This app performs preliminary analysis on chest X-ray images using a pretrained DenseNet model from `torchxrayvision`. It displays predicted pathologies and a Grad-CAM heatmap overlay to highlight regions that most influence the prediction.

**Important**: This tool is for research and education only. It is not a medical device and must not be used for diagnosis or treatment.

## Features
- Upload JPG/PNG or DICOM (.dcm) chest X-rays
- Auto-windowing for DICOM
- Predictions for common chest pathologies (CheXpert-style labels)
- Grad-CAM heatmap overlay
- Simple Streamlit UI

## Quickstart
1. Create a virtual environment (Python 3.9+ recommended)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Notes
- Model supports **posterior-anterior/ap radiographs**. It is not trained for CT, MRI, or ultrasound.
- First run downloads model weights automatically.
- No PHI should be used.