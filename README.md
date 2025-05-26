# Plant Identification Project

This project implements a simple plant identification system using a Convolutional Neural Network (CNN) and Flask web application. It can identify three types of plants: daisy, dandelion, and rose.

## Project Structure
```
plant_identification/
├── app.py              # Flask web application
├── model.py            # CNN model architecture
├── train.py            # Training script
├── utils.py            # Utility functions
├── requirements.txt    # Project dependencies
├── templates/          # HTML templates
│   └── index.html     # Web interface
├── dataset/           # Training data
│   ├── train/        # Training images
│   └── validation/   # Validation images
└── uploads/          # Temporary upload directory
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Create a `dataset` folder with `train` and `validation` subfolders
   - Place your plant images in respective class folders:
     ```
     dataset/
     ├── train/
     │   ├── daisy/
     │   ├── dandelion/
     │   └── rose/
     └── validation/
         ├── daisy/
         ├── dandelion/
         └── rose/
     ```

4. Train the model:
```bash
python train.py
```

5. Run the web application:
```bash
python app.py
```

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Environment Variables: None required

## Usage

1. Open the web application in your browser
2. Upload an image of a plant
3. The system will predict the plant type and show the confidence level

## Notes

- The model is trained on a small dataset for demonstration purposes
- For better accuracy, use a larger dataset with more plant types
- The web interface supports JPG, JPEG, and PNG image formats
- Maximum file size is limited to 16MB 