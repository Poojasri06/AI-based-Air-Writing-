# AI-Based Air Writing

An AI-powered application that lets you write in the air using hand gestures, powered by MediaPipe and TensorFlow.

## 🚀 Features

- ✍️ Write in the air using your index finger
- ✌️ Add spaces with two fingers (index + middle)
- ✊ Delete with a fist gesture
- 🎨 Choose your drawing color
- 🤖 AI-powered character recognition (when model is available)

## 📦 Installation

### For Streamlit Cloud Deployment

1. Fork/clone this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy the app by pointing to `app.py` as the main file
4. All dependencies will be installed automatically from `requirements.txt`

### For Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## 📁 Project Structure

```
.
├── app.py                  # Streamlit web application (USE THIS FOR DEPLOYMENT)
├── main.py                 # Original OpenCV desktop application
├── dataset_collector.py    # Tool for collecting training data
├── train.py                # Model training script
├── models/                 # Pre-trained models
│   └── english_cnn.h5     # English alphabet model
├── requirements.txt        # Python dependencies
└── .gitignore             # Git ignore file
```

## 🎯 Usage

### Streamlit Web App (app.py)

1. Run the app: `streamlit run app.py`
2. Allow camera access when prompted
3. Use the camera input to capture frames with your hand gestures
4. Follow the on-screen instructions to write in the air

### Desktop App (main.py)

The original desktop application (`main.py`) uses OpenCV's GUI functions and is designed for local use only. It **cannot** be deployed to Streamlit Cloud.

To run locally:
```bash
python main.py
```

## 🛠️ Technical Details

### Dependencies

- **opencv-python-headless**: Image processing (headless version for cloud deployment)
- **mediapipe**: Hand tracking and gesture recognition
- **tensorflow**: Neural network model for character recognition
- **streamlit**: Web application framework
- **numpy**: Numerical computations
- **Pillow**: Image handling

### Model

The application uses a Convolutional Neural Network (CNN) trained on hand-written characters. The model architecture includes:
- 2 Convolutional layers with MaxPooling
- Dense layers for classification
- Trained on 28x28 grayscale images

## 🔧 Training Your Own Model

1. Collect data using `dataset_collector.py`
2. Organize data in the `data/` directory by language and character
3. Run `python train.py` to train the model
4. The trained model will be saved in the `models/` directory

## 📝 Notes

- The Streamlit app uses `st.camera_input()` which requires user interaction to capture frames
- For best results, use good lighting and a plain background
- The model file (`models/english_cnn.h5`) is included in the repository

## 🐛 Troubleshooting

### ModuleNotFoundError on Streamlit Cloud

- Make sure `app.py` is set as the main file in Streamlit Cloud settings
- Verify that `requirements.txt` is present in the repository root
- Check that `opencv-python-headless` (not `opencv-python`) is in requirements.txt

### Model Not Loading

- Ensure the `models/english_cnn.h5` file exists in the repository
- The app will run in draw-only mode if the model is not found
- Check the Streamlit logs for specific error messages

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- MediaPipe by Google for hand tracking
- TensorFlow for machine learning framework
- Streamlit for the web application framework
