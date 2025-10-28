# Cat Emote

I have no idea why I made this 


## Notes

- **Real-time emotion detection** using DeepFace
- **7 emotion categories**: angry, disgust, fear, happy, sad, surprise, neutral

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the setup script to configure your emotion images:
```bash
python setup_emotion_images.py
```

3. Or run the main application directly:
```bash
python emotion_detector.py
```

## File Structure

```
catemote/
├── emotion_detector.py          # Main application
├── setup_emotion_images.py     # Setup script for emotion images
├── requirements.txt            # Python dependencies
├── emotion_images/            # Directory for emotion images
│   ├── angry.png
│   ├── disgust.png
│   ├── fear.png
│   ├── happy.png
│   ├── sad.png
│   ├── surprise.png
│   └── neutral.png
└── README.md
```

## Technical Details

- **OpenCV**: Webcam capture and video processing
- **DeepFace**: Emotion detection using deep learning
- **Tkinter**: GUI framework for the split-screen interface
- **PIL/Pillow**: Image processing and display
- **Threading**: Non-blocking emotion detection

## License

This project is open source and available under the MIT License.