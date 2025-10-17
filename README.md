# Sign Language to Speech (A-G)

This project converts sign language gestures (A-G) from a live webcam to speech.

## How to Run

1. Initialize dummy model:
```bash
python scripts/init_model.py
```

2. Install dependencies:
```bash
pip install -r app/requirements.txt
```

3. Run Streamlit app:
```bash
streamlit run app/live_demo.py
```

## Supported Gestures
- Letters: A, B, C, D, E, F, G
