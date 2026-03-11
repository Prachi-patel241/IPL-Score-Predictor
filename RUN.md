# How to Run IPL Score Predictor

## Quick Start Guide

### Step 1: Install Dependencies
Make sure you have all required Python packages installed:
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model (First Time Only)
Train the machine learning model with sample data:
```bash
python train_model.py
```
This will:
- Generate 10,000 training samples
- Train a Random Forest model
- Save the model to `model/ipl_score_predictor.pkl`
- Create initial player data in `data/player_performances.json`

**Note:** You only need to run this once, or when you want to retrain the model.

### Step 3: Start the Application
Run the Flask server:
```bash
python app.py
```

You should see output like:
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

### Step 4: Open in Browser
Open your web browser and go to:
```
http://localhost:5000
```
or
```
http://127.0.0.1:5000
```

## Using the Application

1. **Fill in the Form:**
   - Select batting and bowling teams
   - Enter batsman and bowler names
   - Choose the stadium
   - Enter current over (e.g., 12.3 for 12 overs 3 balls)
   - Enter current score
   - **Enter current wickets (0-10)** ← New Feature!
   - Adjust pitch moisture (0-100%)
   - Select ball type

2. **Get Prediction:**
   - Click "Predict Score" button
   - View predicted runs (0-6)
   - See probability distribution
   - Check influencing factors

3. **View History:**
   - All predictions are automatically saved
   - Scroll down to see prediction history
   - Each entry shows match details, prediction, and timestamp
   - Use "Clear History" to remove all saved predictions

## Troubleshooting

### Model Not Found Error
If you see an error about missing model file:
```bash
python train_model.py
```

### Port Already in Use
If port 5000 is busy, you can change it in `app.py`:
```python
app.run(debug=True, port=5001)  # Change to any available port
```

### Import Errors
If you get import errors, install dependencies:
```bash
pip install Flask==3.0.0 scikit-learn==1.3.2 numpy==1.24.3 Werkzeug==3.0.1
```

## Features Included

✅ Pitch moisture tracking
✅ Player-specific statistics
✅ Ball type selection
✅ Current over and score tracking
✅ **Current wickets tracking** ← New!
✅ Stadium-specific performance
✅ Head-to-head player statistics
✅ Team vs team performance
✅ **Prediction history/results** ← New!
✅ Machine learning predictions
✅ Probability distributions

## File Structure

```
.
├── app.py                      # Main Flask application
├── train_model.py             # Model training script
├── requirements.txt           # Python dependencies
├── templates/
│   └── index.html            # Frontend HTML
├── static/
│   └── css/
│       └── style.css         # Styling
├── model/
│   └── ipl_score_predictor.pkl  # Trained ML model (generated)
└── data/
    └── player_performances.json  # Player data (generated)
```

## Next Steps

The application will:
- Learn from your predictions and update player statistics
- Store all predictions in browser localStorage
- Improve predictions over time as you use it more

Enjoy predicting IPL scores! 🏏

