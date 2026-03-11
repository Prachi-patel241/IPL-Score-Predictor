# 🏏 IPL Score Predictor

A web-based application for predicting IPL cricket scores using machine learning. The predictor considers multiple factors including pitch conditions, player performance, stadium statistics, and match context.

## Features

- **Real-time Score Prediction**: Predicts runs (0-6) for the next ball
- **Multi-factor Analysis**: Considers:
  - Pitch moisture level
  - Current batsman and bowler
  - Ball type (yorker, bouncer, slower ball, etc.)
  - Current over and score
  - Stadium characteristics
  - Player performance history at specific stadiums
  - Head-to-head player statistics
  - Team vs team performance
- **Beautiful UI**: Modern, responsive web interface
- **Probability Distribution**: Shows likelihood of each possible score (0-6)
- **Learning System**: Updates player statistics based on predictions

## Project Structure

```
.
├── app.py                  # Flask backend application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Frontend HTML
├── static/
│   └── css/
│       └── style.css      # Styling
├── model/
│   └── ipl_score_predictor.pkl  # Trained ML model (generated)
└── data/
    └── player_performances.json  # Player performance data (generated)
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

First, train the machine learning model:

```bash
python train_model.py
```

This will:
- Generate sample training data
- Train a Random Forest model
- Save the model to `model/ipl_score_predictor.pkl`
- Create initial player performance data

### 3. Run the Application

Start the Flask server:

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## How to Use

1. **Select Teams**: Choose the batting and bowling teams
2. **Enter Players**: Input the names of the batsman on strike and the bowler
3. **Select Stadium**: Choose the match venue
4. **Match Status**: Enter current over (e.g., 12.3) and current score
5. **Pitch Conditions**: Adjust the pitch moisture slider (0-100%)
6. **Ball Type**: Select the type of delivery
7. **Predict**: Click "Predict Score" to get the prediction

## Model Features

The model uses 13 features:
1. Pitch moisture level
2. Batsman's average at stadium
3. Batsman's strike rate at stadium
4. Bowler's economy at stadium
5. Bowler's wickets per over at stadium
6. Head-to-head average (batsman vs bowler)
7. Head-to-head strike rate
8. Team average score per over
9. Current over
10. Current score
11. Ball type multiplier
12. Moisture factor
13. Over phase factor

## Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn (Random Forest Regressor)
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Storage**: JSON files

## Customization

### Adding New Players

The system automatically learns from predictions. Simply use new player names in predictions, and the system will track their performance over time.

### Retraining the Model

To retrain with more data:

```bash
python train_model.py
```

You can modify `train_model.py` to:
- Add more training samples
- Use different algorithms
- Include additional features

## Performance

The model provides predictions with probability distributions for each possible score (0-6 runs). The system continuously improves by updating player statistics based on actual predictions made.

## Future Enhancements

- Integration with real IPL match data
- More sophisticated ML models (Neural Networks, XGBoost)
- Historical match analysis
- Team composition recommendations
- Player performance dashboards

## License

This project is for educational purposes.

## Contact

For questions or suggestions, please open an issue or submit a pull request.

