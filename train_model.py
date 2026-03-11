"""
Script to train the IPL Score Prediction Model
Generates sample data and trains a machine learning model
"""

import numpy as np
import pickle
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import random

def generate_sample_data(num_samples=5000):
    """Generate sample training data for the model"""
    
    # Sample players
    batsmen = ['Virat Kohli', 'MS Dhoni', 'Rohit Sharma', 'KL Rahul', 'Suryakumar Yadav', 
               'Shubman Gill', 'Rishabh Pant', 'Hardik Pandya', 'Ravindra Jadeja', 'Sanju Samson']
    
    bowlers = ['Jasprit Bumrah', 'Mohammed Shami', 'Yuzvendra Chahal', 'Ravindra Jadeja',
               'Bhuvneshwar Kumar', 'Kagiso Rabada', 'Trent Boult', 'Rashid Khan', 'Pat Cummins', 'Mohammed Siraj']
    
    stadiums = ['Wankhede Stadium', 'M. Chinnaswamy Stadium', 'Eden Gardens', 'Feroz Shah Kotla',
                'MA Chidambaram Stadium', 'Rajiv Gandhi Stadium', 'Sawai Mansingh Stadium',
                'IS Bindra Stadium', 'Narendra Modi Stadium', 'BRSABV Ekana Stadium']
    
    ball_types = ['regular', 'yorker', 'bouncer', 'slower', 'full_toss', 'wide', 'no_ball']
    
    # Create player performance profiles
    player_profiles = {}
    
    # Initialize all batsmen first
    for batsman in batsmen:
        player_profiles[batsman] = {
            'base_average': random.uniform(5.0, 7.5),
            'stadium_multipliers': {s: random.uniform(0.85, 1.15) for s in stadiums},
            'vs_bowler_multipliers': {}
        }
    
    # Initialize all bowlers and set head-to-head stats
    # Note: Some players might be both batsmen and bowlers (e.g., Ravindra Jadeja)
    for bowler in bowlers:
        if bowler in player_profiles:
            # Player is also a batsman, add bowler stats to existing profile
            player_profiles[bowler]['economy'] = random.uniform(7.0, 9.5)
            player_profiles[bowler]['bowler_stadium_multipliers'] = {s: random.uniform(0.9, 1.1) for s in stadiums}
        else:
            # New bowler only
            player_profiles[bowler] = {
                'economy': random.uniform(7.0, 9.5),
                'bowler_stadium_multipliers': {s: random.uniform(0.9, 1.1) for s in stadiums}
            }
        
        # Set head-to-head multipliers for each batsman against this bowler
        for batsman in batsmen:
            if batsman in player_profiles and 'vs_bowler_multipliers' in player_profiles[batsman]:
                player_profiles[batsman]['vs_bowler_multipliers'][bowler] = random.uniform(0.8, 1.2)
    
    # Generate samples
    X = []
    y = []
    
    for _ in range(num_samples):
        # Random selection
        batsman = random.choice(batsmen)
        bowler = random.choice(bowlers)
        stadium = random.choice(stadiums)
        ball_type = random.choice(ball_types)
        
        pitch_moisture = random.uniform(20, 90)
        current_over = random.uniform(0.1, 19.6)
        current_score = random.randint(0, 200)
        current_wickets = random.randint(0, 9)  # 0-9 wickets (10 wickets means all out)
        
        # Calculate features
        batsman_profile = player_profiles.get(batsman, {})
        bowler_profile = player_profiles.get(bowler, {})
        
        batsman_stadium_avg = batsman_profile.get('base_average', 6.0) * batsman_profile.get('stadium_multipliers', {}).get(stadium, 1.0)
        batsman_strike_rate = 100 + (batsman_stadium_avg - 6) * 20  # Approximate strike rate
        
        # Handle bowlers who might also be batsmen
        bowler_stadium_multipliers = bowler_profile.get('bowler_stadium_multipliers', bowler_profile.get('stadium_multipliers', {}))
        bowler_stadium_avg = bowler_profile.get('economy', 8.0) * bowler_stadium_multipliers.get(stadium, 1.0)
        bowler_wickets = random.uniform(0.05, 0.3)
        
        head_to_head_multiplier = batsman_profile.get('vs_bowler_multipliers', {}).get(bowler, 1.0)
        batsman_vs_bowler_avg = batsman_stadium_avg * head_to_head_multiplier
        batsman_vs_bowler_sr = batsman_strike_rate * head_to_head_multiplier
        
        team_avg_score = random.uniform(6.5, 8.5)
        
        # Ball type multiplier
        ball_type_multiplier = {
            'regular': 1.0,
            'yorker': 0.7,
            'bouncer': 0.8,
            'slower': 1.1,
            'full_toss': 1.3,
            'wide': 0.5,
            'no_ball': 1.2
        }.get(ball_type, 1.0)
        
        # Moisture factor
        moisture_factor = 0.8 + (pitch_moisture / 100) * 0.4
        
        # Over factor
        if current_over >= 16:
            over_factor = 1.2
        elif current_over >= 10:
            over_factor = 1.0
        else:
            over_factor = 0.9
        
        # Wicket pressure factor
        wicket_factor = 1.0 - (current_wickets / 10) * 0.15
        wicket_factor = max(0.7, wicket_factor)
        
        # Calculate target (what the runs should be)
        base_score = (batsman_stadium_avg + batsman_vs_bowler_avg) / 2
        bowler_impact = (bowler_stadium_avg / 10)
        predicted_runs = base_score * (1 - bowler_impact / 2) * ball_type_multiplier * moisture_factor * over_factor * wicket_factor
        
        # Add some randomness
        predicted_runs += random.uniform(-1, 1)
        predicted_runs = max(0, min(6, predicted_runs))
        
        # Round to nearest integer (actual runs)
        actual_runs = int(round(predicted_runs))
        
        # Create feature vector
        features = [
            pitch_moisture,
            batsman_stadium_avg,
            batsman_strike_rate / 100,
            bowler_stadium_avg,
            bowler_wickets,
            batsman_vs_bowler_avg,
            batsman_vs_bowler_sr / 100,
            team_avg_score,
            current_over,
            current_score,
            current_wickets,
            ball_type_multiplier,
            moisture_factor,
            over_factor,
            wicket_factor
        ]
        
        X.append(features)
        y.append(actual_runs)
    
    return np.array(X), np.array(y), player_profiles

def train_model():
    """Train the Random Forest model"""
    print("Generating training data...")
    X, y, player_profiles = generate_sample_data(num_samples=10000)
    
    print(f"Generated {len(X)} samples")
    print(f"Feature shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y.astype(int))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Clip predictions to valid range
    train_pred = np.clip(train_pred, 0, 6)
    test_pred = np.clip(test_pred, 0, 6)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print("\nModel Performance:")
    print(f"Train MAE: {train_mae:.3f}")
    print(f"Test MAE: {test_mae:.3f}")
    print(f"Train R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    
    # Save model
    os.makedirs('model', exist_ok=True)
    model_path = 'model/ipl_score_predictor.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to {model_path}")
    
    # Save initial player data
    os.makedirs('data', exist_ok=True)
    initial_data = {}
    
    # Convert profiles to the format expected by the app
    for player, profile in player_profiles.items():
        if 'base_average' in profile:  # Batsman (or all-rounder)
            initial_data[player] = {
                'stadiums': {
                    stadium: {
                        'average_score': profile['base_average'] * multiplier,
                        'strike_rate': 100 + ((profile['base_average'] * multiplier - 6) * 20),
                        'total_runs': int(profile['base_average'] * multiplier * 100),
                        'total_balls': 100
                    }
                    for stadium, multiplier in profile.get('stadium_multipliers', {}).items()
                },
                'vs_bowlers': {
                    bowler: {
                        'average_score': profile['base_average'] * multiplier,
                        'strike_rate': 100 + ((profile['base_average'] * multiplier - 6) * 20),
                        'total_runs': int(profile['base_average'] * multiplier * 50),
                        'total_balls': 50
                    }
                    for bowler, multiplier in profile.get('vs_bowler_multipliers', {}).items()
                }
            }
        
        # Add bowler stats if player is also a bowler
        if 'economy' in profile:  # Bowler (or all-rounder)
            bowler_stadium_multipliers = profile.get('bowler_stadium_multipliers', profile.get('stadium_multipliers', {}))
            if player not in initial_data:
                initial_data[player] = {
                    'stadiums': {}
                }
            # Update or add bowler stats
            for stadium, multiplier in bowler_stadium_multipliers.items():
                if 'stadiums' not in initial_data[player]:
                    initial_data[player]['stadiums'] = {}
                if stadium not in initial_data[player]['stadiums']:
                    initial_data[player]['stadiums'][stadium] = {}
                initial_data[player]['stadiums'][stadium]['economy'] = profile['economy'] * multiplier
                initial_data[player]['stadiums'][stadium]['wickets_per_over'] = random.uniform(0.1, 0.3)
                initial_data[player]['stadiums'][stadium]['total_runs'] = int(profile['economy'] * multiplier * 20)
                initial_data[player]['stadiums'][stadium]['total_overs'] = 20
    
    data_path = 'data/player_performances.json'
    with open(data_path, 'w') as f:
        json.dump(initial_data, f, indent=2)
    
    print(f"Initial player data saved to {data_path}")
    
    return model

if __name__ == '__main__':
    train_model()
    print("\nTraining completed successfully!")

