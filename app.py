from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import json
import os
from datetime import datetime

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model/ipl_score_predictor.pkl'
DATA_PATH = 'data/player_performances.json'

# Load player performance data
def load_player_data():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_player_data(data):
    os.makedirs('data', exist_ok=True)
    with open(DATA_PATH, 'w') as f:
        json.dump(data, f, indent=2)

# Load model if exists
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()
player_data = load_player_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features
        pitch_moisture = float(data.get('pitch_moisture', 50))
        bowler = data.get('bowler', '')
        batsman = data.get('batsman', '')
        ball_type = data.get('ball_type', 'regular')
        current_over = float(data.get('current_over', 0))
        current_score = int(data.get('current_score', 0))
        current_wickets = int(data.get('current_wickets', 0))
        stadium = data.get('stadium', '')
        batting_team = data.get('batting_team', '')
        bowling_team = data.get('bowling_team', '')
        
        # Get player performance data
        batsman_performance = player_data.get(batsman, {})
        bowler_performance = player_data.get(bowler, {})
        
        # Stadium performance for batsman
        batsman_stadium_avg = batsman_performance.get('stadiums', {}).get(stadium, {}).get('average_score', 6.0)
        batsman_stadium_strike_rate = batsman_performance.get('stadiums', {}).get(stadium, {}).get('strike_rate', 120)
        
        # Stadium performance for bowler
        bowler_stadium_avg = bowler_performance.get('stadiums', {}).get(stadium, {}).get('economy', 8.0)
        bowler_stadium_wickets = bowler_performance.get('stadiums', {}).get(stadium, {}).get('wickets_per_over', 0.1)
        
        # Head-to-head performance
        batsman_vs_bowler = batsman_performance.get('vs_bowlers', {}).get(bowler, {})
        batsman_vs_bowler_avg = batsman_vs_bowler.get('average_score', 6.0)
        batsman_vs_bowler_strike_rate = batsman_vs_bowler.get('strike_rate', 120)
        
        # Team vs team performance
        batting_team_vs_bowling = player_data.get('_team_stats', {}).get(f"{batting_team}_vs_{bowling_team}", {})
        team_avg_score = batting_team_vs_bowling.get('average_score_per_over', 7.0)
        
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
        
        # Pitch moisture impact (higher moisture = easier to bat)
        moisture_factor = 0.8 + (pitch_moisture / 100) * 0.4
        
        # Over pressure factor (death overs have different scoring)
        if current_over >= 16:
            over_factor = 1.2  # Death overs - higher scoring
        elif current_over >= 10:
            over_factor = 1.0  # Middle overs
        else:
            over_factor = 0.9  # Powerplay
        
        # Wicket pressure factor (more wickets = more pressure, different approach)
        wicket_factor = 1.0 - (current_wickets / 10) * 0.15  # Each wicket reduces scoring by 1.5%
        wicket_factor = max(0.7, wicket_factor)  # Minimum 70% even at 10 wickets
        
        # Calculate base prediction
        if model:
            # Use trained model
            features = np.array([[
                pitch_moisture,
                batsman_stadium_avg,
                batsman_stadium_strike_rate / 100,
                bowler_stadium_avg,
                bowler_stadium_wickets,
                batsman_vs_bowler_avg,
                batsman_vs_bowler_strike_rate / 100,
                team_avg_score,
                current_over,
                current_score,
                current_wickets,
                ball_type_multiplier,
                moisture_factor,
                over_factor,
                wicket_factor
            ]])
            
            predicted_runs = model.predict(features)[0]
            probability_distribution = None
            
            # For probability, use a simple heuristic
            runs_probabilities = {}
            for runs in range(0, 7):
                if runs == int(predicted_runs):
                    runs_probabilities[runs] = 0.35
                elif abs(runs - predicted_runs) == 1:
                    runs_probabilities[runs] = 0.25
                elif abs(runs - predicted_runs) == 2:
                    runs_probabilities[runs] = 0.15
                else:
                    runs_probabilities[runs] = 0.05
            
        else:
            # Fallback heuristic model
            base_score = (batsman_stadium_avg + batsman_vs_bowler_avg) / 2
            bowler_impact = (bowler_stadium_avg / 10)  # Lower economy = lower runs
            predicted_runs = base_score * (1 - bowler_impact / 2) * ball_type_multiplier * moisture_factor * over_factor * wicket_factor
            predicted_runs = max(0, min(6, round(predicted_runs)))
            
            # Simple probability distribution
            runs_probabilities = {}
            for runs in range(0, 7):
                if runs == predicted_runs:
                    runs_probabilities[runs] = 0.40
                elif abs(runs - predicted_runs) == 1:
                    runs_probabilities[runs] = 0.30
                elif abs(runs - predicted_runs) == 2:
                    runs_probabilities[runs] = 0.20
                else:
                    runs_probabilities[runs] = 0.10 / (6 - abs(runs - predicted_runs))
        
        # Update player performance data after prediction (for learning)
        update_player_stats(batsman, bowler, stadium, batting_team, bowling_team, predicted_runs)
        
        return jsonify({
            'success': True,
            'predicted_runs': int(predicted_runs),
            'probabilities': runs_probabilities,
                'factors': {
                'batsman_stadium_performance': round(batsman_stadium_avg, 2),
                'bowler_stadium_performance': round(bowler_stadium_avg, 2),
                'head_to_head': round(batsman_vs_bowler_avg, 2),
                'pitch_condition': 'Dry' if pitch_moisture < 40 else 'Normal' if pitch_moisture < 70 else 'Moist',
                'over_phase': 'Powerplay' if current_over < 6 else 'Middle' if current_over < 16 else 'Death',
                'wicket_pressure': f'{current_wickets} wickets down'
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def update_player_stats(batsman, bowler, stadium, batting_team, bowling_team, runs):
    """Update player performance statistics"""
    global player_data
    
    if batsman not in player_data:
        player_data[batsman] = {
            'stadiums': {},
            'vs_bowlers': {},
            'total_runs': 0,
            'total_balls': 0
        }
    
    if bowler not in player_data:
        player_data[bowler] = {
            'stadiums': {},
            'total_runs_conceded': 0,
            'total_overs': 0
        }
    
    # Update batsman stats
    if stadium not in player_data[batsman]['stadiums']:
        player_data[batsman]['stadiums'][stadium] = {
            'total_runs': 0,
            'total_balls': 0,
            'average_score': 0,
            'strike_rate': 0
        }
    
    player_data[batsman]['stadiums'][stadium]['total_runs'] += runs
    player_data[batsman]['stadiums'][stadium]['total_balls'] += 1
    player_data[batsman]['stadiums'][stadium]['average_score'] = (
        player_data[batsman]['stadiums'][stadium]['total_runs'] /
        player_data[batsman]['stadiums'][stadium]['total_balls']
    )
    player_data[batsman]['stadiums'][stadium]['strike_rate'] = (
        player_data[batsman]['stadiums'][stadium]['average_score'] * 100
    )
    
    # Update head-to-head
    if bowler not in player_data[batsman]['vs_bowlers']:
        player_data[batsman]['vs_bowlers'][bowler] = {
            'total_runs': 0,
            'total_balls': 0,
            'average_score': 0,
            'strike_rate': 0
        }
    
    player_data[batsman]['vs_bowlers'][bowler]['total_runs'] += runs
    player_data[batsman]['vs_bowlers'][bowler]['total_balls'] += 1
    player_data[batsman]['vs_bowlers'][bowler]['average_score'] = (
        player_data[batsman]['vs_bowlers'][bowler]['total_runs'] /
        player_data[batsman]['vs_bowlers'][bowler]['total_balls']
    )
    player_data[batsman]['vs_bowlers'][bowler]['strike_rate'] = (
        player_data[batsman]['vs_bowlers'][bowler]['average_score'] * 100
    )
    
    # Update bowler stats
    if stadium not in player_data[bowler]['stadiums']:
        player_data[bowler]['stadiums'][stadium] = {
            'total_runs': 0,
            'total_overs': 0,
            'economy': 0,
            'wickets_per_over': 0
        }
    
    player_data[bowler]['stadiums'][stadium]['total_runs'] += runs
    player_data[bowler]['stadiums'][stadium]['total_overs'] += 1/6  # 1 ball = 1/6 over
    player_data[bowler]['stadiums'][stadium]['economy'] = (
        player_data[bowler]['stadiums'][stadium]['total_runs'] /
        player_data[bowler]['stadiums'][stadium]['total_overs']
        if player_data[bowler]['stadiums'][stadium]['total_overs'] > 0 else 8.0
    )
    
    # Update team stats
    if '_team_stats' not in player_data:
        player_data['_team_stats'] = {}
    
    team_key = f"{batting_team}_vs_{bowling_team}"
    if team_key not in player_data['_team_stats']:
        player_data['_team_stats'][team_key] = {
            'total_runs': 0,
            'total_overs': 0,
            'average_score_per_over': 0
        }
    
    player_data['_team_stats'][team_key]['total_runs'] += runs
    player_data['_team_stats'][team_key]['total_overs'] += 1/6
    player_data['_team_stats'][team_key]['average_score_per_over'] = (
        player_data['_team_stats'][team_key]['total_runs'] /
        player_data['_team_stats'][team_key]['total_overs']
        if player_data['_team_stats'][team_key]['total_overs'] > 0 else 7.0
    )
    
    save_player_data(player_data)

@app.route('/api/predict-final', methods=['POST'])
def predict_final_score():
    """Predict the final score of the innings"""
    try:
        data = request.json
        
        # Extract features
        current_over = float(data.get('current_over', 0))
        current_score = int(data.get('current_score', 0))
        current_wickets = int(data.get('current_wickets', 0))
        pitch_moisture = float(data.get('pitch_moisture', 50))
        stadium = data.get('stadium', '')
        batting_team = data.get('batting_team', '')
        bowling_team = data.get('bowling_team', '')
        
        # Calculate remaining overs
        total_overs = 20.0
        remaining_overs = max(0, total_overs - current_over)
        # Calculate remaining balls (convert overs to balls)
        remaining_balls = int(remaining_overs * 6)
        
        if remaining_overs <= 0:
            return jsonify({
                'success': True,
                'final_score': current_score,
                'message': 'Innings already completed'
            })
        
        # Calculate current run rate
        if current_over > 0:
            current_run_rate = current_score / current_over
        else:
            current_run_rate = 7.0  # Default
        
        # Calculate wickets remaining
        wickets_remaining = 10 - current_wickets
        
        # Get team statistics
        batting_team_vs_bowling = player_data.get('_team_stats', {}).get(f"{batting_team}_vs_{bowling_team}", {})
        team_avg_score = batting_team_vs_bowling.get('average_score_per_over', 7.0)
        
        # Pitch moisture impact
        moisture_factor = 0.8 + (pitch_moisture / 100) * 0.4
        
        # Over phase impact (death overs have higher scoring)
        if current_over >= 16:
            # Already in death overs, maintain high rate
            expected_run_rate = max(current_run_rate, 8.5) * moisture_factor
        elif current_over >= 10:
            # Middle overs transitioning to death
            expected_run_rate = (team_avg_score * 0.9 + current_run_rate * 0.1) * 1.1 * moisture_factor
        else:
            # Early/middle overs
            expected_run_rate = (team_avg_score * 0.7 + current_run_rate * 0.3) * moisture_factor
        
        # Wicket pressure impact (fewer wickets = more conservative)
        if wickets_remaining <= 3:
            wicket_adjustment = 0.85  # Very conservative with few wickets
        elif wickets_remaining <= 5:
            wicket_adjustment = 0.92  # Slightly conservative
        elif wickets_remaining <= 7:
            wicket_adjustment = 0.97  # Minor adjustment
        else:
            wicket_adjustment = 1.0  # Full scoring potential
        
        # Calculate predicted runs for remaining overs
        predicted_runs_remaining = expected_run_rate * remaining_overs * wicket_adjustment
        
        # Add some variance based on match situation
        if current_over >= 16:
            # Death overs - higher variance
            variance = np.random.normal(0, predicted_runs_remaining * 0.15)
        else:
            variance = np.random.normal(0, predicted_runs_remaining * 0.10)
        
        predicted_runs_remaining = max(0, predicted_runs_remaining + variance)
        
        # Calculate final score
        predicted_final_score = int(current_score + predicted_runs_remaining)
        
        # Calculate confidence range (±10% of predicted runs remaining)
        confidence_range = int(predicted_runs_remaining * 0.1)
        min_score = int(current_score + predicted_runs_remaining - confidence_range)
        max_score = int(current_score + predicted_runs_remaining + confidence_range)
        
        # Calculate projected run rate for remaining overs
        if remaining_overs > 0:
            projected_run_rate = predicted_runs_remaining / remaining_overs
        else:
            projected_run_rate = 0
        
        return jsonify({
            'success': True,
            'final_score': predicted_final_score,
            'min_score': max(min_score, current_score),
            'max_score': max_score,
            'runs_remaining': int(predicted_runs_remaining),
            'remaining_overs': round(remaining_overs, 1),
            'remaining_balls': remaining_balls,
            'current_run_rate': round(current_run_rate, 2),
            'projected_run_rate': round(projected_run_rate, 2),
            'wickets_remaining': wickets_remaining,
            'factors': {
                'pitch_condition': 'Dry' if pitch_moisture < 40 else 'Normal' if pitch_moisture < 70 else 'Moist',
                'over_phase': 'Powerplay' if current_over < 6 else 'Middle' if current_over < 16 else 'Death',
                'team_average': round(team_avg_score, 2),
                'wicket_adjustment': round(wicket_adjustment * 100, 1)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/players', methods=['GET'])
def get_players():
    """Get list of players from stored data"""
    players = list(set([p for p in player_data.keys() if not p.startswith('_')]))
    return jsonify({'players': sorted(players)})

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    app.run(debug=True, port=5000)

