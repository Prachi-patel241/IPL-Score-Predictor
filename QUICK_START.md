# Quick Start - Step by Step

## Step 1: Open Terminal/Command Prompt
Navigate to your project folder:
```bash
cd "d:\cursor ai project"
```

## Step 2: Start the Flask Server
```bash
python app.py
```

You should see:
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

**Keep this terminal window open!** The server needs to keep running.

## Step 3: Open Your Browser
Open any web browser (Chrome, Firefox, Edge) and go to:
```
http://localhost:5000
```

OR

```
http://127.0.0.1:5000
```

## Step 4: Test the Application

1. **Fill in the form:**
   - Select Batting Team (e.g., MI)
   - Select Bowling Team (e.g., CSK)
   - Enter Batsman name (e.g., Virat Kohli)
   - Enter Bowler name (e.g., Jasprit Bumrah)
   - Select Stadium
   - Enter Current Over (e.g., 10.0)
   - Enter Current Score (e.g., 100)
   - Enter Current Wickets (e.g., 2)
   - Adjust Pitch Moisture (slider)
   - Select Ball Type

2. **Click "Predict Next Ball"** - Should show predicted runs (0-6)

3. **Click "Predict Final Score"** - Should show predicted final score

## If It Still Doesn't Work:

### Check Browser Console (F12)
1. Press F12 in your browser
2. Go to "Console" tab
3. Look for any red error messages
4. Copy and share the error if you see one

### Check Terminal Output
Look at the terminal where you ran `python app.py` for any error messages.

### Common Issues:

**Issue:** "This site can't be reached" or "Connection refused"
- **Fix:** Make sure Flask server is running (Step 2)

**Issue:** Button does nothing when clicked
- **Fix:** Check browser console (F12) for JavaScript errors

**Issue:** "Model file not found"
- **Fix:** Run `python train_model.py` first

**Issue:** Port 5000 already in use
- **Fix:** Change port in app.py line 406 to 5001, then go to http://localhost:5001

## Still Having Issues?
Share:
1. What error message you see (if any)
2. What happens when you click the buttons
3. Any error from browser console (F12)
4. Any error from terminal

