# Troubleshooting Guide

## If the app doesn't run, check these:

### 1. Check if Flask is installed
```bash
python -c "import flask; print('Flask version:', flask.__version__)"
```

### 2. Check if model file exists
```bash
python -c "import os; print('Model exists:', os.path.exists('model/ipl_score_predictor.pkl'))"
```

### 3. Try running the app
```bash
python app.py
```

### 4. Check for common errors:

#### Error: "ModuleNotFoundError: No module named 'flask'"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

#### Error: "Model file not found"
**Solution:** Train the model first
```bash
python train_model.py
```

#### Error: "Port 5000 already in use"
**Solution:** Change the port in app.py (line 406)
```python
app.run(debug=True, port=5001)  # Change to 5001 or any other port
```

#### Error: "Address already in use"
**Solution:** Kill the process using port 5000
```bash
# Windows PowerShell
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F
```

### 5. Check browser console for JavaScript errors
- Open browser (Chrome/Firefox)
- Press F12 to open Developer Tools
- Go to Console tab
- Look for any red error messages

### 6. Common JavaScript issues:
- **Button not working:** Check if form elements have correct IDs
- **API not responding:** Check if Flask server is running
- **CORS errors:** Make sure you're accessing via localhost:5000

### 7. Test the endpoints manually:
Open browser and go to:
- `http://localhost:5000` - Should show the form
- `http://localhost:5000/api/players` - Should return JSON with players list

### 8. If still not working:
1. Check the terminal where Flask is running for error messages
2. Share the error message you see
3. Check browser console (F12) for JavaScript errors

