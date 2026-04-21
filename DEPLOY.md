# Phase 5 — Deployment Guide
## Deploy Neural Digit Engine to Render (Free)

---

## Why Render?
- Free tier available
- Supports C++ compilation
- Auto-deploys from GitHub
- No credit card needed

---

## Step 1 — Push to GitHub

```bash
# In your project folder
git init
git add .
git commit -m "Neural Digit Engine - full stack from scratch"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/neural-digit-engine.git
git push -u origin main
```

> Make sure weights/ folder is committed too (it has your trained model)

---

## Step 2 — Create Render Account
1. Go to https://render.com
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repo

---

## Step 3 — Configure Render Settings

| Setting | Value |
|---------|-------|
| Name | neural-digit-engine |
| Environment | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn backend.app:app` |
| Instance Type | Free |

> Render will auto-compile C++ via the app.py startup script

---

## Step 4 — Set Environment Variable (optional)
In Render dashboard → Environment:
```
PYTHON_VERSION = 3.11.0
```

---

## Step 5 — Deploy!
Click "Create Web Service"
Render will:
1. Install Python packages
2. Start Flask app
3. Flask auto-compiles engine.cpp on first request
4. Your app is live at: https://neural-digit-engine.onrender.com

---

## Your Live URLs
```
Homepage  : https://neural-digit-engine.onrender.com
API       : https://neural-digit-engine.onrender.com/predict
Health    : https://neural-digit-engine.onrender.com/health
```

---

## Project Structure for Deployment
```
neural-digit-engine/          ← root (git repo)
│
├── backend/
│   └── app.py                ← Flask app (entry point)
│
├── cpp/
│   └── engine.cpp            ← C++ source (compiled on server)
│
├── frontend/
│   └── index.html            ← served at /
│
├── weights/
│   ├── fc1_weight.txt        ← MUST be committed to git
│   ├── fc1_bias.txt
│   ├── fc2_weight.txt
│   └── fc2_bias.txt
│
├── requirements.txt          ← flask, gunicorn, flask-cors
├── Procfile                  ← web: gunicorn backend.app:app
└── .gitignore
```

---

## Important: Weights Must Be in Git
The trained weight files MUST be committed to GitHub.
They are your model — without them the C++ engine has nothing to load.

```bash
# Make sure weights are tracked
git add weights/
git status   # should show weights/*.txt as committed
```

---

## Local Development
```bash
# Terminal 1 — Flask backend
pip install flask flask-cors
python backend/app.py

# Open in browser
open frontend/index.html
# OR visit http://localhost:5000 (Flask also serves frontend)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Engine not found | Check cpp/engine.cpp path is correct |
| Weights missing | Commit weights/ folder to git |
| 500 error | Check Render logs for g++ compile error |
| CORS error | Flask-cors is installed in requirements.txt |
| Slow first load | Render free tier sleeps — wait 30 seconds |

---

## Free Tier Note
Render free tier sleeps after 15 mins of inactivity.
First visit after sleep takes ~30 seconds to wake up.
This is normal — just tell your interviewer "it's on free hosting."
