# GitHub Actions Setup — NBA Oracle v4.0

One-time setup to run NBA Oracle in the cloud with no computer required.

---

## Step 1: Create a GitHub Repository

1. Go to https://github.com/new
2. Name it `nba-oracle` (or anything you want)
3. Set it to **Private**
4. Do NOT initialize with a README (we'll push existing code)
5. Click **Create repository**

---

## Step 2: Add Your Discord Webhook as a Secret

1. In your new repo, go to **Settings → Secrets and variables → Actions**
2. Click **New repository secret**
3. Name: `DISCORD_WEBHOOK_URL`
4. Value: your full Discord webhook URL (from your `.env` file)
5. Click **Add secret**

---

## Step 3: Push the Code

Open a terminal in the NBA folder and run:

```bash
cd "c:\Users\kalch\OneDrive\Desktop\Kalshi\NBA"

git init
git add .
git commit -m "NBA Oracle v4.0 initial commit"

# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/nba-oracle.git
git branch -M main
git push -u origin main
```

---

## Step 4: Enable Workflows

1. In your repo on GitHub, click the **Actions** tab
2. If prompted "Workflows aren't being run on this forked repository", click **I understand my workflows, go ahead and enable them**

GitHub Actions will now automatically run:

| Workflow | Schedule | What it does |
|---|---|---|
| Morning Picks | 6:00 AM CST daily | Fetches today's games, runs predictions, sends Discord picks + bet recommendations |
| Nightly Recap | 1:00 AM CST daily | Fetches final scores, sends Discord accuracy recap |

> **Note on Daylight Saving Time**: During CDT (mid-March through early November), messages arrive 1 hour later — 7 AM CDT and 2 AM CDT. GitHub Actions runs in UTC and doesn't adjust for DST.

---

## Step 5: Test Manually

To verify everything works before waiting for the scheduled time:

1. Go to **Actions → Morning Picks** in your repo
2. Click **Run workflow → Run workflow**
3. Watch the logs — you should receive a Discord message within ~2 minutes

---

## How State Persists (No Database Server Needed)

The SQLite database (`data/nba_oracle.db`) is saved as a GitHub Actions artifact after every run and downloaded at the start of the next run. This means:

- Predictions from morning are available for the recap at night
- Elo ratings accumulate across the entire season
- Accuracy tracking works correctly

Artifacts are kept for 7 days, so there's always overlap.

---

## Updating the ML Model

If you want to retrain the model with new data:

```bash
cd "c:\Users\kalch\OneDrive\Desktop\Kalshi\NBA"
python python/build_dataset.py   # fetch latest game data
python python/train_model.py     # retrain and save model artifacts

git add data/model/
git commit -m "retrain model with latest data"
git push
```

The new model artifacts will be picked up by the next GitHub Actions run automatically.
