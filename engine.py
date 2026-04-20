from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import joblib
import numpy as np
import uvicorn
import yfinance as yf
from collections import deque
import time
import sys
import math

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

app = FastAPI()

# ==========================================
# CRITICAL ADDITION: CORS MIDDLEWARE
# This allows your index.html file to communicate with this Python server
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (essential for HTML frontend)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. LOAD AI MODELS
# Ensure your models are inside a folder named 'models'
# ==========================================
try:
    # `compile=False` avoids deserializing training-only objects from legacy H5 files.
    lstm_model = tf.keras.models.load_model('models/lstm_model.h5', compile=False)
    xgb_model = joblib.load('models/xgboost_model.pkl')
    xgb_feature_count = int(getattr(xgb_model, "n_features_in_", 10))
    print("✅ Models loaded successfully from 'models/' directory.")
except Exception as e:
    print(f"⚠️ WARNING: Could not load models. Error: {e}")
    # Fallback dummy models so the server doesn't crash during UI testing
    class DummyModel:
        def predict(self, *args, **kwargs):
            return np.array([[0.015]]) if len(args[0].shape) == 3 else np.array([0.012])
    lstm_model = DummyModel()
    xgb_model = DummyModel()
    xgb_feature_count = 10

# ==========================================
# 2. SYSTEM STATE & ADAPTIVE MATH (TCP-RM)
# ==========================================
class SystemState:
    def __init__(self):
        self.c_t = 1.96 
        self.mean_v = 0.0
        self.m2_v = 0.0
        self.count = 0
        self.history = deque(maxlen=50)

state = SystemState()

DEFAULT_MARKET_DATA = {
    "nifty_it": 35240.00,
    "tcs": 3950.00,
    "infy": 1480.00,
    "wipro": 492.00,
}
MARKET_REFRESH_INTERVAL = 15.0

state.market_snapshot = DEFAULT_MARKET_DATA.copy()
state.market_source = "SIMULATED"
state.last_market_refresh = time.time()

def safe_float(value, fallback=0.0):
    """Return a JSON-safe finite float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return number if math.isfinite(number) else float(fallback)

def sanitize_market_data(data):
    return {
        key: round(safe_float(data.get(key), fallback), 2)
        for key, fallback in DEFAULT_MARKET_DATA.items()
    }

def sanitize_packet(packet):
    sanitized = {}
    for key, value in packet.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            sanitized[key] = safe_float(value)
        else:
            sanitized[key] = value
    return sanitized

def update_welford_variance(error):
    error = safe_float(error)
    state.count += 1
    delta = error - state.mean_v
    state.mean_v += delta / state.count
    state.m2_v += delta * (error - state.mean_v)
    variance = state.m2_v / state.count if state.count > 1 else 1e-6
    return max(safe_float(variance, 1e-6), 1e-6)

def simulate_market_data(previous_data):
    """Keep the dashboard lively between slower live market refreshes."""
    previous = sanitize_market_data(previous_data)
    sector_move = safe_float(np.random.normal(0, 0.0007), 0.0)

    move_map = {
        "nifty_it": sector_move + safe_float(np.random.normal(0, 0.0002), 0.0),
        "tcs": sector_move * 0.90 + safe_float(np.random.normal(0, 0.0010), 0.0),
        "infy": sector_move * 1.05 + safe_float(np.random.normal(0, 0.0012), 0.0),
        "wipro": sector_move * 1.15 + safe_float(np.random.normal(0, 0.0014), 0.0),
    }

    next_snapshot = {}
    for key, price in previous.items():
        anchor = DEFAULT_MARKET_DATA[key]
        moved_price = price * (1 + move_map[key])
        bounded_price = min(anchor * 1.40, max(anchor * 0.60, moved_price))
        next_snapshot[key] = round(bounded_price, 2)

    return sanitize_market_data(next_snapshot)

def get_market_data():
    """
    Return market data quickly on every tick.
    Live quotes refresh occasionally; intermediate ticks animate from the latest snapshot.
    """
    now = time.time()
    refreshed_live = False

    if now - state.last_market_refresh >= MARKET_REFRESH_INTERVAL:
        try:
            state.market_snapshot = fetch_live_market_data()
            state.market_source = "LIVE"
            refreshed_live = True
        except Exception:
            state.market_source = "SIMULATED"
        finally:
            state.last_market_refresh = now

    if not refreshed_live:
        state.market_snapshot = simulate_market_data(state.market_snapshot)

    return state.market_snapshot.copy(), state.market_source

# ==========================================
# 3. LIVE MARKET DATA INGESTION
# ==========================================
def fetch_live_market_data():
    """Fetches live NIFTY IT and component data from Yahoo Finance."""
    tickers = ["^CNXIT", "TCS.NS", "INFY.NS", "WIPRO.NS"]
    data = yf.download(tickers, period="1d", interval="1m", progress=False)['Close']
    
    if data.empty:
        raise ValueError("Market Closed or No Data Available")
        
    latest = data.ffill().iloc[-1]
    
    return sanitize_market_data({
        "nifty_it": round(float(latest["^CNXIT"]), 2),
        "tcs": round(float(latest["TCS.NS"]), 2),
        "infy": round(float(latest["INFY.NS"]), 2),
        "wipro": round(float(latest["WIPRO.NS"]), 2)
    })

# ==========================================
# 4. THE API ENDPOINT (Listens to the HTML file)
# ==========================================
@app.get("/api/tick")
async def process_tick():
    # 1. Fetch Real Market Data
    market_data, market_source = get_market_data()

    # 2. Model Inputs (For now, simulating the arrays until you map real history)
    mock_log_ret = safe_float(np.random.normal(0, 0.012))
    seq = np.random.rand(1, 60, 1)
    feat = np.random.rand(1, xgb_feature_count)
    
    # 3. Hybrid Inference
    lstm_p = safe_float(lstm_model.predict(seq, verbose=0)[0][0], 0.015)
    xgb_p = safe_float(xgb_model.predict(feat)[0], 0.012)
    
    # Calculate Agreement (Clamp between 0 and 1)
    agreement = 1 - (abs(lstm_p - xgb_p) / max(abs(lstm_p), 1e-6))
    agreement = max(0.0, min(1.0, agreement))
    
    # 4. Apply Adaptive Mathematics (TCP-RM)
    var = update_welford_variance(mock_log_ret)
    std_dev = safe_float(np.sqrt(var), 1e-3)
    upper_bound = safe_float(state.c_t * std_dev, 1e-3)
    
    # 5. Regime Classification
    status = "STABLE"
    if abs(mock_log_ret) > upper_bound: 
        status = "CRITICAL RISK"
        state.c_t = min(3.5, state.c_t + 0.05) # Widening Bounds
    elif lstm_p > std_dev * 1.2: 
        status = "RISING UNCERTAINTY"
    else:
        state.c_t = max(1.5, state.c_t - 0.005) # Tightening Bounds
    
    # 6. JSON Packet Delivery
    packet = sanitize_packet({
        "time": time.strftime('%H:%M:%S'),
        "nifty_it": market_data["nifty_it"],
        "tcs": market_data["tcs"],
        "infy": market_data["infy"],
        "wipro": market_data["wipro"],
        "log_return": mock_log_ret, 
        "volatility": lstm_p,
        "upper_bound": upper_bound,
        "confidence": agreement,
        "threshold": state.c_t,
        "data_source": market_source,
        "status": status
    })
    state.history.append(packet)
    
    return {"current": packet, "history": list(state.history)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
