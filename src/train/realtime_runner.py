"""
Real-time Discord Bot for GPT-2 Time Series Predictions
Sends hourly predictions for all configured coin pairs
"""

import os
import sys
import json
import time
import asyncio
import discord
from discord.ext import tasks
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import logging
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports (align with visualizer)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_manager import COIN_PAIRS
from model import GPT2TimeSeries
from config import build_config

# ==================== CONFIGURATION ====================
DISCORD_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"  # Replace with your Discord bot token
DISCORD_CHANNEL_ID = 123456789012345678    # Replace with your channel ID

# Model paths
CHECKPOINT_PATH = Path("runs/gpt2ts/checkpoints/checkpoint_step_495000.pt")
TOKENIZER_DIR = Path("tokenizer")

# Trading parameters (matching original script defaults)
CONTEXT_LENGTH = 100
PREDICTION_LENGTH = 15
ATR_PERIOD = 14
SWING_THRESHOLD_ATR = 1.5  # Multiplier for ATR to determine trade signals
ENSEMBLE_BACKSHIFTS = 5
ENSEMBLE_WEIGHTS = [5, 4, 3, 2, 1]

# Timezone
TZ = pytz.timezone("America/Los_Angeles")

# ========================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictionEngine:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = None
        self._vocab_token_ids = None
        self._vocab_close_vals = None
        self.config = None
        self.prediction_cache = {}
        
        self.load_model_and_vocab()
    
    def load_model_and_vocab(self):
        """Load the trained model and vocabulary"""
        try:
            logger.info("Loading model checkpoint...")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
            
            self.config = build_config()
            
            vocab_path = TOKENIZER_DIR / "vocab.json"
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
            self.config.model.vocab_size = len(self.vocab)
            
            sorted_keys = sorted(self.vocab.keys(), key=lambda x: int(x))
            self._vocab_token_ids = np.array([int(k) for k in sorted_keys], dtype=np.int32)
            self._vocab_close_vals = np.array(
                [float(self.vocab[k].get('close', 0.5)) for k in sorted_keys],
                dtype=np.float32
            )
            
            self.model = GPT2TimeSeries(self.config).to(self.device)
            
            if 'state' in checkpoint and 'model' in checkpoint['state']:
                self.model.load_state_dict(checkpoint['state']['model'])
            
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def fetch_binance_klines(self, symbol: str, interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
        """Fetch latest klines from Binance; drop still-forming last candle."""
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(max(limit, 10), 1000)
        }
        
        for attempt in range(3):  # Retry logic
            try:
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    break
                time.sleep(1)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)
        
        data = resp.json()
        
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume","close_time",
            "quote_asset_volume","number_of_trades","taker_buy_base",
            "taker_buy_quote","ignore"
        ])
        
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        # Drop still-forming last candle (align with visualizer/realtime runner)
        df["open_time_dt_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time_dt_utc"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        now_utc = pd.Timestamp.now(tz=pytz.UTC)
        df = df[df["close_time_dt_utc"] <= now_utc]

        df["date"] = df["open_time_dt_utc"].dt.tz_convert(TZ)
        
        return df[["date","open","high","low","close","volume"]].reset_index(drop=True)
    
    def norm_close_to_token(self, norm_close: float) -> int:
        """Map normalized close to nearest token"""
        x = float(np.clip(norm_close, 0.0, 1.0))
        idx = int(np.argmin(np.abs(self._vocab_close_vals - x)))
        return int(self._vocab_token_ids[idx])
    
    def token_to_norm_close(self, token_id: int) -> float:
        """Map token to normalized close"""
        token_str = str(token_id)
        if self.vocab and token_str in self.vocab and 'close' in self.vocab[token_str]:
            return float(self.vocab[token_str]['close'])
        return 0.5
    
    def compute_atr(self, df: pd.DataFrame, end_index: int, period: int = ATR_PERIOD) -> float:
        """Compute ATR"""
        if len(df) < 2:
            return 0.0
        
        end_index = min(max(1, end_index), len(df) - 1)
        start_index = max(1, end_index - period + 1)
        
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        
        trs = []
        for i in range(start_index, end_index + 1):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i - 1]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        
        return float(np.mean(trs)) if trs else 0.0
    
    def classify_by_atr(self, path: List[float], atr_val: float) -> str:
        """Classify prediction as bullish/bearish/neutral based on ATR"""
        if not path or len(path) < 2:
            return "neutral"
        
        start_price = float(path[0])
        end_price = float(path[-1])
        delta = end_price - start_price
        
        if atr_val <= 1e-12 or start_price <= 0:
            return "neutral"
        
        norm_move = delta / atr_val
        
        if norm_move >= SWING_THRESHOLD_ATR:
            return "bullish"
        elif norm_move <= -SWING_THRESHOLD_ATR:
            return "bearish"
        else:
            return "neutral"
    
    def rollout_greedy(self, context_tokens: List[int], steps: int) -> List[int]:
        """Greedy rollout from context"""
        current_context = context_tokens[-CONTEXT_LENGTH:].copy()
        predicted_tokens = []
        
        with torch.no_grad():
            for _ in range(steps):
                context_tensor = torch.tensor([current_context], dtype=torch.long, device=self.device)
                logits = self.model(context_tensor)
                next_token = torch.argmax(logits[0, -1, :]).item()
                predicted_tokens.append(next_token)
                
                current_context.append(next_token)
                if len(current_context) > 200:
                    current_context = current_context[-200:]
        
        return predicted_tokens

    def ensemble_predict(self, symbol: str) -> Tuple[str, float, str]:
        """Generate ensemble prediction for a symbol (aligned with visualizer logic)."""
        try:
            # Fetch data (with incomplete candle dropped)
            df = self.fetch_binance_klines(symbol, "1h", 500)
            
            # Prepare tokens + per-index norm frame (roll 100)
            close = df["close"]
            roll_min = close.rolling(window=100, min_periods=1).min()
            roll_max = close.rolling(window=100, min_periods=1).max()
            denom = (roll_max - roll_min).replace(0.0, 1e-9)
            norm_close = ((close - roll_min) / denom).clip(0.0, 1.0)
            tokens = norm_close.apply(self.norm_close_to_token).values
            norm_data = pd.DataFrame({"price_min": roll_min.values, "price_max": roll_max.values})

            # Choose position exactly like the visualizer:
            position = len(tokens) - CONTEXT_LENGTH - PREDICTION_LENGTH
            if position < 0:
                return "neutral", float(df["close"].iloc[-1]), "HOLD"

            context_tokens = tokens[position:position + CONTEXT_LENGTH].tolist()
            base_norm_start = position + CONTEXT_LENGTH

            # Ensemble prediction with backshifts
            valid_ks = []
            for k in range(1, ENSEMBLE_BACKSHIFTS + 1):
                if position - k >= 0:
                    valid_ks.append(k)
            
            if not valid_ks:
                # Single prediction
                predicted_tokens = self.rollout_greedy(context_tokens, PREDICTION_LENGTH)
                predicted_closes = []
                for i, tok in enumerate(predicted_tokens):
                    norm_val = self.token_to_norm_close(tok)
                    idx = min(base_norm_start + i, len(norm_data) - 1)
                    row = norm_data.iloc[idx]
                    price = norm_val * (row["price_max"] - row["price_min"]) + row["price_min"]
                    predicted_closes.append(price)
            else:
                # Weighted ensemble (k=1..5 with weights)
                weights = np.array(ENSEMBLE_WEIGHTS[:len(valid_ks)], dtype=np.float64)
                weights = weights / weights.sum()
                
                per_k_paths = []
                token_end = position + CONTEXT_LENGTH
                for k in valid_ks:
                    ctx_start = position - k
                    ctx_end = token_end - k
                    context_k = tokens[ctx_start:ctx_end].tolist()
                    
                    total_steps = PREDICTION_LENGTH + k
                    tokens_generated = self.rollout_greedy(context_k, total_steps)
                    kept_tokens = tokens_generated[k:k + PREDICTION_LENGTH]
                    
                    path_prices = []
                    for j, tok in enumerate(kept_tokens):
                        norm_val = self.token_to_norm_close(tok)
                        idx = min(base_norm_start + j, len(norm_data) - 1)
                        row = norm_data.iloc[idx]
                        price = norm_val * (row["price_max"] - row["price_min"]) + row["price_min"]
                        path_prices.append(price)
                    
                    per_k_paths.append(path_prices)
                
                paths_array = np.array(per_k_paths, dtype=np.float64)
                predicted_closes = (weights[:, None] * paths_array).sum(axis=0).tolist()
            
            # Classify trend using ATR at the SAME boundary as the visualizer
            boundary_idx = position + CONTEXT_LENGTH - 1
            atr_val = self.compute_atr(df, boundary_idx, ATR_PERIOD)
            trend = self.classify_by_atr(predicted_closes, atr_val)
            
            # Display the boundary close (same candle the visualizer uses for entry)
            boundary_price = float(df.iloc[boundary_idx]["close"])
            display_trend = "**LONG**" if trend == "bullish" else "**SHORT**" if trend == "bearish" else "HOLD"
            
            return trend, boundary_price, display_trend
            
        except Exception as e:
            logger.error(f"Error predicting {symbol}: {e}")
            return "neutral", 0.0, "ERROR"

class DiscordBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.engine = PredictionEngine()
        self.channel = None
    
    async def on_ready(self):
        logger.info(f'Bot logged in as {self.user}')
        self.channel = self.get_channel(DISCORD_CHANNEL_ID)
        if not self.channel:
            logger.error(f"Could not find channel {DISCORD_CHANNEL_ID}")
            return
        
        # Start hourly task
        self.hourly_predictions.start()
    
    @tasks.loop(minutes=60)
    async def hourly_predictions(self):
        """Send predictions every hour"""
        try:
            # Wait 5 seconds after the hour to ensure API data is ready
            now = datetime.now(TZ)
            if now.minute == 0:
                await asyncio.sleep(5)
            
            logger.info(f"Generating predictions at {now}")
            
            predictions = []
            for symbol in sorted(COIN_PAIRS):  # Alphabetical order
                trend, price, display = self.engine.ensemble_predict(symbol)
                
                # Format price based on magnitude
                if price >= 1000:
                    price_str = f"{price:,.2f}"
                elif price >= 10:
                    price_str = f"{price:.2f}"
                elif price >= 1:
                    price_str = f"{price:.3f}"
                elif price >= 0.01:
                    price_str = f"{price:.4f}"
                else:
                    price_str = f"{price:.5f}"
                
                predictions.append(f"{symbol}: {display} (price: {price_str})")
                
                # Small delay between API calls to avoid rate limits
                await asyncio.sleep(0.5)
            
            # Send message
            message = "\n".join(predictions)
            timestamp = now.strftime("%Y-%m-%d %I:%M %p %Z")
            full_message = f"**Hourly Predictions - {timestamp}**\n```\n{message}\n```"
            
            if self.channel:
                await self.channel.send(full_message)
                logger.info("Predictions sent successfully")
            
        except Exception as e:
            logger.error(f"Error in hourly predictions: {e}")
    
    @hourly_predictions.before_loop
    async def before_hourly_predictions(self):
        """Wait until the next hour before starting"""
        now = datetime.now(TZ)
        next_hour = now.replace(minute=0, second=5, microsecond=0) + timedelta(hours=1)
        wait_seconds = (next_hour - now).total_seconds()
        
        logger.info(f"Waiting {wait_seconds:.0f} seconds until next hour")
        await asyncio.sleep(wait_seconds)

async def main():
    bot = DiscordBot()
    await bot.start(DISCORD_BOT_TOKEN)

if __name__ == "__main__":
    asyncio.run(main())
