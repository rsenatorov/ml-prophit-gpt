"""
GPT-2 Time Series Predictor with TradingView-style Candlestick Visualization
Auto-regressive token prediction with real-time comparison to actual prices
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pytz
import logging
import torch
import torch.nn as nn
import requests
from typing import Dict, List, Tuple, Optional
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_manager import data_manager, COIN_PAIRS
from model import GPT2TimeSeries
from config import build_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingViewPredictor:
    """GPT-2 Time Series Predictor with TradingView-style visualization"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GPT-2 Time Series - TradingView Style Predictor")
        self.root.geometry("1600x900")
        
        # TradingView color scheme
        self.colors = {
            'bg': '#131722',           # Dark background
            'chart_bg': '#1e222d',      # Chart background
            'grid': '#363c4e',          # Grid lines
            'text': '#d1d4dc',          # Text color
            'bullish': '#26a69a',       # Green candles
            'bearish': '#ef5350',       # Red candles
            'actual_line': '#2962ff',   # Blue line for actual price
            'predicted_line': '#ff9800', # Orange for predictions
            'volume': '#26a69a33'       # Volume bars (transparent)
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # Model paths
        self.tokenizer_dir = Path("tokenizer")
        self.checkpoint_path = Path("runs/gpt2ts/checkpoints/checkpoint_step_495000.pt")
        
        # Model and data
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = None
        self.kmeans_centers = None
        self.config = None
        
        # Data storage
        self.price_data = None
        self.norm_data = None
        self.tokens_data = None
        
        # Parameters
        self.context_length = 512
        self.current_symbol = None
        self.tz = pytz.timezone("America/Los_Angeles")
        
        # Auto-run state
        self.auto_running = False
        self.auto_position = 0
        self.predicted_tokens = []
        self.predicted_prices = []
        
        # Load model
        self.load_model_and_tokenizer()
        
        # Setup UI
        self.setup_ui()
        
        # Load initial data
        self.load_initial_data()
    
    def load_model_and_tokenizer(self):
        """Load the trained model and tokenizer"""
        try:
            logger.info("Loading model...")
            
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
            
            # Build config
            self.config = build_config()
            
            # Load vocabulary
            vocab_path = self.tokenizer_dir / "vocab.json"
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
            self.config.model.vocab_size = len(self.vocab)
            logger.info(f"Loaded vocabulary: {len(self.vocab)} tokens")
            
            # Load k-means centers
            centers_path = self.tokenizer_dir / "kmeans_centers.npy"
            self.kmeans_centers = np.load(centers_path).astype(np.float32)
            logger.info(f"Loaded k-means centers: {self.kmeans_centers.shape}")
            
            # Create and load model
            self.model = GPT2TimeSeries(self.config).to(self.device)
            self.model.load_state_dict(checkpoint['state']['model'])
            self.model.eval()
            
            # Set context length from config
            self.context_length = min(
                getattr(self.config.data, 'sequence_length', 512),
                getattr(self.config.model, 'max_seq_length', 512)
            )
            logger.info(f"Using context length: {self.context_length}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            sys.exit(1)
    
    def setup_ui(self):
        """Setup the user interface with TradingView styling"""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        # Symbol selection
        tk.Label(control_frame, text="Symbol:", bg=self.colors['bg'], 
                fg=self.colors['text'], font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        
        self.symbol_var = tk.StringVar()
        self.symbol_combo = ttk.Combobox(control_frame, textvariable=self.symbol_var, 
                                         width=15, state='readonly')
        self.symbol_combo['values'] = sorted(list(COIN_PAIRS))
        self.symbol_combo.pack(side=tk.LEFT, padx=5)
        self.symbol_combo.bind('<<ComboboxSelected>>', self.on_symbol_change)
        
        # Auto-run controls
        tk.Label(control_frame, text="  |  ", bg=self.colors['bg'], 
                fg=self.colors['text']).pack(side=tk.LEFT, padx=5)
        
        self.auto_btn = tk.Button(control_frame, text="▶ Start Auto-Run", 
                                 command=self.toggle_auto_run,
                                 bg='#26a69a', fg='white', font=('Arial', 10, 'bold'),
                                 padx=15, pady=5)
        self.auto_btn.pack(side=tk.LEFT, padx=5)
        
        self.speed_var = tk.DoubleVar(value=0.5)
        tk.Label(control_frame, text="Speed:", bg=self.colors['bg'], 
                fg=self.colors['text'], font=('Arial', 10)).pack(side=tk.LEFT, padx=(20, 5))
        tk.Scale(control_frame, from_=0.1, to=10.0, resolution=0.1,
                orient=tk.HORIZONTAL, variable=self.speed_var,
                bg=self.colors['bg'], fg=self.colors['text'],
                highlightthickness=0, length=150).pack(side=tk.LEFT)
        
        # Reset button
        tk.Button(control_frame, text="Reset", command=self.reset_prediction,
                 bg='#ef5350', fg='white', font=('Arial', 10),
                 padx=15, pady=5).pack(side=tk.LEFT, padx=20)
        
        # Status label
        self.status_label = tk.Label(control_frame, text="Ready", 
                                    bg=self.colors['bg'], fg=self.colors['text'],
                                    font=('Arial', 10))
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Chart frame
        chart_frame = tk.Frame(main_frame, bg=self.colors['chart_bg'])
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure with TradingView style
        plt.style.use('dark_background')
        self.figure = Figure(figsize=(14, 8), dpi=80, facecolor=self.colors['chart_bg'])
        self.ax = self.figure.add_subplot(111, facecolor=self.colors['chart_bg'])
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Info panel
        info_frame = tk.Frame(main_frame, bg=self.colors['bg'], height=100)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        info_frame.pack_propagate(False)
        
        self.info_text = tk.Text(info_frame, height=5, bg=self.colors['chart_bg'],
                                 fg=self.colors['text'], font=('Courier', 9),
                                 wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def fetch_binance_data(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch live data from Binance"""
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": "1h",
            "limit": min(max(limit, 10), 1000)
        }
        
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            raise RuntimeError(f"Binance API error: {resp.status_code}")
        
        data = resp.json()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(self.tz)
        
        return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    
    def assign_tokens(self, norm_ohlc: pd.DataFrame) -> np.ndarray:
        """Assign tokens using k-means centers"""
        df = norm_ohlc.copy()
        df["direction"] = df["close"] - df["open"]
        X = df[["open", "high", "low", "close", "direction"]].values.astype(np.float32)
        
        # Compute distances to all centers
        tokens = []
        for x in X:
            distances = np.sum((self.kmeans_centers - x) ** 2, axis=1)
            tokens.append(np.argmin(distances))
        
        return np.array(tokens, dtype=np.int32)
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare price, norm, and token data"""
        lookback = 100
        
        # Stochastic normalization
        norm_df = data_manager.stochastic_normalize_with_price_range(
            df[["date", "open", "high", "low", "close"]].copy(), 
            lookback=lookback
        )
        
        # Align data
        price_data = df.iloc[lookback-1:].reset_index(drop=True)
        norm_data = norm_df[["price_min", "price_max"]].reset_index(drop=True)
        norm_ohlc = norm_df[["open", "high", "low", "close"]].reset_index(drop=True)
        
        # Get tokens
        tokens = self.assign_tokens(norm_ohlc)
        tokens_data = pd.DataFrame({"token": tokens})
        
        # Ensure same length
        n = min(len(price_data), len(norm_data), len(tokens_data))
        return (
            price_data.iloc[:n].reset_index(drop=True),
            norm_data.iloc[:n].reset_index(drop=True),
            tokens_data.iloc[:n].reset_index(drop=True)
        )
    
    def load_initial_data(self):
        """Load initial data"""
        symbols = sorted(list(COIN_PAIRS))
        if symbols:
            self.symbol_var.set(symbols[0])
            self.load_symbol_data(symbols[0])
    
    def load_symbol_data(self, symbol: str):
        """Load data for a specific symbol"""
        try:
            self.status_label.config(text=f"Loading {symbol}...")
            self.root.update()
            
            # Fetch data
            raw_df = self.fetch_binance_data(symbol, limit=1000)
            
            # Prepare data
            self.price_data, self.norm_data, self.tokens_data = self.prepare_data(raw_df)
            
            self.current_symbol = symbol
            self.reset_prediction()
            
            self.status_label.config(text=f"Loaded {symbol}: {len(self.price_data)} candles")
            self.update_chart()
            
        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
            messagebox.showerror("Error", f"Failed to load {symbol}:\n{str(e)}")
    
    def token_to_ohlc(self, token_id: int) -> Dict[str, float]:
        """Convert token to normalized OHLC values"""
        token_str = str(token_id)
        if self.vocab and token_str in self.vocab:
            return {
                'open': self.vocab[token_str].get('open', 0.5),
                'high': self.vocab[token_str].get('high', 0.5),
                'low': self.vocab[token_str].get('low', 0.5),
                'close': self.vocab[token_str].get('close', 0.5)
            }
        return {'open': 0.5, 'high': 0.5, 'low': 0.5, 'close': 0.5}
    
    def denormalize_ohlc(self, norm_ohlc: Dict[str, float], price_min: float, price_max: float) -> Dict[str, float]:
        """Denormalize OHLC values"""
        price_range = max(price_max - price_min, 1e-8)
        return {
            'open': norm_ohlc['open'] * price_range + price_min,
            'high': norm_ohlc['high'] * price_range + price_min,
            'low': norm_ohlc['low'] * price_range + price_min,
            'close': norm_ohlc['close'] * price_range + price_min
        }
    
    def predict_next_token(self, context_tokens: List[int]) -> int:
        """Predict next token using greedy decoding"""
        # Ensure context is exactly context_length
        if len(context_tokens) > self.context_length:
            context_tokens = context_tokens[-self.context_length:]
        elif len(context_tokens) < self.context_length:
            # Pad if needed (shouldn't happen in normal operation)
            context_tokens = [0] * (self.context_length - len(context_tokens)) + context_tokens
        
        context_tensor = torch.tensor([context_tokens], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            logits = self.model(context_tensor)
            next_token = torch.argmax(logits[0, -1, :]).item()
        
        return next_token
    
    def draw_candlestick(self, ax, x, ohlc: Dict[str, float], width=0.6, is_predicted=False):
        """Draw a single candlestick"""
        open_price = ohlc['open']
        high_price = ohlc['high']
        low_price = ohlc['low']
        close_price = ohlc['close']
        
        # Determine color
        if close_price >= open_price:
            color = self.colors['bullish']
            body_color = self.colors['bullish']
        else:
            color = self.colors['bearish']
            body_color = self.colors['bearish']
        
        # Apply transparency for predicted candles
        if is_predicted:
            alpha = 0.7
        else:
            alpha = 1.0
        
        # Draw high-low line (wick)
        ax.plot([x, x], [low_price, high_price], color=color, linewidth=1, alpha=alpha)
        
        # Draw body
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        rect = Rectangle((x - width/2, body_bottom), width, body_height,
                        facecolor=body_color, edgecolor=color, 
                        linewidth=1, alpha=alpha)
        ax.add_patch(rect)
    
    def update_chart(self):
        """Update the chart with candlesticks and prediction"""
        self.ax.clear()
        
        if self.price_data is None or len(self.price_data) == 0:
            self.ax.text(0.5, 0.5, 'No data available', 
                        ha='center', va='center', transform=self.ax.transAxes,
                        color=self.colors['text'])
            self.canvas.draw()
            return
        
        # Determine what to show
        if self.auto_running and len(self.predicted_prices) > 0:
            # Show last 100 predicted candles during auto-run
            display_window = 100
            
            # Historical candles to show
            hist_end = self.auto_position
            hist_start = max(0, hist_end - (display_window - len(self.predicted_prices)))
            
            historical_data = self.price_data.iloc[hist_start:hist_end]
            
            # Prepare x-axis
            dates = []
            all_ohlc = []
            is_predicted = []
            
            # Add historical candles
            for idx, row in historical_data.iterrows():
                dates.append(row['date'])
                all_ohlc.append({
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close']
                })
                is_predicted.append(False)
            
            # Add predicted candles
            last_date = dates[-1] if dates else self.price_data.iloc[0]['date']
            for i, pred_ohlc in enumerate(self.predicted_prices[-display_window:]):
                pred_date = last_date + timedelta(hours=i+1)
                dates.append(pred_date)
                all_ohlc.append(pred_ohlc)
                is_predicted.append(True)
            
            # Draw candlesticks
            for i, (date, ohlc, is_pred) in enumerate(zip(dates, all_ohlc, is_predicted)):
                self.draw_candlestick(self.ax, mdates.date2num(date), ohlc, 
                                     width=0.6/24, is_predicted=is_pred)
            
            # Draw actual price line for comparison
            actual_end = min(self.auto_position + len(self.predicted_prices), len(self.price_data))
            actual_data = self.price_data.iloc[hist_start:actual_end]
            if len(actual_data) > 0:
                self.ax.plot(actual_data['date'], actual_data['close'],
                           color=self.colors['actual_line'], linewidth=2,
                           label='Actual Price', alpha=0.8)
            
            # Add divider line
            if hist_end < len(dates):
                divider_date = dates[len(historical_data)-1] if len(historical_data) > 0 else dates[0]
                self.ax.axvline(x=divider_date, color='white', linestyle='--', 
                              alpha=0.3, linewidth=1)
            
        else:
            # Normal view - show recent data
            display_window = 100
            start_idx = max(0, len(self.price_data) - display_window)
            display_data = self.price_data.iloc[start_idx:]
            
            # Draw candlesticks
            for idx, row in display_data.iterrows():
                ohlc = {
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close']
                }
                self.draw_candlestick(self.ax, mdates.date2num(row['date']), ohlc, width=0.6/24)
            
            dates = display_data['date'].tolist()
        
        # Format chart
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M', tz=self.tz))
        self.ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        self.figure.autofmt_xdate(rotation=45)
        
        # Styling
        self.ax.grid(True, alpha=0.2, color=self.colors['grid'], linestyle='-', linewidth=0.5)
        self.ax.set_xlabel('Date/Time (PDT)', color=self.colors['text'], fontsize=10)
        self.ax.set_ylabel('Price (USDT)', color=self.colors['text'], fontsize=10)
        self.ax.set_title(f'{self.current_symbol} - 1H Candlesticks with GPT-2 Predictions',
                         color=self.colors['text'], fontsize=12, fontweight='bold')
        
        # Set text colors
        self.ax.tick_params(colors=self.colors['text'])
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.colors['grid'])
        
        if self.auto_running:
            self.ax.legend(loc='upper left', framealpha=0.9, facecolor=self.colors['chart_bg'])
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Update info panel
        self.update_info()
    
    def update_info(self):
        """Update the information panel"""
        info = f"Symbol: {self.current_symbol}\n"
        info += f"Context Length: {self.context_length} tokens\n"
        
        if self.auto_running:
            info += f"Auto-Run Progress: {self.auto_position}/{len(self.tokens_data)}\n"
            info += f"Predicted Tokens: {len(self.predicted_tokens)}\n"
            
            if len(self.predicted_prices) > 0:
                last_pred = self.predicted_prices[-1]['close']
                actual_idx = min(self.auto_position + len(self.predicted_prices) - 1, 
                               len(self.price_data) - 1)
                if actual_idx >= 0:
                    last_actual = self.price_data.iloc[actual_idx]['close']
                    error = abs(last_pred - last_actual) / last_actual * 100
                    info += f"Last Prediction Error: {error:.2f}%"
        else:
            info += f"Total Candles: {len(self.price_data)}\n"
            info += "Click 'Start Auto-Run' to begin prediction"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
    
    def auto_run_step(self):
        """Perform one step of auto-run"""
        if not self.auto_running:
            return
        
        if self.auto_position >= len(self.tokens_data) - 1:
            self.auto_running = False
            self.auto_btn.config(text="▶ Start Auto-Run", bg='#26a69a')
            self.status_label.config(text="Auto-run complete")
            return
        
        # Get context
        context_start = max(0, self.auto_position - self.context_length + 1)
        context_end = self.auto_position + 1
        context_tokens = self.tokens_data.iloc[context_start:context_end]['token'].tolist()
        
        # Include previously predicted tokens in context if needed
        if len(self.predicted_tokens) > 0:
            # Replace actual tokens with predicted ones as we go
            overlap_start = max(0, len(context_tokens) - len(self.predicted_tokens))
            for i in range(overlap_start, len(context_tokens)):
                pred_idx = i - overlap_start
                if pred_idx < len(self.predicted_tokens):
                    context_tokens[i] = self.predicted_tokens[pred_idx]
        
        # Predict next token
        next_token = self.predict_next_token(context_tokens)
        self.predicted_tokens.append(next_token)
        
        # Convert to OHLC
        norm_ohlc = self.token_to_ohlc(next_token)
        
        # Use fixed denormalization range (from current position)
        norm_row = self.norm_data.iloc[self.auto_position]
        price_min = norm_row['price_min']
        price_max = norm_row['price_max']
        
        ohlc = self.denormalize_ohlc(norm_ohlc, price_min, price_max)
        self.predicted_prices.append(ohlc)
        
        # Move position
        self.auto_position += 1
        
        # Update display
        self.update_chart()
        
        # Schedule next step
        delay = int(1000 / self.speed_var.get())  # Convert speed to milliseconds
        self.root.after(delay, self.auto_run_step)
    
    def toggle_auto_run(self):
        """Toggle auto-run mode"""
        if self.auto_running:
            self.auto_running = False
            self.auto_btn.config(text="▶ Start Auto-Run", bg='#26a69a')
            self.status_label.config(text="Auto-run stopped")
        else:
            self.auto_running = True
            self.auto_btn.config(text="⏸ Pause", bg='#ef5350')
            self.status_label.config(text="Auto-run in progress...")
            
            # Start from beginning if needed
            if self.auto_position == 0:
                self.auto_position = self.context_length
            
            # Start auto-run
            self.auto_run_step()
    
    def reset_prediction(self):
        """Reset prediction state"""
        self.auto_running = False
        self.auto_position = self.context_length if self.tokens_data is not None else 0
        self.predicted_tokens = []
        self.predicted_prices = []
        self.auto_btn.config(text="▶ Start Auto-Run", bg='#26a69a')
        self.status_label.config(text="Reset")
        self.update_chart()
    
    def on_symbol_change(self, event=None):
        """Handle symbol change"""
        symbol = self.symbol_var.get()
        if symbol and symbol != self.current_symbol:
            self.load_symbol_data(symbol)
    
    def run(self):
        """Run the application"""
        logger.info("Starting TradingView-style Predictor")
        self.root.mainloop()

def main():
    """Main entry point"""
    app = TradingViewPredictor()
    app.run()

if __name__ == "__main__":
    main()