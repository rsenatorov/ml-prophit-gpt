"""
Data Visualizer - Interactive application for visualizing price, normalized, and tokenized data
Fixed to show same time periods across modes and allow conversion to price
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_manager import data_manager, COIN_PAIRS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CandlestickVisualizer:
    """Interactive candlestick chart visualizer with proper time alignment"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Crypto Candlestick Visualizer - Enhanced")
        self.root.geometry("1600x900")
        
        # Set theme colors
        self.colors = {
            'bg': '#1e1e1e',
            'fg': '#ffffff',
            'chart_bg': '#2d2d2d',
            'grid': '#404040',
            'bullish': '#26a69a',
            'bearish': '#ef5350',
            'neutral': '#ffa726'  # Orange for doji/neutral candles
        }
        
        # Configure root window
        self.root.configure(bg=self.colors['bg'])
        
        # Data directories
        self.price_dir = Path("data/price")
        self.norm_dir = Path("data/norm")
        self.tokens_dir = Path("data/tokens")
        self.tokenizer_dir = Path("tokenizer")
        
        # Load vocabulary if available
        self.vocab = self.load_vocabulary()
        
        # Current data storage
        self.price_data = None
        self.norm_data = None
        self.tokens_data = None
        self.current_display_data = None
        
        # Navigation state
        self.current_mode = "price"
        self.current_symbol = None
        self.current_start_idx = 0
        self.candles_to_display = 150
        self.show_as_price = False  # Toggle for showing normalized/tokens as price
        
        # Data alignment offset (normalized data starts 99 candles later)
        self.norm_offset = 99
        
        # Time alignment: track current position by date/timestamp
        self.current_time_position = None
        
        # Statistics cache
        self.stats_cache = {}
        
        # Setup UI
        self.setup_ui()
        
        # Load initial data
        self.load_initial_data()
        
        # Bind keyboard shortcuts
        self.setup_keyboard_shortcuts()
    
    def load_vocabulary(self):
        """Load token vocabulary if available"""
        vocab_path = self.tokenizer_dir / "vocab.json"
        if vocab_path.exists():
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
                logger.info(f"Loaded vocabulary with {len(vocab)} tokens")
                return vocab
        return None
    
    def setup_ui(self):
        """Setup the enhanced user interface"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel (left side)
        control_panel = ttk.LabelFrame(main_container, text="Controls", padding="10")
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Symbol selection
        ttk.Label(control_panel, text="Coin Pair:").pack(anchor=tk.W, pady=(0, 5))
        self.symbol_var = tk.StringVar()
        self.symbol_combo = ttk.Combobox(control_panel, textvariable=self.symbol_var, width=15)
        self.symbol_combo['values'] = self.get_available_symbols()
        self.symbol_combo.pack(fill=tk.X, pady=(0, 10))
        self.symbol_combo.bind('<<ComboboxSelected>>', self.on_symbol_change)
        
        # Mode selection
        ttk.Label(control_panel, text="Display Mode:").pack(anchor=tk.W, pady=(0, 5))
        self.mode_var = tk.StringVar(value="price")
        modes = [("Price Data", "price"), ("Normalized", "normalized"), ("Tokens", "tokens")]
        for text, value in modes:
            ttk.Radiobutton(control_panel, text=text, variable=self.mode_var, 
                          value=value, command=self.on_mode_change).pack(anchor=tk.W)
        
        # Separator
        ttk.Separator(control_panel, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Price conversion toggle
        self.price_convert_var = tk.BooleanVar(value=False)
        self.price_convert_check = ttk.Checkbutton(
            control_panel, 
            text="Show as Price",
            variable=self.price_convert_var,
            command=self.on_price_convert_toggle,
            state=tk.DISABLED
        )
        self.price_convert_check.pack(anchor=tk.W, pady=(0, 10))
        
        # Separator
        ttk.Separator(control_panel, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Navigation
        ttk.Label(control_panel, text="Navigation:").pack(anchor=tk.W, pady=(0, 5))
        
        # Candles per view
        candle_frame = ttk.Frame(control_panel)
        candle_frame.pack(fill=tk.X, pady=5)
        ttk.Label(candle_frame, text="Candles:").pack(side=tk.LEFT)
        self.candles_var = tk.IntVar(value=150)
        self.candles_spin = ttk.Spinbox(candle_frame, from_=10, to=500, increment=1, 
                                        textvariable=self.candles_var, width=10,
                                        command=self.on_candles_change)
        self.candles_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # Navigation buttons
        nav_frame = ttk.Frame(control_panel)
        nav_frame.pack(fill=tk.X, pady=10)
        ttk.Button(nav_frame, text="◀◀ -10", command=lambda: self.move_candles(-10)).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_frame, text="◀ -1", command=lambda: self.move_candles(-1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_frame, text="+1 ▶", command=lambda: self.move_candles(1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_frame, text="+10 ▶▶", command=lambda: self.move_candles(10)).pack(side=tk.LEFT, padx=1)
        
        nav_frame2 = ttk.Frame(control_panel)
        nav_frame2.pack(fill=tk.X, pady=5)
        ttk.Button(nav_frame2, text="Start", command=self.go_to_start).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame2, text="End", command=self.go_to_end).pack(side=tk.LEFT, padx=2)
        
        # Time slider
        ttk.Label(control_panel, text="Position:").pack(anchor=tk.W, pady=(10, 5))
        self.time_slider = ttk.Scale(control_panel, from_=0, to=100, orient=tk.HORIZONTAL)
        self.time_slider.pack(fill=tk.X, pady=(0, 5))
        self.time_slider.bind('<ButtonRelease-1>', self.on_time_change)
        
        self.position_label = ttk.Label(control_panel, text="")
        self.position_label.pack(anchor=tk.W)
        
        self.date_label = ttk.Label(control_panel, text="", font=('Courier', 9))
        self.date_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Separator
        ttk.Separator(control_panel, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # Statistics
        ttk.Label(control_panel, text="Statistics:").pack(anchor=tk.W, pady=(0, 5))
        self.stats_text = tk.Text(control_panel, height=10, width=35, wrap=tk.WORD, font=('Courier', 9))
        self.stats_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Export button
        ttk.Button(control_panel, text="Export View", command=self.export_chart).pack(fill=tk.X)
        
        # Chart container (right side)
        chart_container = ttk.LabelFrame(main_container, text="Chart", padding="5")
        chart_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure with dark theme
        plt.style.use('dark_background')
        self.figure = Figure(figsize=(12, 8), dpi=80, facecolor=self.colors['chart_bg'])
        self.ax = self.figure.add_subplot(111, facecolor=self.colors['chart_bg'])
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add matplotlib toolbar
        toolbar_frame = ttk.Frame(chart_container)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))
    
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for navigation"""
        self.root.bind('<Left>', lambda e: self.move_candles(-1))
        self.root.bind('<Right>', lambda e: self.move_candles(1))
        self.root.bind('<Shift-Left>', lambda e: self.move_candles(-10))
        self.root.bind('<Shift-Right>', lambda e: self.move_candles(10))
        self.root.bind('<Home>', lambda e: self.go_to_start())
        self.root.bind('<End>', lambda e: self.go_to_end())
        self.root.bind('<Control-e>', lambda e: self.export_chart())
        self.root.bind('<F5>', lambda e: self.refresh_data())
    
    def get_available_symbols(self):
        """Get list of available symbols with data"""
        symbols = []
        for symbol in COIN_PAIRS:
            price_file = self.price_dir / f"{symbol.lower()}.csv"
            if price_file.exists():
                symbols.append(symbol)
        return symbols
    
    def load_initial_data(self):
        """Load initial data for display"""
        symbols = self.get_available_symbols()
        if symbols:
            self.symbol_var.set(symbols[0])
            self.load_all_data_for_symbol(symbols[0])
    
    def load_all_data_for_symbol(self, symbol):
        """Load all data types for a symbol to maintain alignment"""
        self.current_symbol = symbol
        
        try:
            self.status_bar.config(text=f"Loading all data for {symbol}...")
            self.root.update()
            
            # Load price data
            price_path = self.price_dir / f"{symbol.lower()}.csv"
            if price_path.exists():
                self.price_data = pd.read_csv(price_path)
            else:
                self.price_data = None
            
            # Load normalized data
            norm_path = self.norm_dir / f"{symbol.lower()}.csv"
            if norm_path.exists():
                self.norm_data = pd.read_csv(norm_path)
            else:
                self.norm_data = None
            
            # Load token data
            tokens_path = self.tokens_dir / f"{symbol.lower()}.csv"
            if tokens_path.exists():
                self.tokens_data = pd.read_csv(tokens_path)
            else:
                self.tokens_data = None
            
            # Reset position to start
            self.current_start_idx = 0
            self.current_time_position = None
            
            # Update UI state
            self.update_display_data()
            self.update_ui_state()
            self.update_chart()
            self.update_statistics()
            
            self.status_bar.config(text=f"Loaded data for {symbol}")
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_bar.config(text="Error loading data")
    
    def find_matching_time_position(self, target_date):
        """Find the index in current display data that matches the target date"""
        if self.current_display_data is None or target_date is None:
            return 0
        
        # Look for exact match first
        matches = self.current_display_data[self.current_display_data['date'] == target_date]
        if not matches.empty:
            return matches.index[0]
        
        # If no exact match, find the closest date
        dates = pd.to_datetime(self.current_display_data['date'])
        target_dt = pd.to_datetime(target_date)
        
        # Find closest date
        time_diffs = abs(dates - target_dt)
        closest_idx = time_diffs.idxmin()
        
        return closest_idx
    
    def update_display_data(self):
        """Update the current display data based on mode"""
        mode = self.mode_var.get()
        
        if mode == "price":
            self.current_display_data = self.price_data
            self.price_convert_check.config(state=tk.DISABLED)
            self.show_as_price = False
        elif mode == "normalized":
            if self.show_as_price and self.norm_data is not None:
                # Convert normalized to price
                self.current_display_data = self.convert_normalized_to_price()
            else:
                self.current_display_data = self.norm_data
            self.price_convert_check.config(state=tk.NORMAL if self.norm_data is not None else tk.DISABLED)
        elif mode == "tokens":
            if self.tokens_data is not None and self.vocab is not None:
                if self.show_as_price and self.norm_data is not None:
                    # Convert tokens to price
                    self.current_display_data = self.convert_tokens_to_price()
                else:
                    # Convert tokens to normalized
                    self.current_display_data = self.convert_tokens_to_normalized()
            else:
                self.current_display_data = None
            self.price_convert_check.config(state=tk.NORMAL if self.tokens_data is not None else tk.DISABLED)
        
        # Maintain time position when switching modes
        if self.current_time_position is not None and self.current_display_data is not None:
            new_idx = self.find_matching_time_position(self.current_time_position)
            # Adjust index to show the same window position
            self.current_start_idx = max(0, min(new_idx, len(self.current_display_data) - self.candles_to_display))
    
    def convert_normalized_to_price(self):
        """Convert normalized data back to price using saved price_min/price_max"""
        if self.norm_data is None:
            return None
        
        converted_data = []
        for _, row in self.norm_data.iterrows():
            norm_values = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            }
            price_values = data_manager.denormalize_to_price(
                norm_values, 
                row['price_min'], 
                row['price_max']
            )
            converted_data.append({
                'date': row['date'],
                'open': price_values['open'],
                'high': price_values['high'],
                'low': price_values['low'],
                'close': price_values['close']
            })
        
        return pd.DataFrame(converted_data)
    
    def convert_tokens_to_normalized(self):
        """Convert token data to normalized OHLC using vocabulary"""
        if self.tokens_data is None or self.vocab is None:
            return None
        
        ohlc_data = []
        for _, row in self.tokens_data.iterrows():
            token_id = str(int(row['token']))
            if token_id in self.vocab:
                candle = self.vocab[token_id]
                ohlc_data.append({
                    'date': row['date'],
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close']
                })
            else:
                # Default values for unknown tokens
                ohlc_data.append({
                    'date': row['date'],
                    'open': 0.5,
                    'high': 0.5,
                    'low': 0.5,
                    'close': 0.5
                })
        
        return pd.DataFrame(ohlc_data)
    
    def convert_tokens_to_price(self):
        """Convert token data to price using vocabulary and norm data price ranges"""
        if self.tokens_data is None or self.vocab is None or self.norm_data is None:
            return None
        
        # First convert tokens to normalized
        norm_from_tokens = self.convert_tokens_to_normalized()
        if norm_from_tokens is None:
            return None
        
        # Then convert to price using the corresponding price ranges from norm_data
        converted_data = []
        for i, row in norm_from_tokens.iterrows():
            # Find corresponding norm data row by date
            norm_row = self.norm_data[self.norm_data['date'] == row['date']]
            
            if not norm_row.empty:
                norm_row = norm_row.iloc[0]
                norm_values = {
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close']
                }
                price_values = data_manager.denormalize_to_price(
                    norm_values,
                    norm_row['price_min'],
                    norm_row['price_max']
                )
                converted_data.append({
                    'date': row['date'],
                    'open': price_values['open'],
                    'high': price_values['high'],
                    'low': price_values['low'],
                    'close': price_values['close']
                })
            else:
                # Fallback if date not found
                converted_data.append({
                    'date': row['date'],
                    'open': row['open'] * 100,  # Rough approximation
                    'high': row['high'] * 100,
                    'low': row['low'] * 100,
                    'close': row['close'] * 100
                })
        
        return pd.DataFrame(converted_data)
    
    def get_aligned_index_range(self):
        """Get the valid index range considering data alignment"""
        if self.current_display_data is None:
            return 0, 0
        
        mode = self.mode_var.get()
        
        if mode == "price":
            # Price data: full range
            return 0, len(self.current_display_data)
        else:
            # Normalized/tokens: offset by 99 candles in price view
            return 0, len(self.current_display_data)
    
    def update_ui_state(self):
        """Update UI elements based on current data"""
        if self.current_display_data is None or len(self.current_display_data) == 0:
            return
        
        # Update slider range
        min_idx, max_idx = self.get_aligned_index_range()
        max_start = max(0, max_idx - self.candles_to_display)
        self.time_slider.configure(from_=min_idx, to=max_start)
        
        # Adjust current position if needed
        if self.current_start_idx > max_start:
            self.current_start_idx = max_start
        elif self.current_start_idx < min_idx:
            self.current_start_idx = min_idx
        
        self.time_slider.set(self.current_start_idx)
        self.update_position_label()
    
    def update_position_label(self):
        """Update position label with current view info"""
        if self.current_display_data is None:
            return
        
        mode = self.mode_var.get()
        total = len(self.current_display_data)
        
        # Calculate actual display indices
        start = self.current_start_idx + 1
        end = min(self.current_start_idx + self.candles_to_display, total)
        
        # Show position
        self.position_label.config(text=f"Candles {start}-{end} of {total}")
        
        # Show date range and store current time position
        if end > self.current_start_idx:
            start_date = self.current_display_data.iloc[self.current_start_idx]['date']
            end_date = self.current_display_data.iloc[min(end-1, total-1)]['date']
            
            # Store current time position for mode switching
            self.current_time_position = start_date
            
            # Extract just the date and time parts
            start_parts = start_date.split('-')
            end_parts = end_date.split('-')
            
            if len(start_parts) >= 4 and len(end_parts) >= 4:
                start_display = f"{start_parts[1]}/{start_parts[2]} {start_parts[3]}"
                end_display = f"{end_parts[1]}/{end_parts[2]} {end_parts[3]}"
                self.date_label.config(text=f"From: {start_display} PST\nTo:   {end_display} PST")
            else:
                self.date_label.config(text="")
    
    def calculate_statistics(self, data):
        """Calculate statistics for the current view"""
        if data is None or len(data) == 0:
            return {}
        
        # Get visible data
        end_idx = min(self.current_start_idx + self.candles_to_display, len(data))
        visible_data = data.iloc[self.current_start_idx:end_idx]
        
        if len(visible_data) == 0:
            return {}
        
        stats = {}
        
        # Basic stats
        stats['visible_candles'] = len(visible_data)
        
        # Price/value statistics
        ohlc_cols = ['open', 'high', 'low', 'close']
        for col in ohlc_cols:
            if col in visible_data.columns:
                stats[f'{col}_mean'] = visible_data[col].mean()
                stats[f'{col}_std'] = visible_data[col].std()
                stats[f'{col}_min'] = visible_data[col].min()
                stats[f'{col}_max'] = visible_data[col].max()
        
        # Bullish/bearish ratio
        bullish = (visible_data['close'] > visible_data['open']).sum()
        bearish = (visible_data['close'] < visible_data['open']).sum()
        neutral = (visible_data['close'] == visible_data['open']).sum()
        
        total = len(visible_data)
        stats['bullish_pct'] = (bullish / total) * 100 if total > 0 else 0
        stats['bearish_pct'] = (bearish / total) * 100 if total > 0 else 0
        stats['neutral_pct'] = (neutral / total) * 100 if total > 0 else 0
        
        return stats
    
    def update_statistics(self):
        """Update statistics display"""
        if self.current_display_data is None:
            return
        
        self.stats_text.delete(1.0, tk.END)
        
        stats = self.calculate_statistics(self.current_display_data)
        
        # Format and display statistics
        mode = self.mode_var.get()
        is_price = mode == "price" or self.show_as_price
        
        stats_str = f"Symbol: {self.current_symbol}\n"
        stats_str += f"Mode: {mode.capitalize()}"
        if self.show_as_price and mode != "price":
            stats_str += " (as Price)"
        stats_str += f"\n"
        stats_str += f"Visible: {stats.get('visible_candles', 0)} candles\n"
        stats_str += f"\n"
        
        if is_price:
            stats_str += f"Price Statistics (USDT):\n"
            stats_str += f"Mean:  ${stats.get('close_mean', 0):.2f}\n"
            stats_str += f"Std:   ${stats.get('close_std', 0):.2f}\n"
            stats_str += f"Min:   ${stats.get('low_min', 0):.2f}\n"
            stats_str += f"Max:   ${stats.get('high_max', 0):.2f}\n"
        else:
            stats_str += f"Normalized Statistics:\n"
            stats_str += f"Mean:  {stats.get('close_mean', 0):.4f}\n"
            stats_str += f"Std:   {stats.get('close_std', 0):.4f}\n"
            stats_str += f"Min:   {stats.get('low_min', 0):.4f}\n"
            stats_str += f"Max:   {stats.get('high_max', 0):.4f}\n"
        
        stats_str += f"\nCandle Distribution:\n"
        stats_str += f"Bullish: {stats.get('bullish_pct', 0):.1f}%\n"
        stats_str += f"Bearish: {stats.get('bearish_pct', 0):.1f}%\n"
        stats_str += f"Neutral: {stats.get('neutral_pct', 0):.1f}%"
        
        self.stats_text.insert(1.0, stats_str)
    
    def update_chart(self):
        """Update the candlestick chart"""
        if self.current_display_data is None or len(self.current_display_data) == 0:
            return
        
        # Clear previous chart
        self.ax.clear()
        
        # Get data to display
        end_idx = min(self.current_start_idx + self.candles_to_display, len(self.current_display_data))
        display_data = self.current_display_data.iloc[self.current_start_idx:end_idx]
        
        if len(display_data) == 0:
            return
        
        # Track min/max for y-axis scaling
        y_min = float('inf')
        y_max = float('-inf')
        
        # Draw candlesticks
        for i, (_, candle) in enumerate(display_data.iterrows()):
            o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
            
            # Skip invalid candles
            if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
                continue
            
            # Update min/max
            y_min = min(y_min, l)
            y_max = max(y_max, h)
            
            # Determine color
            if c > o:
                color = self.colors['bullish']
                edge_color = self.colors['bullish']
            elif c < o:
                color = self.colors['bearish']
                edge_color = self.colors['bearish']
            else:
                color = self.colors['neutral']  # Orange for doji candles
                edge_color = self.colors['neutral']
            
            # Draw high-low line (wick)
            self.ax.plot([i, i], [l, h], color=edge_color, linewidth=1, alpha=0.8)
            
            # Draw body
            body_height = abs(c - o)
            body_bottom = min(o, c)
            
            if body_height > 0:
                rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                               facecolor=color, edgecolor=edge_color, 
                               alpha=0.9, linewidth=0.5)
                self.ax.add_patch(rect)
            else:
                # Doji candle (neutral/orange)
                self.ax.plot([i - 0.3, i + 0.3], [o, o], color=edge_color, linewidth=2)
        
        # Set labels and title
        mode = self.mode_var.get()
        is_price = mode == "price" or self.show_as_price
        
        title = f"{self.current_symbol} - {mode.capitalize()} Mode"
        if self.show_as_price and mode != "price":
            title += " (Converted to Price)"
        
        self.ax.set_title(title, fontsize=14, fontweight='bold', color=self.colors['fg'])
        
        # Y-axis label
        if is_price:
            self.ax.set_ylabel("Price (USDT)", fontsize=12, color=self.colors['fg'])
        else:
            self.ax.set_ylabel("Normalized Value", fontsize=12, color=self.colors['fg'])
        
        # X-axis configuration
        self.ax.set_xlabel("Time (PST)", fontsize=12, color=self.colors['fg'])
        
        # Set x-axis labels (adaptive spacing)
        if self.candles_to_display <= 50:
            label_step = 5
        elif self.candles_to_display <= 150:
            label_step = 15
        else:
            label_step = 30
            
        label_indices = list(range(0, len(display_data), label_step))
        labels = []
        for idx in label_indices:
            date_str = display_data.iloc[idx]['date']
            # Simplify date display
            parts = date_str.split('-')
            if len(parts) >= 4:
                time_part = parts[3].replace('-PST', '')
                labels.append(f"{parts[1]}/{parts[2]}\n{time_part}")
            else:
                labels.append(date_str)
        
        self.ax.set_xticks(label_indices)
        self.ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        
        # Y-axis scaling with padding
        if y_min != float('inf') and y_max != float('-inf'):
            y_range = y_max - y_min
            y_padding = y_range * 0.1
            self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Grid
        self.ax.grid(True, alpha=0.2, color=self.colors['grid'], linestyle='--')
        
        # Style axes
        self.ax.spines['bottom'].set_color(self.colors['grid'])
        self.ax.spines['top'].set_color(self.colors['grid'])
        self.ax.spines['left'].set_color(self.colors['grid'])
        self.ax.spines['right'].set_color(self.colors['grid'])
        self.ax.tick_params(colors=self.colors['fg'])
        
        # Adjust layout
        self.figure.tight_layout()
        
        # Refresh canvas
        self.canvas.draw()
        
        # Update statistics
        self.update_statistics()
    
    def on_symbol_change(self, event=None):
        """Handle symbol selection change"""
        symbol = self.symbol_var.get()
        if symbol:
            self.load_all_data_for_symbol(symbol)
    
    def on_mode_change(self):
        """Handle display mode change"""
        if self.current_symbol:
            self.update_display_data()
            self.update_ui_state()
            self.update_chart()
    
    def on_price_convert_toggle(self):
        """Handle price conversion toggle"""
        self.show_as_price = self.price_convert_var.get()
        self.update_display_data()
        self.update_chart()
    
    def on_candles_change(self):
        """Handle candles per view change"""
        try:
            self.candles_to_display = self.candles_var.get()
            self.update_ui_state()
            self.update_chart()
        except:
            pass
    
    def on_time_change(self, event=None):
        """Handle time slider change"""
        self.current_start_idx = int(self.time_slider.get())
        self.update_position_label()
        self.update_chart()
    
    def move_candles(self, delta):
        """Move view by delta candles"""
        if self.current_display_data is None:
            return
        
        new_idx = self.current_start_idx + delta
        max_idx = max(0, len(self.current_display_data) - self.candles_to_display)
        new_idx = max(0, min(new_idx, max_idx))
        
        self.current_start_idx = new_idx
        self.time_slider.set(new_idx)
        self.update_position_label()
        self.update_chart()
    
    def go_to_start(self):
        """Go to the beginning of the data"""
        self.current_start_idx = 0
        self.time_slider.set(0)
        self.update_position_label()
        self.update_chart()
    
    def go_to_end(self):
        """Go to the end of the data"""
        if self.current_display_data is not None:
            max_idx = max(0, len(self.current_display_data) - self.candles_to_display)
            self.current_start_idx = max_idx
            self.time_slider.set(max_idx)
            self.update_position_label()
            self.update_chart()
    
    def refresh_data(self):
        """Refresh current data"""
        self.stats_cache.clear()
        if self.current_symbol:
            self.load_all_data_for_symbol(self.current_symbol)
            self.status_bar.config(text="Data refreshed")
    
    def export_chart(self):
        """Export current chart view"""
        if self.current_display_data is None:
            messagebox.showwarning("No Data", "No data to export")
            return
        
        # Ask for file location
        mode = self.mode_var.get()
        suffix = "_price" if self.show_as_price else ""
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
            initialfile=f"{self.current_symbol}_{mode}{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        if filename:
            try:
                self.figure.savefig(filename, dpi=150, bbox_inches='tight', facecolor=self.colors['chart_bg'])
                self.status_bar.config(text=f"Chart exported to {filename}")
                messagebox.showinfo("Export Successful", f"Chart saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Export Failed", f"Failed to export chart:\n{str(e)}")
    
    def run(self):
        """Run the application"""
        logger.info("Starting Enhanced Data Visualizer")
        self.root.mainloop()

def main():
    """Main entry point"""
    # Check if data exists
    data_dirs = [Path("data/price"), Path("data/norm"), Path("data/tokens")]
    data_exists = False
    
    for data_dir in data_dirs:
        if data_dir.exists() and any(data_dir.glob("*.csv")):
            data_exists = True
            break
    
    if not data_exists:
        messagebox.showerror("No Data", "No data found. Please run the data collection scripts first.")
        return
    
    # Create and run visualizer
    app = CandlestickVisualizer()
    app.run()

if __name__ == "__main__":
    main()