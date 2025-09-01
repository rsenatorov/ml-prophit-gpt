"""
Data Manager - Core module for data transformation logic
This module contains all shared logic for data processing that will be used
both in training data collection and real-time inference.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import pytz
from typing import List, Dict, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import hashlib
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define 40 major cryptocurrency pairs (excluding stablecoins vs stablecoins)
COIN_PAIRS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT', 'DOTUSDT', 'POLUSDT',
    'LINKUSDT', 'LTCUSDT', 'ATOMUSDT', 'UNIUSDT', 'XLMUSDT',
    'ETCUSDT', 'FILUSDT', 'APTUSDT', 'ARBUSDT', 'OPUSDT',
    'NEARUSDT', 'VETUSDT', 'ALGOUSDT', 'GRTUSDT', 'SANDUSDT',
    'MANAUSDT', 'AAVEUSDT', 'AXSUSDT', 'EGLDUSDT', 'THETAUSDT',
    'RUNEUSDT', 'SUSDT', 'SUSHIUSDT', 'SNXUSDT', 'ENJUSDT',
    'CRVUSDT', 'CHZUSDT', 'GALAUSDT', 'CELOUSDT', 'FLOWUSDT'
]

@dataclass
class CandleData:
    """Structured candle data with validation"""
    date: str
    open: float
    high: float
    low: float
    close: float
    timestamp_ms: Optional[int] = None
    
    def __post_init__(self):
        """Validate candle data on creation"""
        if not self.is_valid():
            logger.warning(f"Invalid candle created: {self}")
    
    def is_valid(self) -> bool:
        """Check if candle maintains OHLC relationships"""
        return (self.high >= max(self.open, self.close, self.low) and 
                self.low <= min(self.open, self.close, self.high))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'date': self.date,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close
        }

class DataManager:
    """Core data management and transformation class"""
    
    def __init__(self):
        self.pst_tz = pytz.timezone('America/Los_Angeles')
        self.utc_tz = pytz.UTC
        self._cache = {}  # Cache for expensive operations
        
    def timestamp_to_pst_string(self, timestamp_ms: int) -> str:
        """
        Convert millisecond timestamp to PST string format
        Format: 2025-08-14-6:35PM-PST
        """
        dt_utc = datetime.fromtimestamp(timestamp_ms / 1000, tz=self.utc_tz)
        dt_pst = dt_utc.astimezone(self.pst_tz)
        
        # Format with AM/PM and timezone
        hour_12 = dt_pst.strftime('%I').lstrip('0')  # Remove leading zero
        minute = dt_pst.strftime('%M')
        am_pm = dt_pst.strftime('%p')
        
        date_str = f"{dt_pst.year:04d}-{dt_pst.month:02d}-{dt_pst.day:02d}-{hour_12}:{minute}{am_pm}-PST"
        return date_str
    
    def pst_string_to_timestamp(self, date_str: str) -> int:
        """
        Convert PST string format back to millisecond timestamp
        Optimized with caching for repeated conversions
        """
        # Use cache for repeated conversions
        if date_str in self._cache:
            return self._cache[date_str]
        
        # Remove timezone suffix if present
        if '-PST' in date_str:
            date_str = date_str.replace('-PST', '')
        
        # Parse the custom format
        parts = date_str.split('-')
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        time_part = parts[3]
        
        # Extract time components
        if 'PM' in time_part:
            time_str = time_part.replace('PM', '')
            is_pm = True
        else:
            time_str = time_part.replace('AM', '')
            is_pm = False
        
        hour, minute = map(int, time_str.split(':'))
        if is_pm and hour != 12:
            hour += 12
        elif not is_pm and hour == 12:
            hour = 0
            
        dt_pst = self.pst_tz.localize(datetime(year, month, day, hour, minute))
        dt_utc = dt_pst.astimezone(self.utc_tz)
        
        timestamp = int(dt_utc.timestamp() * 1000)
        
        # Cache the result
        if len(self._cache) < 10000:  # Limit cache size
            self._cache[date_str] = timestamp
        
        return timestamp
    
    def stochastic_normalize_with_price_range(self, df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
        """
        Perform stochastic normalization on OHLC data and save price range
        Uses current candle + previous 99 candles (100 total)
        Returns dataframe with normalized values AND price_min/price_max columns
        """
        if len(df) < lookback:
            raise ValueError(f"Insufficient data for normalization: {len(df)} < {lookback}")
        
        # Pre-allocate result arrays
        normalized_data = []
        
        # Use rolling window for efficient computation
        for i in range(lookback - 1, len(df)):
            window = df.iloc[i - lookback + 1:i + 1]
            
            # Vectorized min/max calculation
            high_max = window['high'].max()
            low_min = window['low'].min()
            price_range = max(high_max - low_min, 1e-8)  # Avoid division by zero
            
            # Normalize current candle
            current = df.iloc[i]
            normalized_data.append({
                'date': current['date'],
                'open': (current['open'] - low_min) / price_range,
                'high': (current['high'] - low_min) / price_range,
                'low': (current['low'] - low_min) / price_range,
                'close': (current['close'] - low_min) / price_range,
                'price_min': low_min,  # Save min price for denormalization
                'price_max': high_max   # Save max price for denormalization
            })
        
        return pd.DataFrame(normalized_data)
    
    def denormalize_to_price(self, norm_values: Dict, price_min: float, price_max: float) -> Dict:
        """
        Denormalize a candle back to price values using saved price range
        """
        price_range = max(price_max - price_min, 1e-8)
        
        denorm = {
            'open': norm_values['open'] * price_range + price_min,
            'high': norm_values['high'] * price_range + price_min,
            'low': norm_values['low'] * price_range + price_min,
            'close': norm_values['close'] * price_range + price_min
        }
        
        # Ensure OHLC relationships are maintained
        denorm['high'] = max(denorm['high'], denorm['open'], denorm['close'], denorm['low'])
        denorm['low'] = min(denorm['low'], denorm['open'], denorm['close'], denorm['high'])
        
        return denorm
    
    def token_to_normalized(self, token_id: int, vocab: Dict) -> Dict:
        """
        Convert token ID to normalized OHLC values using vocabulary
        """
        token_str = str(token_id)
        if token_str in vocab:
            candle = vocab[token_str]
            return {
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close']
            }
        else:
            # Default values for unknown tokens
            return {
                'open': 0.5,
                'high': 0.5,
                'low': 0.5,
                'close': 0.5
            }
    
    def token_to_price(self, token_id: int, vocab: Dict, price_min: float, price_max: float) -> Dict:
        """
        Convert token ID directly to price values
        """
        norm_values = self.token_to_normalized(token_id, vocab)
        return self.denormalize_to_price(norm_values, price_min, price_max)
    
    def validate_candle_integrity(self, candle: Union[Dict, CandleData]) -> bool:
        """
        Validate that candle data maintains proper OHLC relationships
        """
        if isinstance(candle, CandleData):
            return candle.is_valid()
        
        o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
        
        # High should be the highest (with small tolerance for floating point)
        if h < max(o, l, c) - 1e-8:
            return False
        
        # Low should be the lowest (with small tolerance for floating point)  
        if l > min(o, h, c) + 1e-8:
            return False
        
        return True
    
    def fill_missing_intervals(self, df: pd.DataFrame, interval_ms: int = 300000) -> pd.DataFrame:
        """
        Fill missing 5-minute intervals with interpolated data
        Optimized with vectorized operations
        """
        if len(df) < 2:
            return df
        
        logger.info("Checking for gaps and filling missing intervals...")
        
        # Vectorized timestamp conversion if not already present
        if 'timestamp' not in df.columns:
            df['timestamp'] = df['date'].apply(self.pst_string_to_timestamp)
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Find the complete range of timestamps
        start_ts = df['timestamp'].iloc[0]
        end_ts = df['timestamp'].iloc[-1]
        
        # Create complete timestamp range (every 5 minutes)
        complete_timestamps = np.arange(start_ts, end_ts + 1, interval_ms)
        
        # Create a complete dataframe with all timestamps
        complete_df = pd.DataFrame({'timestamp': complete_timestamps})
        complete_df['date'] = complete_df['timestamp'].apply(self.timestamp_to_pst_string)
        
        # Merge with existing data
        merged_df = complete_df.merge(df, on='timestamp', how='left', suffixes=('', '_old'))
        
        # Count gaps before filling
        gaps_mask = merged_df[['open', 'high', 'low', 'close']].isna().any(axis=1)
        gaps_count = gaps_mask.sum()
        
        if gaps_count > 0:
            gap_minutes = gaps_count * 5
            gap_hours = gap_minutes / 60
            logger.info(f"  Found {gaps_count} missing intervals ({gap_hours:.1f} hours of gaps)")
            
            # Smart interpolation based on gap size
            for col in ['open', 'high', 'low', 'close']:
                # For small gaps (< 1 hour), use linear interpolation
                # For larger gaps, use forward fill then backward fill
                gap_sizes = gaps_mask.astype(int).groupby((~gaps_mask).cumsum()).cumsum()
                small_gaps = gap_sizes <= 12  # 12 * 5min = 1 hour
                
                # Apply different strategies
                merged_df.loc[small_gaps, col] = merged_df.loc[small_gaps, col].interpolate(method='linear')
                merged_df[col] = merged_df[col].fillna(method='ffill').fillna(method='bfill')
            
            # Ensure OHLC relationships are maintained after interpolation
            merged_df['high'] = merged_df[['open', 'high', 'close']].max(axis=1)
            merged_df['low'] = merged_df[['open', 'low', 'close']].min(axis=1)
            
            # Use the date from complete_df
            merged_df['date'] = merged_df['date'].fillna(merged_df['date_old'])
            
            logger.info(f"  ✓ Filled {gaps_count} gaps using smart interpolation")
        else:
            logger.info("  ✓ No gaps found - data is complete")
        
        # Select only the columns we need
        result_df = merged_df[['date', 'open', 'high', 'low', 'close']].copy()
        
        # Final validation
        if result_df.isna().any().any():
            nan_count = result_df.isna().sum().sum()
            logger.warning(f"  ⚠ Warning: {nan_count} NaN values remain after interpolation")
            result_df = result_df.dropna()
        
        return result_df
    
    def detect_missing_intervals(self, df: pd.DataFrame, interval_ms: int = 300000) -> List[Tuple[int, int, str]]:
        """
        Optimized gap detection using vectorized operations
        Returns list of (index, gap_size, time_description) tuples
        """
        if len(df) < 2:
            return []
        
        # Vectorized timestamp conversion if not already present
        if 'timestamp' not in df.columns:
            timestamps = pd.Series([self.pst_string_to_timestamp(date) for date in df['date']])
        else:
            timestamps = df['timestamp']
        
        # Calculate differences
        time_diffs = timestamps.diff()
        
        # Find gaps (where diff > 5 minutes)
        gap_mask = time_diffs > interval_ms
        gap_indices = gap_mask[gap_mask].index.tolist()
        
        gaps = []
        for idx in gap_indices:
            gap_ms = time_diffs.iloc[idx]
            gap_size = int(gap_ms / interval_ms) - 1
            if gap_size > 0:
                # Create human-readable time description
                gap_minutes = gap_size * 5
                if gap_minutes < 60:
                    time_desc = f"{gap_minutes}min"
                elif gap_minutes < 1440:
                    time_desc = f"{gap_minutes/60:.1f}hrs"
                else:
                    time_desc = f"{gap_minutes/1440:.1f}days"
                
                gaps.append((idx, gap_size, time_desc))
        
        return gaps
    
    def calculate_candle_features(self, candle: Union[Dict, CandleData]) -> Dict:
        """
        Calculate additional candle features for better tokenization
        Enhanced with more technical indicators
        """
        if isinstance(candle, CandleData):
            candle = candle.to_dict()
        
        o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
        
        body_size = abs(c - o)
        full_range = max(h - l, 1e-8)
        
        # Basic features
        features = {
            'is_bullish': c > o,
            'body_ratio': body_size / full_range,
            'upper_wick_ratio': (h - max(o, c)) / full_range,
            'lower_wick_ratio': (min(o, c) - l) / full_range,
            'body_position': ((max(o, c) + min(o, c)) / 2 - l) / full_range,
        }
        
        # Pattern detection features
        features['is_doji'] = body_size / full_range < 0.1
        features['is_hammer'] = (features['lower_wick_ratio'] > 0.6 and 
                                 features['body_ratio'] < 0.3 and 
                                 features['upper_wick_ratio'] < 0.1)
        features['is_shooting_star'] = (features['upper_wick_ratio'] > 0.6 and 
                                        features['body_ratio'] < 0.3 and 
                                        features['lower_wick_ratio'] < 0.1)
        
        return features
    
    def get_test_mode_indices(self, total_length: int, test_days: int = 30) -> Tuple[int, int]:
        """
        Get indices for test mode (most recent month of data)
        Assumes 5-minute candles: 288 candles per day
        """
        candles_per_day = 288  # 24 hours * 60 minutes / 5 minutes
        test_size = min(candles_per_day * test_days, total_length)
        start_idx = max(0, total_length - test_size)
        return start_idx, total_length
    
    def calculate_data_hash(self, df: pd.DataFrame) -> str:
        """
        Calculate hash of dataframe for integrity checking
        """
        # Convert DataFrame to bytes
        df_bytes = pd.util.hash_pandas_object(df).values.tobytes()
        # Calculate SHA256 hash
        return hashlib.sha256(df_bytes).hexdigest()[:16]
    
    def get_progress_stats(self, processed: int, total: int, start_time: datetime) -> Dict:
        """
        Calculate processing progress statistics
        """
        if processed == 0:
            return {'percent': 0, 'eta': 'calculating...', 'rate': 0}
        
        percent = (processed / total) * 100
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = processed / elapsed if elapsed > 0 else 0
        
        if rate > 0:
            remaining = total - processed
            eta_seconds = remaining / rate
            eta = timedelta(seconds=int(eta_seconds))
            eta_str = str(eta).split('.')[0]  # Remove microseconds
        else:
            eta_str = 'calculating...'
        
        return {
            'percent': percent,
            'eta': eta_str,
            'rate': rate,
            'elapsed': timedelta(seconds=int(elapsed))
        }

# Create global instance
data_manager = DataManager()