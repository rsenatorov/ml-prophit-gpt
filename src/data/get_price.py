"""
Get Price Data - Fetch historical 5-minute candle data from Binance
Fixed API issues and improved error handling
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Optional, List, Dict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_manager import data_manager, COIN_PAIRS

# Configuration
CONFIG = {
    'test_mode': False,  # Set to True for testing with most recent month
    'test_days': 30,  # Days to fetch in test mode
    'max_retries': 5,
    'retry_delay': 1,  # seconds  
    'rate_limit_delay': 0.2,  # seconds between requests (5 requests per second max)
    'batch_save_interval': 10000,  # Save to disk every N candles
    'start_year': 2020,  # Start fetching from this year
    'connection_timeout': 10,  # seconds
    'read_timeout': 30,  # seconds
}

# Setup logging with colors for better visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BinancePriceFetcher:
    """Fetch historical price data from Binance public API with robust error handling"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.interval = "5m"
        self.limit = 1000  # Max allowed by Binance per request
        self.output_dir = Path("data/price")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path("data/temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup session with retry strategy
        self.session = self._create_session()
        
        # Track API stats
        self.api_calls = 0
        self.api_errors = 0
        self.start_time = datetime.now()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=CONFIG['max_retries'],
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default timeout
        session.timeout = (CONFIG['connection_timeout'], CONFIG['read_timeout'])
        
        return session
    
    def fetch_klines(self, symbol: str, start_time: int = None, end_time: int = None) -> Optional[List]:
        """
        Fetch kline/candlestick data from Binance with improved error handling
        """
        params = {
            'symbol': symbol,
            'interval': self.interval,
            'limit': self.limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        for attempt in range(CONFIG['max_retries']):
            try:
                response = self.session.get(
                    self.base_url, 
                    params=params,
                    timeout=(CONFIG['connection_timeout'], CONFIG['read_timeout'])
                )
                response.raise_for_status()
                
                self.api_calls += 1
                data = response.json()
                
                # Validate response
                if not isinstance(data, list):
                    raise ValueError(f"Invalid response format: expected list, got {type(data)}")
                
                return data
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1} for {symbol}")
                self.api_errors += 1
                
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error on attempt {attempt + 1} for {symbol}")
                self.api_errors += 1
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limited - wait longer
                    logger.warning(f"Rate limited for {symbol}, waiting 60 seconds...")
                    time.sleep(60)
                else:
                    logger.warning(f"HTTP error {e.response.status_code} on attempt {attempt + 1} for {symbol}")
                self.api_errors += 1
                
            except Exception as e:
                logger.warning(f"Unexpected error on attempt {attempt + 1} for {symbol}: {e}")
                self.api_errors += 1
            
            # Wait before retry
            if attempt < CONFIG['max_retries'] - 1:
                wait_time = CONFIG['retry_delay'] * (2 ** attempt)  # Exponential backoff
                time.sleep(wait_time)
        
        logger.error(f"Failed to fetch data for {symbol} after {CONFIG['max_retries']} attempts")
        return None
    
    def save_batch(self, df: pd.DataFrame, symbol: str, batch_num: int, is_final: bool = False):
        """
        Save batch of data to temporary file with progress tracking
        """
        if df.empty:
            return
        
        if is_final:
            # Final save to actual output directory
            output_path = self.output_dir / f"{symbol.lower()}.csv"
            df.to_csv(output_path, index=False)
            
            # Calculate hash for integrity
            data_hash = data_manager.calculate_data_hash(df)
            logger.info(f"  ✓ Final save: {len(df):,} candles | Hash: {data_hash}")
            
            # Clean up temp files
            for temp_file in self.temp_dir.glob(f"{symbol.lower()}_batch_*.csv"):
                temp_file.unlink()
        else:
            # Temporary batch save
            temp_path = self.temp_dir / f"{symbol.lower()}_batch_{batch_num}.csv"
            df.to_csv(temp_path, index=False)
            logger.debug(f"  Batch {batch_num} saved: {len(df)} candles")
    
    def load_existing_batches(self, symbol: str) -> pd.DataFrame:
        """
        Load existing batch files if any (for recovery)
        """
        batch_files = sorted(self.temp_dir.glob(f"{symbol.lower()}_batch_*.csv"))
        if batch_files:
            logger.info(f"  ℹ Found {len(batch_files)} existing batch files, resuming...")
            dfs = [pd.read_csv(f) for f in batch_files]
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    
    def find_earliest_trading_time(self, symbol: str) -> Optional[int]:
        """
        Find when a symbol started trading on Binance
        """
        try:
            # Binary search for earliest trading time
            # Start from 2017 (Binance launch year)
            start_year = 2017
            end_year = datetime.now().year
            
            earliest_found = None
            
            for year in range(start_year, end_year + 1):
                test_time = int(datetime(year, 1, 1).timestamp() * 1000)
                
                params = {
                    'symbol': symbol,
                    'interval': self.interval,
                    'startTime': test_time,
                    'limit': 1
                }
                
                response = self.session.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if data and len(data) > 0:
                    earliest_found = data[0][0]
                    break
                
                # Rate limiting
                time.sleep(0.1)
            
            if earliest_found:
                earliest_date = datetime.fromtimestamp(earliest_found / 1000)
                logger.info(f"  ℹ {symbol} started trading: {earliest_date.strftime('%Y-%m-%d')}")
            
            return earliest_found
            
        except Exception as e:
            logger.warning(f"  ⚠ Error finding earliest trading time: {e}")
            # Default to configured start year if we can't determine
            return int(datetime(CONFIG['start_year'], 1, 1).timestamp() * 1000)
    
    def fetch_all_historical_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch all available historical data for a symbol with improved progress tracking
        """
        # Determine time range
        if CONFIG['test_mode']:
            # Test mode: fetch only recent month
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = end_time - (CONFIG['test_days'] * 24 * 60 * 60 * 1000)
            logger.info(f"  📊 Test mode: Fetching last {CONFIG['test_days']} days")
        else:
            # Full mode: find earliest trading time
            earliest_time = self.find_earliest_trading_time(symbol)
            if earliest_time is None:
                logger.error(f"  ✗ Could not determine trading start time")
                return pd.DataFrame()
            
            earliest_year = datetime.fromtimestamp(earliest_time / 1000).year
            
            # Use the later of: configured start year or when symbol started trading
            if earliest_year > CONFIG['start_year']:
                start_time = earliest_time
            else:
                start_time = int(datetime(CONFIG['start_year'], 1, 1).timestamp() * 1000)
            
            end_time = int(datetime.now().timestamp() * 1000)
        
        # Check for existing batches (recovery mode)
        all_klines = []
        existing_df = self.load_existing_batches(symbol)
        if not existing_df.empty:
            all_klines = existing_df.to_dict('records')
            last_timestamp = data_manager.pst_string_to_timestamp(existing_df.iloc[-1]['date'])
            start_time = last_timestamp + 300000  # Continue from last saved point
            logger.info(f"  ↻ Resuming from: {existing_df.iloc[-1]['date']}")
        
        # Calculate expected requests
        time_range_ms = end_time - start_time
        expected_candles = time_range_ms / 300000  # 5 minutes in ms
        expected_requests = int(expected_candles / self.limit) + 1
        
        # Progress tracking
        batch_num = len(all_klines) // CONFIG['batch_save_interval']
        request_count = 0
        last_progress_update = datetime.now()
        
        # Create progress bar
        start_date = datetime.fromtimestamp(start_time / 1000)
        end_date = datetime.fromtimestamp(end_time / 1000)
        total_days = (end_date - start_date).days
        
        with tqdm(total=expected_requests, desc=f"  Fetching {symbol}", 
                 unit="req", ncols=100, 
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            while start_time < end_time:
                # Fetch batch
                klines = self.fetch_klines(symbol, start_time=start_time)
                
                if not klines:
                    logger.warning(f"  ⚠ No data returned, stopping fetch")
                    break
                
                # Process and add klines
                for kline in klines:
                    timestamp_ms = kline[0]
                    date_str = data_manager.timestamp_to_pst_string(timestamp_ms)
                    
                    all_klines.append({
                        'date': date_str,
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4])
                    })
                
                request_count += 1
                pbar.update(1)
                
                # Update progress info
                if klines:
                    current_date = datetime.fromtimestamp(klines[-1][0] / 1000)
                    pbar.set_postfix({
                        'Date': current_date.strftime('%Y-%m-%d'),
                        'Candles': f'{len(all_klines):,}'
                    })
                
                # Save batch if needed
                if len(all_klines) % CONFIG['batch_save_interval'] == 0:
                    batch_num += 1
                    df_batch = pd.DataFrame(all_klines[-CONFIG['batch_save_interval']:])
                    self.save_batch(df_batch, symbol, batch_num)
                
                # Check if we've reached the end
                if len(klines) < self.limit:
                    break
                
                # Update start_time for next batch
                last_timestamp = klines[-1][0]
                if last_timestamp >= end_time:
                    break
                
                start_time = last_timestamp + 300000  # Add 5 minutes (one candle)
                
                # Rate limiting
                time.sleep(CONFIG['rate_limit_delay'])
        
        if not all_klines:
            logger.warning(f"  ⚠ No data fetched for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines)
        
        # Remove duplicates if any
        original_len = len(df)
        df = df.drop_duplicates(subset=['date'], keep='first')
        if len(df) < original_len:
            logger.info(f"  ℹ Removed {original_len - len(df)} duplicate candles")
        
        # Sort by date
        df['timestamp'] = df['date'].apply(data_manager.pst_string_to_timestamp)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop('timestamp', axis=1)
        
        logger.info(f"  ✓ Fetched {len(df):,} candles in {request_count} requests")
        
        # Display date range
        if len(df) > 0:
            first_date = df.iloc[0]['date']
            last_date = df.iloc[-1]['date']
            first_year = first_date.split('-')[0]
            last_year = last_date.split('-')[0]
            logger.info(f"  📅 Date range: {first_year} to {last_year}")
        
        return df
    
    def apply_test_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply test mode to keep only most recent month of data
        """
        if CONFIG['test_mode'] and len(df) > 0:
            start_idx, end_idx = data_manager.get_test_mode_indices(len(df), CONFIG['test_days'])
            logger.info(f"  🧪 Test mode: keeping last {end_idx - start_idx:,} candles ({CONFIG['test_days']} days)")
            return df.iloc[start_idx:end_idx].reset_index(drop=True)
        return df
    
    def save_data(self, df: pd.DataFrame, symbol: str):
        """
        Save processed data to CSV with gap filling
        """
        if df.empty:
            logger.warning(f"  ⚠ No data to save for {symbol}")
            return
        
        logger.info(f"  Processing {len(df):,} candles...")
        
        # Fill gaps instead of removing data
        df_filled = data_manager.fill_missing_intervals(df)
        
        if len(df_filled) != len(df):
            diff = len(df_filled) - len(df)
            logger.info(f"  📊 After gap filling: {len(df_filled):,} candles ({diff:+,} added)")
        
        # Apply test mode if enabled
        df_final = self.apply_test_mode(df_filled)
        
        # Quick validation of candle integrity
        invalid_count = 0
        for _, row in df_final.iterrows():
            if not data_manager.validate_candle_integrity(row.to_dict()):
                invalid_count += 1
        
        if invalid_count > 0:
            percent_invalid = (invalid_count / len(df_final)) * 100
            if percent_invalid > 1:
                logger.warning(f"  ⚠ {invalid_count} invalid candles ({percent_invalid:.1f}%)")
            else:
                logger.info(f"  ✓ Data validation passed ({invalid_count} minor issues)")
        else:
            logger.info(f"  ✓ All candles valid")
        
        # Final save
        self.save_batch(df_final, symbol, 0, is_final=True)
    
    def print_progress_bar(self, current: int, total: int, symbol: str, width: int = 50):
        """
        Print a nice progress bar for overall progress
        """
        percent = current / total
        filled = int(width * percent)
        bar = '█' * filled + '░' * (width - filled)
        
        # Calculate stats
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.api_calls / elapsed if elapsed > 0 else 0
        
        print(f"\r[{current}/{total}] {bar} {percent*100:.1f}% | {symbol} | {rate:.1f} req/s", end='')
    
    def run(self):
        """
        Main execution function with improved progress display
        """
        # Clear console for better visibility
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n" + "="*80)
        print(" " * 25 + "BINANCE PRICE DATA FETCHER")
        print("="*80)
        print(f"  Mode: {'TEST (Recent Month)' if CONFIG['test_mode'] else 'FULL HISTORICAL'}")
        if CONFIG['test_mode']:
            print(f"  Test Period: Last {CONFIG['test_days']} days")
        else:
            print(f"  Historical Range: {CONFIG['start_year']} to present")
        print(f"  Interval: 5-minute candles")
        print(f"  Total Pairs: {len(COIN_PAIRS)}")
        print(f"  Rate Limit: {1/CONFIG['rate_limit_delay']:.1f} requests/second")
        print("="*80 + "\n")
        
        successful_pairs = []
        failed_pairs = []
        
        for idx, symbol in enumerate(COIN_PAIRS, 1):
            print(f"\n┌{'─'*76}┐")
            print(f"│ [{idx:2d}/{len(COIN_PAIRS)}] {symbol:<12} {' '*51}│")
            print(f"└{'─'*76}┘")
            
            try:
                # Fetch historical data
                df = self.fetch_all_historical_data(symbol)
                
                if df.empty:
                    failed_pairs.append(symbol)
                    print(f"  ✗ Failed to fetch data")
                else:
                    # Save processed data
                    self.save_data(df, symbol)
                    successful_pairs.append(symbol)
                    print(f"  ✓ Successfully processed {symbol}")
                
            except KeyboardInterrupt:
                print("\n\n⚠ Process interrupted by user")
                break
                
            except Exception as e:
                logger.error(f"  ✗ Failed to process {symbol}: {e}")
                failed_pairs.append(symbol)
                import traceback
                traceback.print_exc()
            
            # Show overall progress
            self.print_progress_bar(idx, len(COIN_PAIRS), symbol)
        
        print("\n")  # New line after progress bar
        
        # Summary
        self.print_summary(successful_pairs, failed_pairs)
    
    def print_summary(self, successful_pairs: List[str], failed_pairs: List[str]):
        """
        Print detailed summary statistics of collected data
        """
        print("\n" + "="*80)
        print(" " * 28 + "DATA COLLECTION SUMMARY")
        print("="*80)
        
        # Calculate totals
        total_candles = 0
        total_size = 0
        date_ranges = {}
        
        for symbol in successful_pairs:
            file_path = self.output_dir / f"{symbol.lower()}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                total_candles += len(df)
                total_size += file_path.stat().st_size
                
                if len(df) > 0:
                    first_date = df.iloc[0]['date']
                    last_date = df.iloc[-1]['date']
                    
                    # Extract years
                    first_year = first_date.split('-')[0]
                    last_year = last_date.split('-')[0]
                    
                    date_ranges[symbol] = (first_year, last_year, len(df))
        
        # Print results
        success_rate = (len(successful_pairs) / len(COIN_PAIRS)) * 100
        print(f"\n  {'Status':<20}: {len(successful_pairs)}/{len(COIN_PAIRS)} pairs ({success_rate:.1f}% success)")
        
        if failed_pairs:
            print(f"  {'Failed Pairs':<20}: {', '.join(failed_pairs)}")
        
        # API Statistics
        elapsed = (datetime.now() - self.start_time).total_seconds()
        avg_rate = self.api_calls / elapsed if elapsed > 0 else 0
        error_rate = (self.api_errors / self.api_calls) * 100 if self.api_calls > 0 else 0
        
        print(f"\n  {'API Calls':<20}: {self.api_calls:,}")
        print(f"  {'API Errors':<20}: {self.api_errors:,} ({error_rate:.1f}%)")
        print(f"  {'Avg Request Rate':<20}: {avg_rate:.2f} req/s")
        print(f"  {'Total Time':<20}: {timedelta(seconds=int(elapsed))}")
        
        # Data Statistics
        print(f"\n  {'Total Candles':<20}: {total_candles:,}")
        print(f"  {'Avg per Pair':<20}: {total_candles/max(len(successful_pairs), 1):,.0f}")
        print(f"  {'Total Size':<20}: {total_size / (1024*1024):.2f} MB")
        
        # Date ranges table
        if date_ranges:
            print("\n  Data Coverage by Symbol:")
            print("  " + "-"*60)
            print(f"  {'Symbol':<12} {'Years':<15} {'Candles':<15} {'Days':<10}")
            print("  " + "-"*60)
            
            for symbol, (start_year, end_year, count) in sorted(date_ranges.items()):
                days = count / 288  # 288 candles per day
                year_range = f"{start_year}-{end_year}"
                print(f"  {symbol:<12} {year_range:<15} {count:<15,} {days:<10.1f}")
        
        print("\n" + "="*80)
        print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")

def main():
    """Main entry point"""
    fetcher = BinancePriceFetcher()
    fetcher.run()

if __name__ == "__main__":
    main()