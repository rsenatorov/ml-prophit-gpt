"""
Get Normalized Data - Convert price data to stochastic normalized format
Now saves price_min and price_max for proper denormalization
Fixed memory issues and added better error recovery
"""

import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import gc  # For garbage collection
import traceback
import psutil  # For memory monitoring

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_manager import data_manager, COIN_PAIRS

# Configuration
CONFIG = {
    'lookback_period': 100,  # Current + 99 previous candles
    'test_mode': False,  # Set to True for testing with recent month
    'test_days': 30,  # Days to process in test mode
    'batch_save_interval': 5000,  # Save progress every N normalized candles (reduced)
    'validate_sample_size': 1000,  # Sample size for validation
    'chunk_size': 10000,  # Process data in smaller chunks to avoid memory issues (reduced)
    'memory_threshold': 85,  # Pause if memory usage exceeds this percentage
    'max_retries': 3,  # Max retries for a failed pair
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataNormalizer:
    """Normalize price data using stochastic normalization with improved progress tracking"""
    
    def __init__(self):
        self.input_dir = Path("data/price")
        self.output_dir = Path("data/norm")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path("data/temp_norm")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.lookback = CONFIG['lookback_period']
        
        # Track processing stats
        self.start_time = datetime.now()
        self.total_processed = 0
        self.total_errors = 0
        
        # Create a status file to track completed pairs
        self.status_file = self.output_dir / "normalization_status.txt"
        
        # Track memory usage
        self.process = psutil.Process()
    
    def check_memory(self):
        """Check current memory usage and pause if needed"""
        try:
            memory_percent = self.process.memory_percent()
            
            if memory_percent > CONFIG['memory_threshold']:
                logger.warning(f"  ⚠ High memory usage: {memory_percent:.1f}%, forcing garbage collection...")
                gc.collect()
                time.sleep(2)  # Give system time to recover
                
                # Check again
                memory_percent = self.process.memory_percent()
                if memory_percent > CONFIG['memory_threshold']:
                    logger.warning(f"  ⚠ Memory still high: {memory_percent:.1f}%, pausing...")
                    time.sleep(5)
            
            return memory_percent
        except:
            return 0
    
    def get_completed_pairs(self) -> set:
        """Get list of already completed pairs from status file"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    return set(line.strip() for line in f if line.strip())
            except Exception as e:
                logger.warning(f"Could not read status file: {e}")
                return set()
        return set()
    
    def mark_pair_complete(self, symbol: str):
        """Mark a pair as complete in the status file"""
        try:
            with open(self.status_file, 'a') as f:
                f.write(f"{symbol}\n")
                f.flush()  # Force write to disk
                os.fsync(f.fileno())  # Ensure it's written
        except Exception as e:
            logger.error(f"Could not update status file: {e}")
    
    def save_progress(self, df: pd.DataFrame, symbol: str, is_final: bool = False):
        """Save normalized data with progress tracking"""
        try:
            if is_final:
                output_path = self.output_dir / f"{symbol.lower()}.csv"
                df.to_csv(output_path, index=False)
                
                # Calculate hash for integrity
                data_hash = data_manager.calculate_data_hash(df)
                logger.info(f"  ✓ Final save: {len(df):,} normalized candles | Hash: {data_hash}")
                
                # Clean up temp file
                temp_path = self.temp_dir / f"{symbol.lower()}_progress.csv"
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass
                
                # Mark as complete
                self.mark_pair_complete(symbol)
            else:
                temp_path = self.temp_dir / f"{symbol.lower()}_progress.csv"
                df.to_csv(temp_path, index=False)
                logger.debug(f"  Progress saved: {len(df)} candles normalized")
        except Exception as e:
            logger.error(f"  ✗ Error saving data for {symbol}: {e}")
            raise
    
    def load_progress(self, symbol: str) -> pd.DataFrame:
        """Load saved progress if exists (for recovery)"""
        temp_path = self.temp_dir / f"{symbol.lower()}_progress.csv"
        if temp_path.exists():
            try:
                logger.info(f"  ↻ Found saved progress, resuming...")
                return pd.read_csv(temp_path)
            except Exception as e:
                logger.warning(f"  Could not load progress file: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def normalize_chunk(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> List[Dict]:
        """
        Normalize a chunk of data to avoid memory issues
        """
        normalized_data = []
        
        for i in range(start_idx, min(end_idx, len(df))):
            try:
                # Get window for current candle
                window_start = max(0, i - self.lookback + 1)
                window_end = i + 1
                window = df.iloc[window_start:window_end]
                
                if len(window) < self.lookback:
                    continue
                
                # Use only the last lookback candles if window is larger
                if len(window) > self.lookback:
                    window = window.iloc[-self.lookback:]
                
                # Get min and max from the window for denormalization
                high_max = float(window['high'].max())
                low_min = float(window['low'].min())
                price_range = max(high_max - low_min, 1e-8)
                
                # Normalize current candle
                current = df.iloc[i]
                normalized_data.append({
                    'date': current['date'],
                    'open': float((current['open'] - low_min) / price_range),
                    'high': float((current['high'] - low_min) / price_range),
                    'low': float((current['low'] - low_min) / price_range),
                    'close': float((current['close'] - low_min) / price_range),
                    'price_min': low_min,  # Save for denormalization
                    'price_max': high_max  # Save for denormalization
                })
                
            except Exception as e:
                logger.warning(f"  ⚠ Error normalizing candle at index {i}: {e}")
                continue
        
        return normalized_data
    
    def process_file(self, symbol: str, retry_count: int = 0) -> bool:
        """
        Process a single symbol's price data with detailed progress and memory management
        """
        input_path = self.input_dir / f"{symbol.lower()}.csv"
        
        if not input_path.exists():
            logger.warning(f"  ⚠ Price data not found for {symbol}")
            return False
        
        try:
            # Check memory before starting
            mem_percent = self.check_memory()
            logger.info(f"  💾 Memory usage: {mem_percent:.1f}%")
            
            # Load price data
            logger.info(f"  📊 Loading price data...")
            df = pd.read_csv(input_path)
            initial_count = len(df)
            logger.info(f"  📊 Loaded {initial_count:,} price candles")
            
            if len(df) < self.lookback:
                logger.warning(f"  ⚠ Insufficient data: {len(df)} candles (need ≥ {self.lookback})")
                return False
            
            # Apply test mode if enabled
            if CONFIG['test_mode']:
                start_idx, end_idx = data_manager.get_test_mode_indices(len(df), CONFIG['test_days'])
                # Ensure we have enough data for lookback even in test mode
                start_idx = max(0, start_idx - self.lookback + 1)
                df = df.iloc[start_idx:end_idx].reset_index(drop=True)
                logger.info(f"  🧪 Test mode: using last {len(df):,} candles ({CONFIG['test_days']} days)")
            
            # Check for existing progress
            existing_normalized = self.load_progress(symbol)
            if not existing_normalized.empty:
                start_idx = len(existing_normalized) + self.lookback - 1
                normalized_data = existing_normalized.to_dict('records')
                logger.info(f"  ↻ Resuming from candle {start_idx}")
            else:
                start_idx = self.lookback - 1
                normalized_data = []
            
            # Calculate total candles to process
            total_to_process = len(df) - start_idx
            
            if total_to_process > 0:
                # Extract date range for display
                first_date = df.iloc[start_idx]['date']
                last_date = df.iloc[-1]['date']
                first_year = first_date.split('-')[0]
                last_year = last_date.split('-')[0]
                
                logger.info(f"  📈 Normalizing {total_to_process:,} candles ({first_year}-{last_year})")
                
                # Process in chunks to avoid memory issues
                chunk_size = CONFIG['chunk_size']
                
                # Process with progress bar
                with tqdm(total=total_to_process, 
                         desc=f"  Normalizing {symbol}", 
                         unit="candles",
                         ncols=100,
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                    
                    # Process in chunks
                    for chunk_start in range(start_idx, len(df), chunk_size):
                        # Check memory periodically
                        if chunk_start % (chunk_size * 5) == 0:
                            self.check_memory()
                        
                        chunk_end = min(chunk_start + chunk_size, len(df))
                        
                        try:
                            # Normalize chunk
                            chunk_normalized = self.normalize_chunk(df, chunk_start, chunk_end)
                            normalized_data.extend(chunk_normalized)
                            
                            # Update progress
                            pbar.update(len(chunk_normalized))
                            
                            # Update year display
                            if chunk_normalized:
                                current_date = chunk_normalized[-1]['date']
                                current_year = current_date.split('-')[0]
                                pbar.set_postfix({'Year': current_year})
                            
                            # Save progress periodically
                            if len(normalized_data) % CONFIG['batch_save_interval'] == 0:
                                temp_df = pd.DataFrame(normalized_data)
                                self.save_progress(temp_df, symbol, is_final=False)
                                
                                # Force garbage collection to free memory
                                gc.collect()
                            
                        except Exception as e:
                            logger.error(f"  ✗ Error processing chunk {chunk_start}-{chunk_end}: {e}")
                            # Try to continue with next chunk
                            continue
                        
                        # Check if we should stop (safety check)
                        if chunk_end >= len(df):
                            break
            else:
                logger.info(f"  ℹ Already normalized, skipping...")
            
            # Create final DataFrame
            df_normalized = pd.DataFrame(normalized_data)
            
            if df_normalized.empty:
                logger.error(f"  ✗ Normalization failed - no data produced")
                return False
            
            # Validate normalized data
            logger.info(f"  🔍 Validating normalized data...")
            validation_result = self.validate_normalized_data(df_normalized)
            if not validation_result['valid']:
                logger.error(f"  ✗ Validation failed: {validation_result['message']}")
                return False
            
            logger.info(f"  ✓ Validation passed: {validation_result['message']}")
            
            # Final save
            self.save_progress(df_normalized, symbol, is_final=True)
            self.total_processed += len(df_normalized)
            
            # Clear memory
            del df
            del df_normalized
            gc.collect()
            
            return True
            
        except KeyboardInterrupt:
            logger.warning(f"  ⚠ Processing interrupted by user")
            raise
        except MemoryError:
            logger.error(f"  ✗ Out of memory processing {symbol}")
            gc.collect()
            
            if retry_count < CONFIG['max_retries']:
                logger.info(f"  ↻ Retrying with smaller chunk size...")
                CONFIG['chunk_size'] = max(1000, CONFIG['chunk_size'] // 2)
                return self.process_file(symbol, retry_count + 1)
            else:
                logger.error(f"  ✗ Failed after {CONFIG['max_retries']} retries")
                return False
        except Exception as e:
            logger.error(f"  ✗ Error processing {symbol}: {e}")
            self.total_errors += 1
            
            if retry_count < CONFIG['max_retries']:
                logger.info(f"  ↻ Retrying (attempt {retry_count + 2}/{CONFIG['max_retries']})...")
                time.sleep(2)  # Wait before retry
                return self.process_file(symbol, retry_count + 1)
            else:
                logger.error(f"  ✗ Failed after {CONFIG['max_retries']} retries")
                traceback.print_exc()
                return False
    
    def validate_normalized_data(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive validation of normalized data
        """
        validation_issues = []
        
        # Check for required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'price_min', 'price_max']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {'valid': False, 'message': f"Missing columns: {missing_cols}"}
        
        # Check for NaN values
        nan_counts = df[['open', 'high', 'low', 'close']].isna().sum()
        if nan_counts.any():
            return {'valid': False, 'message': f"NaN values found: {nan_counts.to_dict()}"}
        
        # Check bounds (should be between 0 and 1 for stochastic normalization)
        for col in ['open', 'high', 'low', 'close']:
            min_val = df[col].min()
            max_val = df[col].max()
            
            if min_val < -0.01 or max_val > 1.01:  # Small tolerance for floating point errors
                validation_issues.append(f"{col} outside [0,1]: [{min_val:.4f}, {max_val:.4f}]")
        
        # Sample validation for candle relationships
        sample_size = min(CONFIG['validate_sample_size'], len(df))
        sample_indices = np.random.choice(len(df), sample_size, replace=False)
        
        invalid_candles = 0
        for idx in sample_indices:
            if not self.validate_normalized_candle(df.iloc[idx].to_dict()):
                invalid_candles += 1
        
        if invalid_candles > 0:
            estimated_invalid = int(invalid_candles * len(df) / sample_size)
            percent_invalid = (estimated_invalid / len(df)) * 100
            if percent_invalid > 1:
                validation_issues.append(f"~{estimated_invalid} invalid candles ({percent_invalid:.1f}%)")
        
        if validation_issues:
            return {'valid': False, 'message': '; '.join(validation_issues)}
        
        # Calculate statistics for success message
        stats = {
            'mean_range': df[['open', 'high', 'low', 'close']].mean().mean(),
            'std_range': df[['open', 'high', 'low', 'close']].std().mean()
        }
        
        return {
            'valid': True, 
            'message': f"μ={stats['mean_range']:.3f}, σ={stats['std_range']:.3f}"
        }
    
    def validate_normalized_candle(self, candle: dict) -> bool:
        """
        Validate normalized candle maintains OHLC relationships
        """
        o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
        
        # High should be the highest (with small tolerance for floating point)
        if h < max(o, l, c) - 0.0001:
            return False
        
        # Low should be the lowest (with small tolerance for floating point)
        if l > min(o, h, c) + 0.0001:
            return False
        
        return True
    
    def clean_temp_files(self):
        """Clean up all temporary files"""
        try:
            for temp_file in self.temp_dir.glob("*.csv"):
                try:
                    temp_file.unlink()
                except:
                    pass
            logger.info("  ✓ Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"  Could not clean temp files: {e}")
    
    def run(self):
        """
        Main execution function with improved display and recovery
        """
        # Import time here to avoid issues
        import time
        
        # Clear console
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n" + "="*80)
        print(" " * 28 + "DATA NORMALIZER")
        print("="*80)
        print(f"  Method: Stochastic Normalization (with price range tracking)")
        print(f"  Lookback Window: {self.lookback} candles")
        print(f"  Mode: {'TEST (Recent Month)' if CONFIG['test_mode'] else 'FULL DATASET'}")
        if CONFIG['test_mode']:
            print(f"  Test Period: Last {CONFIG['test_days']} days")
        print(f"  Batch Save: Every {CONFIG['batch_save_interval']:,} candles")
        print(f"  Chunk Size: {CONFIG['chunk_size']:,} candles")
        print(f"  Memory Threshold: {CONFIG['memory_threshold']}%")
        print(f"  Total Pairs: {len(COIN_PAIRS)}")
        
        # Check for already completed pairs
        completed_pairs = self.get_completed_pairs()
        if completed_pairs:
            print(f"  Already Completed: {len(completed_pairs)} pairs")
            remaining = len(COIN_PAIRS) - len(completed_pairs)
            print(f"  Remaining: {remaining} pairs")
            if remaining > 0:
                print(f"  Resuming from: {len(completed_pairs) + 1}/{len(COIN_PAIRS)}")
        
        print("="*80 + "\n")
        
        successful_pairs = []
        failed_pairs = []
        skipped_pairs = []
        
        # Process each pair
        for idx, symbol in enumerate(COIN_PAIRS, 1):
            # Skip if already completed
            if symbol in completed_pairs:
                skipped_pairs.append(symbol)
                logger.info(f"[{idx:2d}/{len(COIN_PAIRS)}] {symbol}: Already completed, skipping...")
                continue
            
            print(f"\n┌{'─'*76}┐")
            print(f"│ [{idx:2d}/{len(COIN_PAIRS)}] {symbol:<12} {'Normalizing':<50}│")
            print(f"└{'─'*76}┘")
            
            try:
                # Clear memory before processing each symbol
                gc.collect()
                
                if self.process_file(symbol):
                    successful_pairs.append(symbol)
                    print(f"  ✓ Successfully normalized {symbol}")
                else:
                    failed_pairs.append(symbol)
                    print(f"  ✗ Failed to normalize {symbol}")
                    
                # Small delay between pairs to avoid overwhelming the system
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print("\n\n⚠ Process interrupted by user")
                print(f"Progress saved. You can resume from {symbol} by running the script again.")
                break
            except Exception as e:
                logger.error(f"  ✗ Unexpected error for {symbol}: {e}")
                failed_pairs.append(symbol)
                
                # Try to continue with next pair
                gc.collect()
                time.sleep(1)
        
        # Clean up temp files at the end
        self.clean_temp_files()
        
        # Print summary
        self.print_summary(successful_pairs, failed_pairs, skipped_pairs)
    
    def print_summary(self, successful_pairs: List[str], failed_pairs: List[str], skipped_pairs: List[str]):
        """
        Print detailed summary statistics of normalized data
        """
        print("\n" + "="*80)
        print(" " * 26 + "NORMALIZATION SUMMARY")
        print("="*80)
        
        # Include skipped pairs in the total success count
        total_successful = len(successful_pairs) + len(skipped_pairs)
        success_rate = (total_successful / len(COIN_PAIRS)) * 100
        
        print(f"\n  {'Status':<20}: {total_successful}/{len(COIN_PAIRS)} pairs ({success_rate:.1f}% complete)")
        
        if skipped_pairs:
            print(f"  {'Previously Done':<20}: {len(skipped_pairs)} pairs")
        
        if successful_pairs:
            print(f"  {'Newly Processed':<20}: {len(successful_pairs)} pairs")
        
        if failed_pairs:
            print(f"  {'Failed Pairs':<20}: {', '.join(failed_pairs)}")
        
        # Processing statistics
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.total_processed > 0:
            rate = self.total_processed / elapsed if elapsed > 0 else 0
            print(f"\n  {'Processing Time':<20}: {timedelta(seconds=int(elapsed))}")
            print(f"  {'Processing Rate':<20}: {rate:.0f} candles/sec")
        
        if self.total_errors > 0:
            print(f"  {'Total Errors':<20}: {self.total_errors}")
        
        # Memory statistics
        try:
            mem_percent = self.process.memory_percent()
            mem_info = self.process.memory_info()
            print(f"\n  {'Final Memory Usage':<20}: {mem_percent:.1f}%")
            print(f"  {'Memory (RSS)':<20}: {mem_info.rss / (1024*1024*1024):.2f} GB")
        except:
            pass
        
        # Data statistics for all completed pairs
        all_completed = successful_pairs + skipped_pairs
        if all_completed:
            total_candles = 0
            total_size = 0
            stats_summary = []
            
            for symbol in all_completed[:10]:  # Only check first 10 for speed
                file_path = self.output_dir / f"{symbol.lower()}.csv"
                if file_path.exists():
                    try:
                        # Count lines efficiently
                        with open(file_path, 'r') as f:
                            full_count = sum(1 for _ in f) - 1
                        
                        total_candles += full_count
                        total_size += file_path.stat().st_size
                        
                        # Read only first and last lines for date range
                        df_sample = pd.read_csv(file_path, nrows=1)
                        if len(df_sample) > 0:
                            first_date = df_sample.iloc[0]['date']
                            
                            # Read last line for end date
                            with open(file_path, 'rb') as f:
                                f.seek(-2, os.SEEK_END)
                                while f.read(1) != b'\n':
                                    f.seek(-2, os.SEEK_CUR)
                                last_line = f.readline().decode()
                            last_date = last_line.split(',')[0]
                            
                            first_year = first_date.split('-')[0]
                            last_year = last_date.split('-')[0]
                            
                            stats_summary.append({
                                'symbol': symbol,
                                'count': full_count,
                                'years': f"{first_year}-{last_year}"
                            })
                    except Exception as e:
                        logger.warning(f"Could not read stats for {symbol}: {e}")
                        continue
            
            if stats_summary:
                # Estimate total based on sample
                estimated_total_candles = total_candles * len(all_completed) / len(stats_summary)
                estimated_total_size = total_size * len(all_completed) / len(stats_summary)
                
                print(f"\n  {'Est. Total Candles':<20}: {estimated_total_candles:,.0f}")
                print(f"  {'Avg per Pair':<20}: {estimated_total_candles/max(len(all_completed), 1):,.0f}")
                print(f"  {'Reduction':<20}: {self.lookback - 1} candles/pair (lookback)")
                print(f"  {'Est. Total Size':<20}: {estimated_total_size / (1024*1024):.2f} MB")
                print(f"  {'New Columns':<20}: price_min, price_max (for denormalization)")
                
                # Sample statistics table
                if stats_summary:
                    print("\n  Sample Statistics (first 10 pairs):")
                    print("  " + "-"*50)
                    print(f"  {'Symbol':<12} {'Years':<12} {'Candles':<15}")
                    print("  " + "-"*50)
                    
                    for stats in stats_summary:
                        print(f"  {stats['symbol']:<12} {stats['years']:<12} {stats['count']:<15,}")
        
        print("\n" + "="*80)
        print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} PST")
        print("="*80 + "\n")

def main():
    """Main entry point"""
    # Import time at module level
    import time
    
    normalizer = DataNormalizer()
    normalizer.run()

if __name__ == "__main__":
    main()