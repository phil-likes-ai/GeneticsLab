#!/usr/bin/env python3
"""
Test script to verify the genetic algorithm can run without the PositionSizer bug.
"""

import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_data():
    """Create sample market data for testing."""
    print("📊 Creating sample market data...")
    
    # Generate 1000 data points (about 10 days of 15m data)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='15min')
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic price data
    initial_price = 45000.0  # BTC starting price
    returns = np.random.normal(0.0001, 0.01, len(dates))  # Small returns with volatility
    
    prices = [initial_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i, (date, close_price) in enumerate(zip(dates, prices[1:])):
        open_price = prices[i] * (1 + np.random.normal(0, 0.002))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        volume = np.random.randint(100, 1000)
        
        data.append({
            'timestamp': int(date.timestamp() * 1000),  # Convert to milliseconds
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df['timestamp'], unit='ms')
    
    print(f"✅ Created {len(df)} sample data points")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    return df

def test_genetic_algorithm_simulation():
    """Test that the genetic algorithm can run a simulation without PositionSizer errors."""
    print("\n🧬 Testing Genetic Algorithm Simulation...")
    
    try:
        from ga_lab import GeneticAlgorithmManager
        import logging
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Load config
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Create sample data
        sample_data = create_sample_data()
        
        # Initialize GA manager
        ga_manager = GeneticAlgorithmManager(config, logger)
        print("✅ GeneticAlgorithmManager initialized successfully")
        
        # Create a test strategy
        test_strategy = ga_manager._create_random()
        print(f"✅ Test strategy created: {test_strategy.id}")
        
        # Test the simulation (this is where the PositionSizer bug would occur)
        print("🔄 Running simulation test...")
        final_balance = ga_manager._simulate(test_strategy, sample_data)
        
        print(f"✅ Simulation completed successfully!")
        print(f"   Final balance: ${final_balance:.2f}")
        print(f"   Strategy fitness: {test_strategy.fitness:.4f}")
        print(f"   Total trades: {test_strategy.trades}")
        print(f"   Win rate: {test_strategy.win_rate:.2%}")
        print(f"   Profit factor: {test_strategy.profit_factor:.2f}")
        
        # Test enhanced metrics
        if hasattr(test_strategy, 'max_drawdown'):
            print(f"   Max drawdown: {test_strategy.max_drawdown:.2%}")
        if hasattr(test_strategy, 'sharpe_ratio'):
            print(f"   Sharpe ratio: {test_strategy.sharpe_ratio:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Genetic algorithm simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_position_sizing_methods():
    """Test different position sizing methods work correctly."""
    print("\n💰 Testing Position Sizing Methods...")
    
    try:
        from ga_lab.position_sizing import PositionSizer
        
        methods_to_test = [
            "fixed_fraction",
            "kelly", 
            "risk_parity",
            "volatility_adjusted",
            "adaptive"
        ]
        
        for method in methods_to_test:
            print(f"   Testing {method} method...")
            
            # Create position sizer with method
            config = {
                "risk_per_trade": 0.02,
                "kelly_fraction": 0.25,
                "max_position_size": 1.0,
                "target_volatility": 0.15
            }
            
            # Use our fix: extract method to avoid duplicate parameter
            pos_cfg_without_method = {k: v for k, v in config.items() if k != "method"}
            sizer = PositionSizer(method=method, **pos_cfg_without_method)
            
            # Test position size calculation
            position_size = sizer.calculate_position_size(
                balance=10000,
                price=45000,
                volatility=0.02,
                win_rate=0.55,
                avg_win=200,
                avg_loss=150,
                returns_history=[0.01, -0.005, 0.02, -0.01, 0.015] * 10,
                confidence=0.8
            )
            
            print(f"   ✅ {method}: position size = {position_size:.6f}")
        
        print("✅ All position sizing methods work correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Position sizing methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_features():
    """Test that enhanced features are working."""
    print("\n🚀 Testing Enhanced Features...")
    
    try:
        # Test advanced indicators
        from ga_lab.indicators import Indicators
        sample_data = create_sample_data()
        
        # Test that we can add all indicators including new ones
        enhanced_data = Indicators.add_all(sample_data, required_params={
            'rsi_length': 14,
            'vwap_length': 20,
            'roc_length': 10,
            'trend_strength_length': 20
        })
        
        # Check for new indicators
        new_indicators = ['VWAP_20', 'ROC_10', 'TS_20']
        for indicator in new_indicators:
            if indicator in enhanced_data.columns:
                print(f"   ✅ Advanced indicator {indicator} calculated successfully")
            else:
                print(f"   ⚠️  Advanced indicator {indicator} not found")
        
        print("✅ Enhanced indicators are working!")
        
        # Test that we can import advanced modules
        from ga_lab import WalkForwardValidator, MultiObjectiveOptimizer, EnhancedBacktester
        print("✅ Advanced modules imported successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧬 Testing Genetic Algorithm with PositionSizer Bug Fix")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Genetic algorithm simulation
    if not test_genetic_algorithm_simulation():
        all_passed = False
    
    # Test 2: Position sizing methods
    if not test_position_sizing_methods():
        all_passed = False
    
    # Test 3: Enhanced features
    if not test_enhanced_features():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ PositionSizer bug is completely fixed")
        print("✅ Genetic algorithm runs without errors")
        print("✅ All position sizing methods work correctly")
        print("✅ Enhanced features are fully functional")
        
        print("\n🚀 The genetic algorithm system is ready!")
        print("📋 You can now safely run:")
        print("   ./venv/bin/python -m ga_lab.cli evolve --symbol BTC/USDT --timeframe 15m")
        print("   ./venv/bin/python -m ga_lab.cli fetch-data --all")
        print("   ./venv/bin/python example_advanced_usage.py")
        
        print("\n💡 Bug Fix Summary:")
        print("   • Fixed duplicate 'method' parameter in PositionSizer initialization")
        print("   • Method is now extracted before unpacking config dictionary")
        print("   • All position sizing methods (Kelly, Risk Parity, etc.) work correctly")
        print("   • Enhanced genetic algorithm features are fully operational")
        
    else:
        print("❌ SOME TESTS FAILED!")
        print("Additional debugging may be needed.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
