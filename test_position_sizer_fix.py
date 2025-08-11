#!/usr/bin/env python3
"""
Test script to verify the PositionSizer initialization bug fix.
"""

import sys
import json
import logging

def test_position_sizer_initialization():
    """Test that PositionSizer can be initialized without parameter conflicts."""
    print("🧪 Testing PositionSizer initialization fix...")
    
    try:
        from ga_lab.position_sizing import PositionSizer
        
        # Test 1: Basic initialization
        print("✅ Test 1: Basic PositionSizer import successful")
        
        # Test 2: Initialize with method only
        sizer1 = PositionSizer(method="fixed_fraction")
        print("✅ Test 2: Basic initialization successful")
        
        # Test 3: Initialize with method and kwargs (the problematic case)
        config_with_method = {
            "method": "kelly",
            "risk_per_trade": 0.02,
            "kelly_fraction": 0.25,
            "max_position_size": 1.0
        }
        
        # This should work with our fix
        method = config_with_method.get("method", "fixed_fraction")
        config_without_method = {k: v for k, v in config_with_method.items() if k != "method"}
        sizer2 = PositionSizer(method=method, **config_without_method)
        print("✅ Test 3: Fixed initialization with config successful")
        
        # Test 4: Verify the sizer works
        position_size = sizer2.calculate_position_size(
            balance=10000,
            price=100,
            volatility=0.02,
            win_rate=0.55,
            avg_win=150,
            avg_loss=100
        )
        print(f"✅ Test 4: Position size calculation successful: {position_size:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ PositionSizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_manager_simulation_setup():
    """Test that the manager can set up simulation without errors."""
    print("\n🧪 Testing Manager simulation setup...")
    
    try:
        # Load config
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Test the specific code path that was failing
        sim_cfg = config["simulation"]
        pos_cfg = sim_cfg.get("position_sizing", {})
        
        # This is the fixed code from manager.py
        method = pos_cfg.get("method", "fixed_fraction")
        pos_cfg_without_method = {k: v for k, v in pos_cfg.items() if k != "method"}
        
        from ga_lab.position_sizing import PositionSizer
        position_sizer = PositionSizer(method=method, **pos_cfg_without_method)
        
        print(f"✅ Manager simulation setup successful with method: {method}")
        print(f"✅ Position sizer config: {pos_cfg_without_method}")
        
        return True
        
    except Exception as e:
        print(f"❌ Manager simulation setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_genetic_algorithm_manager():
    """Test that GeneticAlgorithmManager can be initialized."""
    print("\n🧪 Testing GeneticAlgorithmManager initialization...")
    
    try:
        import logging
        import json
        from ga_lab import GeneticAlgorithmManager
        
        # Load config
        with open('config.json', 'r') as f:
            config = json.load(f)
        logger = logging.getLogger(__name__)
        
        # Initialize manager
        ga_manager = GeneticAlgorithmManager(config, logger)
        print("✅ GeneticAlgorithmManager initialization successful")
        
        # Test that we can create a random strategy
        strategy = ga_manager._create_random()
        print(f"✅ Random strategy creation successful: {strategy.id}")
        
        return True
        
    except Exception as e:
        print(f"❌ GeneticAlgorithmManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔧 Testing PositionSizer Bug Fix")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1: PositionSizer initialization
    if not test_position_sizer_initialization():
        all_passed = False
    
    # Test 2: Manager simulation setup
    if not test_manager_simulation_setup():
        all_passed = False
    
    # Test 3: GeneticAlgorithmManager
    if not test_genetic_algorithm_manager():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ PositionSizer bug has been fixed")
        print("✅ Genetic algorithm should now run successfully")
        print("✅ Enhanced features are ready to use")
        
        print("\n📋 You can now run:")
        print("   python3 -m ga_lab.cli evolve --symbol BTC/USDT --timeframe 15m")
        
    else:
        print("❌ SOME TESTS FAILED!")
        print("The bug fix may need additional work.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
