#!/usr/bin/env python
"""
Cryptocurrency Google Trends Analysis
-------------------------------------
This script runs a comprehensive analysis of Google Trends data and cryptocurrency prices,
exploring correlations, breakouts, and market phase predictions.
"""

import os
import sys

def main():
    """Run the complete analysis"""
    print("=== Crypto Google Trends Analysis ===")
    print("Running analysis... This may take a minute.\n")
    
    # Make sure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Add scripts directory to Python path
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
    sys.path.append(scripts_dir)
    
    # Run the main analysis
    from scripts.main import main as run_analysis
    run_analysis()
    
    print("\n=== Analysis Complete ===")
    print("Results have been saved to the 'outputs' directory.")

if __name__ == "__main__":
    main() 