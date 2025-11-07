"""
Main entry point for the Credit Line Adjuster system
"""

import streamlit as st
from interface import CreditLineInterface

def main():
    """
    Main function to launch the Credit Line Adjuster application.
    """
    app = CreditLineInterface()
    
    # Note: The actual Streamlit app runs through interface.py
    # This main.py provides the entry point for the modular system
    print("Credit Line Adjuster System")
    print("=" * 40)
    print("To run the application, use: streamlit run interface.py")
    print("\nSystem Modules:")
    print("- environment.py: Credit environment definition")
    print("- agent.py: RL agent implementation (TD-based)") 
    print("- training.py: Model training logic")
    print("- evaluation.py: Model performance evaluation")
    print("- interface.py: Streamlit user interface")
    print("- utils.py: Helper functions")
    print("- main.py: Entry point (this file)")

if __name__ == "__main__":
    main()