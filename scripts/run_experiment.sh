#!/bin/bash

cd "$(dirname "$0")/.." || exit 1

echo "========================================="
echo "    Adversarial ML Lab - Experiment Runner"
echo "========================================="
echo ""
echo "What would you like to run?"
echo ""
echo "1) Data Poisoning (run_poisoning.py) - Train models with poisoned data"
echo "2) Vision Attacks (run_vision.py) - Test FGSM/PGD attacks on vision models"  
echo "3) Visualize Results (visualize_vision.py) - Generate adversarial example plots (requires vision attack to be run first!)"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Selected: Data Poisoning Experiments"
        read -p "Drag and drop your config JSON file here: " config_file
        config_file=$(echo "$config_file" | tr -d "'" | tr -d '"')
        
        if [ ! -f "$config_file" ]; then
            echo "Error: File not found: $config_file"
            exit 1
        fi
        
        echo "Validating config file..."
        python scripts/validate_config.py "$config_file" "poisoning"
        if [ $? -ne 0 ]; then
            echo "Config validation failed. Exiting."
            exit 1
        fi
        
        echo ""
        echo "Config is valid. Ready to run poisoning experiments."
        read -p "Do you want to proceed? (y/n): " confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            echo "Experiment cancelled."
            exit 0
        fi
        
        echo "Starting poisoning experiments..."
        python scripts/run_poisoning.py "$config_file"
        ;;
        
    2)
        echo ""
        echo "Selected: Vision Attack Experiments"
        read -p "Drag and drop your config JSON file here: " config_file
        config_file=$(echo "$config_file" | tr -d "'" | tr -d '"')
        
        if [ ! -f "$config_file" ]; then
            echo "Error: File not found: $config_file"
            exit 1
        fi
        
        echo "Validating config file..."
        python scripts/validate_config.py "$config_file" "vision"
        if [ $? -ne 0 ]; then
            echo "Config validation failed. Exiting."
            exit 1
        fi
        
        echo ""
        echo "Config is valid. Ready to run vision experiments."
        read -p "Do you want to proceed? (y/n): " confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            echo "Experiment cancelled."
            exit 0
        fi
        
        echo "Starting vision experiments..."
        python scripts/run_vision.py "$config_file"
        ;;
        
    3)
        echo ""
        echo "Selected: Visualization"
        read -p "Drag and drop your config JSON file here: " config_file
        config_file=$(echo "$config_file" | tr -d "'" | tr -d '"')
        
        if [ ! -f "$config_file" ]; then
            echo "Error: File not found: $config_file"
            exit 1
        fi
        
        echo "Validating config file..."
        python scripts/validate_config.py "$config_file" "visualize"
        if [ $? -ne 0 ]; then
            echo "Config validation failed. Exiting."
            exit 1
        fi
        
        echo ""
        echo "Config is valid. Ready to generate visualization."
        read -p "Do you want to proceed? (y/n): " confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            echo "Experiment cancelled."
            exit 0
        fi
        
        echo "Starting visualization..."
        python scripts/visualize_vision.py "$config_file"
        ;;
        
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Experiment completed!"