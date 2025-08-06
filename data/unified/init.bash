#!/bin/bash

# Create parent directory for organization
mkdir -p ear_conditions

# Array of conditions from your consolidated list only
conditions=(
    "Normal_Tympanic_Membrane"
    "Acute_Otitis_Media"
    "Myringosclerosis"
    "Chronic_Otitis_Media"
    "Cerumen_Impaction"
    "Tympanostomy_Tubes"
    "Otitis_Externa"
    "Foreign_Object"
    "Pseudo_Membranes"
)

# Create directories and empty JSON files
for condition in "${conditions[@]}"; do
    # Create directory
    mkdir -p "ear_conditions/$condition"
    
    # Create empty JSON file
    touch "ear_conditions/$condition/data.json"
    
    echo "Created directory and empty JSON file for: $condition"
done

echo "All directories and JSON files have been created."
