#!/bin/bash

# Define directories
dir_svs="./patches_512_1_4.svs"
dir_roi="./patches_512_1_4.svs_ROI"

# Initialize the JSON content
json_content="{\"labels\":["

# Iterate over files in the patches_512_1_4.svs directory
first_entry=true
for file_path in "$dir_svs"/*; do
    filename=$(basename "$file_path")
    
    # Extract the coordinates
    coordinates=$(echo "$filename" | grep -oP '(?<=_patch_)\d+_\d+')
    
    # Print coordinates for debugging
    # echo "Coordinates: $coordinates"

    # Check if any file in the ROI directory contains the same coordinates
    roi_file=$(find "$dir_roi" -type f -name "*_patch_1_$coordinates.jpg")

    if [[ -n "$roi_file" ]]; then
        label=1
    else
        label=0
    fi
    
    if $first_entry; then
        first_entry=false
    else
        json_content+=","
    fi

    json_content+="[\"$dir_svs/$filename\",$label]"
done

# Close the JSON content
json_content+="]}"

# Write to labels.json file
echo -e "$json_content" > labels.json

echo "labels.json file created successfully."
