#!/bin/bash

# Uncomment the following command line for Rivanna operation
module load gcc/11.4.0 openmpi/4.1.4 gdal/3.6.3


# Check if two directories are provided as arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

input_directory=$1
output_directory=$2

# Check if the output directory exists, if not, create it
if [ ! -d "$output_directory" ]; then
    mkdir -p "$output_directory"
fi

# st: Threshold of object size (in pixel)
# Iterate over all .tif files in the input directory
for file in "$input_directory"/*.tif; do
    if [ -e "$file" ]; then
        output_name="$output_directory"/siv_$(basename "$file" .tif).tif
        # Skip processing if the output file already exists
        if [ -e "$output_name" ]; then
            echo "Output file already exists for $file. Skipping processing."
        else
            gdal_sieve.py -st 3 -4 "$file" "$output_name"
        fi
    else
        echo "No .tif files found in the specified input directory."
        exit 1
    fi
done

echo "Processing complete!"

