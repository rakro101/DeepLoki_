#!/bin/bash

folder_path="$1"

# Check if the folder path exists
if [ ! -d "$folder_path" ]; then
  echo "Folder not found."
  exit 1
fi

# Function to convert images in a folder
convert_images_in_folder() {
  local subfolder="$1"

  # Iterate through the files in the subfolder
  for bmp_file in "$subfolder"/*.bmp; do
    if [ -f "$bmp_file" ]; then
      png_file="${bmp_file%.bmp}.png"

      # Convert BMP to PNG using ImageMagick's convert command
      convert "$bmp_file" "$png_file"

      # Delete the BMP image if needed
      rm "$bmp_file"

      echo "Converted $bmp_file to $png_file"
    fi
  done
}

# Recursively find and convert images in all subfolders
find "$folder_path" -type d | while read -r subfolder; do
  convert_images_in_folder "$subfolder"
done
