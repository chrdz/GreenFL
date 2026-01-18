#!/bin/bash

# Function to check subdirectories recursively
check_directory() {
    local DIR="$1"

    # Count the number of files in the current directory (ignoring directories)
    FILE_COUNT=$(find "$DIR" -maxdepth 1 -type f | wc -l)

    # If there is exactly one file, print a message
    if [ "$FILE_COUNT" -gt 1 ]; then
        # If there are at least two files, find the oldest file and delete it
        OLDEST_FILE=$(find "$DIR" -maxdepth 1 -type f -printf "%T@ %p\n" | sort -n | head -n 1 | cut -d ' ' -f 2-)
        
        # Delete the oldest file
        if [ -n "$OLDEST_FILE" ]; then
            echo "Deleting oldest file in $DIR (total files: $FILE_COUNT): $OLDEST_FILE"
            rm "$OLDEST_FILE"
        fi
    fi

    # Recursively check subdirectories
    for SUBDIR in "$DIR"/*; do
        if [ -d "$SUBDIR" ]; then
            check_directory "$SUBDIR"  # Call the function recursively
        fi
    done
}

# Check if the directory is provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

BASE_DIR="$1"

# Check if the provided directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist."
    exit 1
fi

# Start the recursive check from the base directory
check_directory "$BASE_DIR"

