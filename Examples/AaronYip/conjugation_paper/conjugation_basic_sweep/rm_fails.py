import os
import shutil

# Define the root directory to check
root_dir = 'test'

# Loop through all immediate subdirectories
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    
    # Only proceed if it's a directory
    if os.path.isdir(subdir_path):
        # Check if "step-00960.pickle" is in this subdirectory
        if "step-00960.pickle" not in os.listdir(subdir_path):
            # Delete the subdirectory if the file is not found
            print(f"Deleting {subdir_path}...")
            shutil.rmtree(subdir_path)
        else:
            print(f"{subdir_path} contains step-00960.pickle. Skipping deletion.")


