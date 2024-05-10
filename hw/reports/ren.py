import os

def rename_files_in_directory(root_dir):
    # Walk through each directory in the root directory
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # Get the full path of the file
            old_file_path = os.path.join(subdir, file)
            # Extract the directory name and use it as a prefix
            directory_name = os.path.basename(subdir)
            new_file_name = f"{directory_name}{file}"
            new_file_path = os.path.join(subdir, new_file_name)
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{old_file_path}' to '{new_file_path}'")

# Specify the path to the root directory of your project
root_directory = "C:/Users/arthu/TheGits/mprvs/hw/reports/"
rename_files_in_directory(root_directory)
