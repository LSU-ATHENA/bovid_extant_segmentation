import os
import sys

def filter_files(folder1, folder2):
    """
    Keeps files in both folders only if they exist in the other folder.
    Deletes files that don't have a matching name in the other folder.
    """
    folder2_compare = os.path.join(folder2, "bw") if os.path.isdir(os.path.join(folder2, "bw")) else folder2

    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2_compare))
    
    # Find files to delete
    delete_from_folder1 = files1 - files2
    delete_from_folder2 = files2 - files1
    
    # Delete files from folder1
    for file in delete_from_folder1:
        file_path = os.path.join(folder1, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
    
    # Delete files from folder2
    for file in delete_from_folder2:
        file_path = os.path.join(folder2_compare, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
    
    # Count remaining files
    remaining_folder1 = len([f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))])
    remaining_folder2 = len([f for f in os.listdir(folder2_compare) if os.path.isfile(os.path.join(folder2_compare, f))])
    
    print(f"\nFiles remaining in {folder1}: {remaining_folder1}")
    print(f"Files remaining in {folder2_compare}: {remaining_folder2}")

if __name__ == "__main__":
    subdirs = ["LM1", "LM2", "LM3", "UM1", "UM2", "UM3"]
    if len(sys.argv) != 3:
        print("Usage: python data_filter.py <folder1> <folder2>")
        sys.exit(1)
    
    folder1 = sys.argv[1]
    folder2 = sys.argv[2]
    
    if not os.path.isdir(folder1) or not os.path.isdir(folder2):
        print("Error: Both arguments must be valid directories")
        sys.exit(1)
    
    for subdir in subdirs:
        folder1_path = os.path.join(folder1, subdir)
        folder2_path = os.path.join(folder2, subdir)
        if os.path.isdir(folder1_path) and os.path.isdir(folder2_path):
            filter_files(folder1_path, folder2_path)
