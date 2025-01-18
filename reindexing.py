import os
import subprocess

# Replace 'your_folder_path' with the path to your main folder
your_folder_path = "/home/evakrueger/Downloads/all_picks"


def reindex_ros_bags(folder_path):
    """
    Reindexes all ROS2 bag files in the given folder.

    :param folder_path: Path to the main folder containing bag file directories.
    """
    # Iterate over all subdirectories in the folder
    print(f"folder_path: {folder_path}")
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            bag_dir_path = os.path.join(root, dir_name)
            print(f"Reindexing ROS bag directory: {dir_name}")
            try:
                # Run the reindex command
                subprocess.run(["ros2", "bag", "reindex", bag_dir_path], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to reindex {bag_dir_path}: {e}")
            except Exception as e:
                print(f"Unexpected error for {bag_dir_path}: {e}")
        # No need to recurse further since we only care about the immediate subdirectories
        break


reindex_ros_bags(your_folder_path)