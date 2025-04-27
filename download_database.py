import os
import subprocess
import sys
import shlex

# --- Configuration ---
DOWNLOAD_DIR = os.path.dirname(os.path.realpath(__file__))
URL = "https://huggingface.co/jiezong/AbCFold-Database/resolve/main/database.tar.gz?download=true"
MIRROR_URL = "https://hf-mirror.com/jiezong/AbCFold-Database/resolve/main/database.tar.gz?download=true"

TAR_FILENAME = "database.tar.gz" # 从URL中提取或硬编码文件名
tar_file_path = os.path.join(DOWNLOAD_DIR, TAR_FILENAME)
db_dir_path = os.path.join(DOWNLOAD_DIR, "database")

# --- Helper Function for Running Commands ---
def run_command(command_list, check=True, cwd=None, capture_output=True):
    """Helper function to run a shell command using subprocess."""
    print(f"Running command: {' '.join(shlex.quote(str(arg)) for arg in command_list)}")
    try:
        # 使用 capture_output=True 来捕获 stdout 和 stderr
        result = subprocess.run(
            command_list,
            check=check,         
            cwd=cwd,             
            text=True,           
            capture_output=capture_output 
        )
        if capture_output:
            print("Command STDOUT:\n", result.stdout)
            print("Command STDERR:\n", result.stderr)
        return True # Success
    except FileNotFoundError:
        print(f"Error: Command not found - '{command_list[0]}'. Please ensure it's installed and in PATH.")
        return False # Failure
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}.")
        if hasattr(e, 'stdout') and e.stdout:
            print("Command STDOUT:\n", e.stdout)
        if hasattr(e, 'stderr') and e.stderr:
            print("Command STDERR:\n", e.stderr)
        return False # Failure
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False # Failure

# --- 1. Download the database with fallback ---
print(f"--- Attempting to download database to {DOWNLOAD_DIR} ---")
download_successful = False

# Try primary URL
print(f"\nTrying primary URL: {URL}")
# wget -c (continue) -P (prefix directory) URL
# 使用 -O (output document) 可能更精确地控制输出文件名
# download_cmd1 = ["wget", "-c", "-O", tar_file_path, URL]
download_cmd1 = ["wget", "-c", "-O", os.path.join(DOWNLOAD_DIR, "database.tar.gz"), URL]
if run_command(download_cmd1, check=False, capture_output=True):
    if os.path.exists(tar_file_path) and os.path.getsize(tar_file_path) > 0:
        print("Successfully downloaded from primary URL.")
        download_successful = True
    else:
        print("Download command ran but target file seems invalid or empty. Trying mirror.")
        if os.path.exists(tar_file_path):
            try:
                os.remove(tar_file_path)
            except OSError as e:
                 print(f"Warning: could not remove potentially incomplete file {tar_file_path}: {e}")

if not download_successful:
    print(f"\nPrimary URL failed or file was invalid. Trying mirror URL: {MIRROR_URL}")
    # download_cmd2 = ["wget", "-c", "-O", tar_file_path, MIRROR_URL]
    download_cmd2 = ["wget", "-c", "-O", os.path.join(DOWNLOAD_DIR, "database.tar.gz"), MIRROR_URL]
    if run_command(download_cmd2, check=False, capture_output=True):
        if os.path.exists(tar_file_path) and os.path.getsize(tar_file_path) > 0:
            print("Successfully downloaded from mirror URL.")
            download_successful = True
        else:
            print("Mirror download command ran but target file seems invalid or empty.")
            if os.path.exists(tar_file_path):
                try:
                    os.remove(tar_file_path)
                except OSError as e:
                     print(f"Warning: could not remove potentially incomplete file {tar_file_path}: {e}")
    else:
        print("Mirror URL download also failed.")


# Exit if download failed completely
if not download_successful:
    print("\nError: Could not download the database from either URL. Exiting.")
    sys.exit(1)

# --- 2. Untar the downloaded file ---
print(f"\n--- Untarring {tar_file_path} into {DOWNLOAD_DIR} ---")
# Check if tar file exists before trying to untar
if not os.path.exists(tar_file_path):
    print(f"Error: Cannot untar, file not found: {tar_file_path}. Exiting.")
    sys.exit(1)

untar_cmd = ["tar", "-zxvf", tar_file_path, "-C", DOWNLOAD_DIR]
if run_command(untar_cmd):
    print("Successfully untarred.")

    # --- 3. Remove the tar.gz file ---
    print(f"\n--- Removing archive file: {tar_file_path} ---")
    try:
        os.remove(tar_file_path)
        print("Archive file removed successfully.")
    except OSError as e:
        # Don't exit on failure to remove, just warn
        print(f"Warning: Could not remove archive file {tar_file_path}. Error: {e}")
else:
    print("Error during untarring. Archive file will not be removed. Exiting.")
    sys.exit(1) # Exit if untar fails

# # --- Optional. Download the alphafold params ---
# print("\n--- Downloading AlphaFold parameters ---")
# script_path = os.path.join(DOWNLOAD_DIR, "openfold/scripts/download_alphafold_params.sh")
# alphafold_params_target_dir = db_dir_path # Params usually go into the extracted folder

# # Check if the script exists
# if not os.path.isfile(script_path):
#     print(f"Error: AlphaFold params download script not found at {script_path}")
#     print("Please ensure the 'openfold' repository structure is correct relative to this script.")
#     sys.exit(1)

# # Check if the target directory (created by untar) exists
# if not os.path.isdir(alphafold_params_target_dir):
#      print(f"Error: Target directory for AlphaFold params not found: {alphafold_params_target_dir}")
#      print("This directory should have been created when untarring the database.")
#      sys.exit(1)

# # Make sure the script is executable (optional, bash might handle it)
# # try:
# #     os.chmod(script_path, os.stat(script_path).st_mode | 0o111) # Add execute permission
# # except OSError as e:
# #     print(f"Warning: Could not set execute permission on {script_path}. Error: {e}")

# alphafold_params_cmd = ["bash", script_path, alphafold_params_target_dir]
# if run_command(alphafold_params_cmd, capture_output=True): # Capture output as these scripts can be verbose
#     print("Successfully executed AlphaFold parameters download script.")
# else:
#     print("Error occurred while executing the AlphaFold parameters download script. Exiting.")
#     sys.exit(1)

# print("\n--- Script finished successfully ---")