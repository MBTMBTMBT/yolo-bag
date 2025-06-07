import os
import sys
import argparse


def process_txt_file(file_path):
    """
    Process a single txt file to change all label numbers to 0
    while keeping other data unchanged.

    Args:
        file_path (str): Path to the txt file to process
    """
    try:
        # Read the original content
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Process each line
        processed_lines = []
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                # Split the line by spaces
                parts = line.split()
                if len(parts) >= 5:  # Ensure we have at least 5 parts (label + 4 coordinates)
                    # Replace the first element (label) with '0' and keep the rest
                    parts[0] = '0'
                    processed_lines.append(' '.join(parts) + '\n')
                else:
                    # If line doesn't have expected format, keep it as is
                    processed_lines.append(line + '\n')
            else:
                processed_lines.append('\n')

        # Write back to the same file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(processed_lines)

        print(f"Successfully processed: {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")


def process_folder(folder_path):
    """
    Process all txt files in the specified folder.

    Args:
        folder_path (str): Path to the folder containing txt files
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory.")
        return

    # Find all txt files in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]

    if not txt_files:
        print(f"No txt files found in folder: {folder_path}")
        return

    print(f"Found {len(txt_files)} txt files to process...")

    # Process each txt file
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        process_txt_file(file_path)

    print(f"Processing completed! Total files processed: {len(txt_files)}")


def main():
    """
    Main function to handle command line arguments and execute the script.
    """
    parser = argparse.ArgumentParser(
        description="Convert all label numbers in txt files to 0 while keeping other data unchanged."
    )
    parser.add_argument(
        'folder_path',
        type=str,
        help='Path to the folder containing txt files to process'
    )

    args = parser.parse_args()

    print(f"Starting to process txt files in folder: {args.folder_path}")
    process_folder(args.folder_path)


if __name__ == "__main__":
    main()
