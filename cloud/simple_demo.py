from pathlib import Path
from cloud.drive_manager import DriveManager


def main() -> None:
    """Simple demo of Google Drive path-based operations."""
    # Initialize DriveManager
    print("Connecting to Google Drive...")
    drive = DriveManager()
    print("Connected successfully!")
    
    # Test file path (using existing project file)
    test_file = "data/the-verdict.txt"
    
    if not Path(test_file).exists():
        print(f"Test file not found: {test_file}")
        print("Please create a test file or modify the path")
        return
    
    try:
        # Example 1: Upload to a folder path
        print(f"\n1. Uploading {test_file} to 'demo/files/' folder...")
        file_id = drive.upload_file(test_file, "demo/files/verdict.txt")
        print(f"Upload complete! File ID: {file_id}")
        
        # Example 2: Upload to root with custom name
        print(f"\n2. Uploading {test_file} to root with custom name...")
        file_id2 = drive.upload_file(test_file, "simple-demo-root.txt")
        print(f"Upload complete! File ID: {file_id2}")
        
        # Example 3: Download using file path
        download_path = "artifacts/downloaded-by-path.txt"
        print(f"\n3. Downloading file using path 'demo/files/verdict.txt' to {download_path}...")
        success = drive.download_file("demo/files/verdict.txt", download_path)
        
        if success:
            print("Download by path successful!")
            print(f"File saved to: {download_path}")
        
        # Example 4: Download using file ID (old method still works)
        download_path2 = "artifacts/downloaded-by-id.txt"
        print(f"\n4. Downloading file using ID to {download_path2}...")
        success2 = drive.download_file(file_id2, download_path2)
        
        if success2:
            print("Download by ID successful!")
            print(f"File saved to: {download_path2}")
        
        # Example 5: List files in root
        print("\n5. Files in Google Drive root:")
        files = drive.list_files(3)
        for i, file in enumerate(files, 1):
            print(f"  {i}. {file['name']}")
        
        # Example 6: List files in specific folder
        print("\n6. Files in 'demo/files/' folder:")
        folder_files = drive.list_files(5, "demo/files")
        for i, file in enumerate(folder_files, 1):
            print(f"  {i}. {file['name']}")
        
        # Example 7: Find a file by path
        print("\n7. Finding file by path...")
        file_info = drive.find_file_by_path("demo/files/verdict.txt")
        if file_info:
            print(f"Found: {file_info['name']} (ID: {file_info['id']})")
        else:
            print("File not found")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
