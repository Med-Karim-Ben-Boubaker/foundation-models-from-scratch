
from pathlib import Path
from cloud.drive_manager import DriveManager

from src.utils.logging import get_logger


logger = get_logger(__name__)


def main() -> None:
    """Simple demo of Google Drive path-based operations."""
    # Initialize DriveManager with headless authentication
    logger.info("Connecting to Google Drive...")
    drive = DriveManager(headless=True)  # Use headless=True for console-based auth
    logger.info("Connected successfully!")
    
    # Test file path (using existing project file)
    test_file = "data/the-verdict.txt"
    
    if not Path(test_file).exists():
        logger.info(f"Test file not found: {test_file}")
        logger.info("Please create a test file or modify the path")
        return
    
    try:
        # Example 1: Upload to a folder path
        logger.info(f"\n1. Uploading {test_file} to 'demo/files/' folder...")
        file_id = drive.upload_file(test_file, "demo/files/verdict.txt")
        logger.info(f"Upload complete! File ID: {file_id}")
        
        # Example 2: Upload to root with custom name
        logger.info(f"\n2. Uploading {test_file} to root with custom name...")
        file_id2 = drive.upload_file(test_file, "simple-demo-root.txt")
        logger.info(f"Upload complete! File ID: {file_id2}")
        
        # Example 3: Download using file path
        download_path = "artifacts/downloaded-by-path.txt"
        logger.info(f"\n3. Downloading file using path 'demo/files/verdict.txt' to {download_path}...")
        success = drive.download_file("demo/files/verdict.txt", download_path)
        
        if success:
            logger.info("Download by path successful!")
            print(f"File saved to: {download_path}")
        
        # Example 4: Download using file ID (old method still works)
        download_path2 = "artifacts/downloaded-by-id.txt"
        logger.info(f"\n4. Downloading file using ID to {download_path2}...")
        success2 = drive.download_file(file_id2, download_path2)
        
        if success2:
            logger.info("Download by ID successful!")
            logger.info(f"File saved to: {download_path2}")
        
        # Example 5: List files in root
        logger.info("\n5. Files in Google Drive root:")
        files = drive.list_files(3)
        for i, file in enumerate(files, 1):
            logger.info(f"  {i}. {file['name']}")
        
        # Example 6: List files in specific folder
        logger.info("\n6. Files in 'demo/files/' folder:")
        folder_files = drive.list_files(5, "demo/files")
        for i, file in enumerate(folder_files, 1):
            logger.info(f"  {i}. {file['name']}")
        
        # Example 7: Find a file by path
        logger.info("\n7. Finding file by path...")
        file_info = drive.find_file_by_path("demo/files/verdict.txt")
        if file_info:
            logger.info(f"Found: {file_info['name']} (ID: {file_info['id']})")
        else:
            logger.info("File not found")
    
    except Exception as e:
        logger.info(f"Error: {e}")


if __name__ == "__main__":
    main()
