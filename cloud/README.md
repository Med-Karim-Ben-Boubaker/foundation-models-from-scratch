# Google Drive API Integration

Python module for Google Drive operations with path-based file management.

## Files

- `drive_manager.py` - Main DriveManager class
- `google_drive.py` - Authentication service
- `simple_demo.py` - Usage examples

## Setup

1. Get Google Drive API credentials from [Google Cloud Console](https://console.developers.google.com/)
2. Download `client_secret.json` to project root
3. Install dependencies: `uv sync`
4. First run opens browser for authentication

## Usage

```python
from cloud.drive_manager import DriveManager

drive = DriveManager()

# Upload to folder path (creates folders automatically)
file_id = drive.upload_file("local.txt", "folder/subfolder/file.txt")

# Download by path or file ID
drive.download_file("folder/subfolder/file.txt", "downloaded.txt")
drive.download_file(file_id, "downloaded.txt")

# List files
files = drive.list_files(10)  # root
files = drive.list_files(10, "folder")  # specific folder

# Find file by path
file_info = drive.find_file_by_path("folder/file.txt")
```

## API

### DriveManager Methods

- `upload_file(local_path: str, drive_path: str = None) -> str` - Upload file, returns file ID
- `download_file(identifier: str, local_path: str) -> bool` - Download by path or file ID
- `list_files(max_results: int = 10, folder_path: str = None) -> List[Dict]` - List files
- `find_file_by_path(path: str) -> Dict` - Find file by path

### Run Demo

```bash
uv run python cloud.simple_demo
```

