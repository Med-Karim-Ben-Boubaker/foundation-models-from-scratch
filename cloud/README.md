# Google Drive API Integration

Python module for Google Drive operations with path-based file management.

## Setup

1. Get Google Drive API credentials from [Google Cloud Console](https://console.developers.google.com/)
2. Download `client_secret.json` to project root
3. Install dependencies: `uv sync`

## Usage

```python
from cloud.drive_manager import DriveManager

# Interactive mode (opens browser)
drive = DriveManager()

# Headless mode (console-based)
drive = DriveManager(headless=True)

# Upload file
file_id = drive.upload_file("local.txt", "folder/file.txt")

# Download file
drive.download_file("folder/file.txt", "downloaded.txt")

# List files
files = drive.list_files(10, "folder")
```

## Run Demo

```bash
uv run python cloud/simple_demo.py
```

