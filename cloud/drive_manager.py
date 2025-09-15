"""
Google Drive file management module for uploading and downloading files.

This module provides a simple interface for Google Drive operations,
building upon the existing authentication system.
"""

from pathlib import Path
from typing import Optional, Any, Dict, List
import logging

from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError

from cloud.google_drive import Create_Service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Drive API configuration
CLIENT_SECRET_FILE_PATH = 'client_secret.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']


class DriveManager:
    """
    A simple manager for Google Drive file operations.
    
    Handles authentication, file uploads, and downloads using the Google Drive API.
    """
    
    def __init__(self) -> None:
        self.service = self._get_authenticated_service()
    
    def _get_authenticated_service(self) -> Any:
        """Get authenticated Google Drive service."""
        service = Create_Service(
            CLIENT_SECRET_FILE_PATH,
            API_NAME,
            API_VERSION,
            SCOPES
        )
        if not service:
            raise Exception("Failed to authenticate with Google Drive API")
        return service
    
    def _create_folder(self, folder_name: str, parent_id: Optional[str] = None) -> str:
        """Create a folder in Google Drive."""
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        if parent_id:
            file_metadata['parents'] = [parent_id]
        
        try:
            folder = self.service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            
            folder_id = folder.get('id')
            logger.info(f"Created folder '{folder_name}' with ID: {folder_id}")
            return folder_id
            
        except HttpError as error:
            logger.error(f"Failed to create folder '{folder_name}': {error}")
            raise
    
    def _find_folder_by_name(self, folder_name: str, parent_id: Optional[str] = None) -> Optional[str]:
        """Find a folder by name in the specified parent (or root if None)."""
        try:
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
            if parent_id:
                query += f" and '{parent_id}' in parents"
            else:
                query += " and 'root' in parents"
            
            results = self.service.files().list(
                q=query,
                fields="files(id,name)"
            ).execute()
            
            files = results.get('files', [])
            if files:
                return files[0]['id']
            return None
            
        except HttpError as error:
            logger.error(f"Failed to find folder '{folder_name}': {error}")
            raise
    
    def _ensure_folder_path(self, folder_path: str) -> str:
        """Ensure folder path exists, creating folders as needed. Returns the final folder ID."""
        if not folder_path or folder_path == '/':
            return 'root'
        
        # Remove leading/trailing slashes and split path
        folder_path = folder_path.strip('/')
        folder_names = folder_path.split('/')
        
        current_parent_id = 'root'
        
        for folder_name in folder_names:
            # Try to find existing folder
            folder_id = self._find_folder_by_name(folder_name, current_parent_id)
            
            if not folder_id:
                # Create folder if it doesn't exist
                folder_id = self._create_folder(folder_name, current_parent_id)
            
            current_parent_id = folder_id
        
        logger.info(f"Folder path '{folder_path}' resolved to ID: {current_parent_id}")
        return current_parent_id
    
    def find_file_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Find a file by its path in Google Drive (folder_path/filename)."""
        try:
            # Split path into folder path and filename
            path_parts = file_path.strip('/').split('/')
            filename = path_parts[-1]
            folder_path = '/'.join(path_parts[:-1]) if len(path_parts) > 1 else ''
            
            # Get folder ID
            if folder_path:
                parent_id = self._ensure_folder_path(folder_path)
                query = f"name='{filename}' and '{parent_id}' in parents"
            else:
                query = f"name='{filename}' and 'root' in parents"
            
            results = self.service.files().list(
                q=query,
                fields="files(id,name,mimeType,createdTime,modifiedTime)"
            ).execute()
            
            files = results.get('files', [])
            if files:
                logger.info(f"Found file '{file_path}': {files[0]['id']}")
                return files[0]
            
            logger.warning(f"File not found: '{file_path}'")
            return None
            
        except HttpError as error:
            logger.error(f"Failed to find file '{file_path}': {error}")
            raise
    
    def upload_file(self, local_file_path: str, drive_path: Optional[str] = None) -> Optional[str]:
        """
        Upload a file to Google Drive at the specified path.
        
        Args:
            local_file_path: Path to the local file to upload
            drive_path: Google Drive path where to upload (e.g., 'folder/subfolder/filename.txt')
                       If None, uploads to root with original filename
        
        Returns:
            File ID of the uploaded file
        """
        local_path_obj = Path(local_file_path)
        
        if not local_path_obj.exists():
            raise FileNotFoundError(f"File not found: {local_file_path}")
        
        # Determine the destination path and filename
        if drive_path is None:
            # Upload to root with original filename
            filename = local_path_obj.name
            parent_id = 'root'
            full_drive_path = filename
        else:
            # Parse the drive path
            drive_path = drive_path.strip('/')
            if '/' in drive_path:
                # Path includes folders
                path_parts = drive_path.split('/')
                filename = path_parts[-1]
                folder_path = '/'.join(path_parts[:-1])
                parent_id = self._ensure_folder_path(folder_path)
            else:
                # Just a filename, upload to root
                filename = drive_path
                parent_id = 'root'
            full_drive_path = drive_path
        
        try:
            file_metadata = {
                'name': filename,
                'parents': [parent_id] if parent_id != 'root' else None
            }
            
            # Remove None parents for root uploads
            if file_metadata['parents'] is None:
                del file_metadata['parents']
            
            media = MediaFileUpload(local_file_path, resumable=True)
            
            logger.info(f"Uploading file: {local_file_path} -> {full_drive_path}")
            request = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,name'
            )
            
            file = request.execute()
            file_id = file.get('id')
            
            logger.info(f"File uploaded successfully to '{full_drive_path}'. ID: {file_id}")
            return file_id
            
        except HttpError as error:
            logger.error(f"Failed to upload file: {error}")
            raise
        except Exception as error:
            logger.error(f"Unexpected error during upload: {error}")
            raise
    
    def download_file(self, drive_identifier: str, local_file_path: str) -> bool:
        """
        Download a file from Google Drive.
        
        Args:
            drive_identifier: Either a Google Drive file ID or a file path (e.g., 'folder/file.txt')
            local_file_path: Local path where to save the file
            
        Returns:
            True if download successful
        """
        try:
            # Determine if identifier is a file ID or path
            if '/' in drive_identifier or not drive_identifier.startswith('1'):
                # Looks like a path, find the file
                logger.info(f"Finding file by path: {drive_identifier}")
                file_info = self.find_file_by_path(drive_identifier)
                if not file_info:
                    raise FileNotFoundError(f"File not found at path: {drive_identifier}")
                file_id = file_info['id']
                file_name = file_info['name']
            else:
                # Looks like a file ID
                file_id = drive_identifier
                # Get file metadata to show filename
                file_metadata = self.service.files().get(fileId=file_id).execute()
                file_name = file_metadata.get('name', 'Unknown')
            
            logger.info(f"Downloading file: {file_name} (ID: {file_id})")
            
            request = self.service.files().get_media(fileId=file_id)
            
            local_path = Path(local_file_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(local_path, 'wb') as file_handle:
                downloader = MediaIoBaseDownload(file_handle, request)
                done = False
                
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        progress = int(status.progress() * 100)
                        logger.info(f"Download progress: {progress}%")
            
            logger.info(f"File downloaded successfully to: {local_file_path}")
            return True
            
        except HttpError as error:
            logger.error(f"Failed to download file: {error}")
            raise
        except Exception as error:
            logger.error(f"Unexpected error during download: {error}")
            raise
    
    def download_file_by_id(self, file_id: str, local_file_path: str) -> bool:
        """Download a file from Google Drive using file ID (legacy method)."""
        return self.download_file(file_id, local_file_path)
    
    def download_file_by_path(self, drive_path: str, local_file_path: str) -> bool:
        """Download a file from Google Drive using file path."""
        return self.download_file(drive_path, local_file_path)
    
    def list_files(self, max_results: int = 10, folder_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List files in Google Drive.
        
        Args:
            max_results: Maximum number of files to return
            folder_path: Optional folder path to list files from (e.g., 'projects/data')
                        If None, lists files from root
        
        Returns:
            List of file metadata dictionaries
        """
        try:
            if folder_path:
                folder_id = self._ensure_folder_path(folder_path)
                query = f"'{folder_id}' in parents"
                logger.info(f"Listing {max_results} files from folder: {folder_path}")
            else:
                query = "'root' in parents"
                logger.info(f"Listing {max_results} files from root")
            
            results = self.service.files().list(
                q=query,
                pageSize=max_results,
                fields="files(id,name,mimeType,createdTime,modifiedTime)"
            ).execute()
            
            files = results.get('files', [])
            logger.info(f"Found {len(files)} files")
            
            return files
            
        except HttpError as error:
            logger.error(f"Failed to list files: {error}")
            raise
        except Exception as error:
            logger.error(f"Unexpected error while listing files: {error}")
            raise
