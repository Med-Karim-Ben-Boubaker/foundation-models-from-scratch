from cloud.google_drive import Create_Service
from src.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    """Test Google Drive API connection and display available methods."""
    CLIENT_SECRET_FILE_PATH = 'client_secret.json'
    API_NAME = 'drive'
    API_VERSION = 'v3'
    SCOPES = ['https://www.googleapis.com/auth/drive']
    
    logger.info("Starting Google Drive API health check...")
    
    try:
        # Create service with headless authentication
        service = Create_Service(
            CLIENT_SECRET_FILE_PATH,
            API_NAME,
            API_VERSION,
            SCOPES,
            headless=True  # Use headless mode for console-based auth
        )
        
        if service is None:
            logger.error("Failed to create Google Drive service")
            return False
            
        logger.info("Google Drive service created successfully!")
        
        # Test basic API functionality
        try:
            # Get about information to test API access
            about = service.about().get(fields="user").execute()
            user_info = about.get('user', {})
            logger.info(f"Connected as: {user_info.get('displayName', 'Unknown')} ({user_info.get('emailAddress', 'No email')})")
            
            # List available methods
            logger.info("Available service methods:")
            methods = [method for method in dir(service) if not method.startswith('_')]
            for method in sorted(methods):
                logger.info(f"  - {method}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error testing API functionality: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error during health check: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("Health check completed successfully!")
    else:
        logger.error("Health check failed!")