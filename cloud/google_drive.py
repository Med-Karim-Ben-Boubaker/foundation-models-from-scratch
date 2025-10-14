import pickle
import os
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import datetime

from src.utils.logging import get_logger


logger = get_logger(__name__)


def Create_Service(client_secret_file, api_name, api_version, scopes, headless=False):
    logger.info(f"Creating service: {client_secret_file}-{api_name}-{api_version}-{scopes}")
    CLIENT_SECRET_FILE = client_secret_file
    API_SERVICE_NAME = api_name
    API_VERSION = api_version
    SCOPES = [scope for scope in scopes]
    logger.info(f"Scopes: {SCOPES}")
    logger.info(f"Headless mode: {headless}")

    cred = None

    pickle_file = f"token_{API_SERVICE_NAME}_{API_VERSION}.pickle"
    # print(pickle_file)

    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as token:
            cred = pickle.load(token)

    if not cred or not hasattr(cred, "valid") or not cred.valid:
        if (
            cred
            and hasattr(cred, "expired")
            and hasattr(cred, "refresh_token")
            and cred.expired
            and cred.refresh_token
        ):
            logger.info("Refreshing expired credentials...")
            cred.refresh(Request())
        else:
            logger.info("Starting OAuth flow...")
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)

            if headless:
                logger.info("Using headless authentication (console-based)")
                logger.info("Please visit this URL to authorize the application:")
                # Use out-of-band redirect URI for headless auth
                flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
                auth_url, _ = flow.authorization_url(prompt="consent")
                logger.info(f"\n{auth_url}\n")
                logger.info(
                    "After authorization, copy the code from the browser and paste it here:"
                )
                code = input("Authorization code: ").strip()
                # Fetch token and get credentials object
                flow.fetch_token(code=code)
                cred = flow.credentials
            else:
                logger.info("Using local server authentication (browser-based)")
                cred = flow.run_local_server(port=8081)

        with open(pickle_file, "wb") as token:
            pickle.dump(cred, token)

    try:
        service = build(API_SERVICE_NAME, API_VERSION, credentials=cred)
        logger.info(f"{API_SERVICE_NAME} service created successfully")
        return service
    except Exception as e:
        logger.info("Unable to connect.")
        logger.info(e)
        return None


def convert_to_RFC_datetime(year=1900, month=1, day=1, hour=0, minute=0):
    dt = datetime.datetime(year, month, day, hour, minute, 0).isoformat() + "Z"
    return dt
