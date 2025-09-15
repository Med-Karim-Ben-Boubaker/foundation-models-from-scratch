from google_drive import Create_Service

CLIENT_SECRET_FILE_PATH = 'client_secret.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']

service = Create_Service(
    CLIENT_SECRET_FILE_PATH,
    API_NAME,
    API_VERSION,
    SCOPES
)

print(dir(service))