from cloud.drive_manager import DriveManager
    
drive = DriveManager(headless=True)
drive.upload_file("data/simple_wiki_cleaned.train", "training/babylm/train_10M/simple_wiki_cleaned.train")