import gdown


def download_file_from_google_drive(
    id="1QGpFSa46GRKfXAnKj1hJ51SkooqHAYUh", destination="./best_model_18.pth"
):
    URL = "https://drive.google.com/uc?id=" + id
    gdown.download(URL, destination, quiet=False)
