import gdown


def download_file_from_google_drive(
    id="1QGpFSa46GRKfXAnKj1hJ51SkooqHAYUh",
):
    URL = "https://drive.google.com/uc?id=" + id
    return gdown.download(URL, quiet=False)
