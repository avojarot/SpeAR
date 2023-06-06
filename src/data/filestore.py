from google.cloud import storage


def upload_blob(
    source_file_name,
    destination_blob_name="audio/",
    bucket_name="spear_bot",
):
    storage_client = storage.Client.from_service_account_json(
        "./credentials/spear-bot-388313-6a23d6901400.json"
    )
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name + source_file_name)

    blob.upload_from_filename(source_file_name)
