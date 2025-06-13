import os
import shutil
import boto3

def upload_to_mock_cloud(local_path, mock_cloud_dir="mock_cloud"):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File not found: {local_path}")

    # Reconstruct path inside mock_cloud
    if "data/" in local_path:
        rel_path = local_path.split("data/")[-1]
        dest_path = os.path.join(mock_cloud_dir, rel_path)
    else:
        dest_path = os.path.join(mock_cloud_dir, os.path.basename(local_path))

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(local_path, dest_path)
    print(f"\n✅ Uploaded {local_path} to {dest_path}")
    return dest_path

def upload_to_s3(file_path, bucket_name, s3_key):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(file_path, bucket_name, s3_key)
        print(f"✅ Uploaded {file_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")