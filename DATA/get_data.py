import requests
import os
import hashlib
from tqdm import tqdm
import tarfile
import gzip
import json
import argparse

output_dir = "nuscenes"
region = "us"  # 'us' or 'asia'

# For nuScenes mini we just need the single v1.0-mini.tgz archive.
download_files = {
    "v1.0-mini.tgz": None,  # MD5 omitted; skip checksum if None
}

def login(username, password):
    headers = {
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
    }

    data = json.dumps({
        "AuthFlow": "USER_PASSWORD_AUTH",
        "ClientId": "7fq5jvs5ffs1c50hd3toobb3b9",
        "AuthParameters": {
            "USERNAME": username,
            "PASSWORD": password,
        },
        "ClientMetadata": {},
    })

    response = requests.post(
        "https://cognito-idp.us-east-1.amazonaws.com/",
        headers=headers,
        data=data,
    )

    if response.status_code == 200:
        try:
            token = json.loads(response.content)["AuthenticationResult"]["IdToken"]
            return token
        except KeyError:
            print("Authentication failed. 'AuthenticationResult' not found in the response.")
    else:
        print("Failed to login. Status code:", response.status_code)

    return None

def download_file(url, save_file, md5=None):
    response = requests.get(url, stream=True)

    if save_file.endswith(".tgz"):
        content_type = response.headers.get("Content-Type", "")
        if content_type == "application/x-tar":
            save_file = save_file.replace(".tgz", ".tar")
        elif content_type != "application/octet-stream":
            print("unknown content type", content_type)
            return save_file

    if os.path.exists(save_file):
        print(save_file, "already exists")
        if md5:
            md5obj = hashlib.md5()
            with open(save_file, "rb") as f:
                for chunk in f:
                    md5obj.update(chunk)
            hash_val = md5obj.hexdigest()
            if hash_val != md5:
                print(save_file, "MD5 mismatch, redownloading.")
            else:
                print(save_file, "MD5 check success")
                return save_file
        else:
            return save_file

    file_size = int(response.headers.get("Content-Length", 0))
    progress_bar = tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=save_file,
        ascii=True,
    )

    md5obj = hashlib.md5() if md5 else None
    with open(save_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                if md5obj:
                    md5obj.update(chunk)
                f.write(chunk)
                progress_bar.update(len(chunk))
    progress_bar.close()

    if md5:
        hash_val = md5obj.hexdigest()
        if hash_val != md5:
            print(save_file, "MD5 check failed")
        else:
            print(save_file, "MD5 check success")

    return save_file

def extract_tgz_to_original_folder(tgz_file_path):
    original_folder = os.path.dirname(tgz_file_path)
    print(f"Extracting {tgz_file_path} to {original_folder}")

    with gzip.open(tgz_file_path, "rb") as f_in:
        with tarfile.open(fileobj=f_in, mode="r") as tar:
            tar.extractall(original_folder)

def extract_tar_to_original_folder(tar_file_path):
    original_folder = os.path.dirname(tar_file_path)
    print(f"Extracting {tar_file_path} to {original_folder}")

    with tarfile.open(tar_file_path, "r") as tar:
        tar.extractall(original_folder)

def main():
    parser = argparse.ArgumentParser(description="Download nuScenes v1.0-mini")
    parser.add_argument("--email", required=True, help="nuScenes account email")
    parser.add_argument("--password", required=True, help="nuScenes account password")
    args = parser.parse_args()

    print("Logging in...")
    bearer_token = login(args.email, args.password)
    if not bearer_token:
        print("Login failed; aborting.")
        return

    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
    }

    print("Getting download urls...")
    download_data = {}
    for filename, md5 in download_files.items():
        api_url = (
            "https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/"
            f"v1/archives/v1.0/{filename}?region={region}&project=nuScenes"
        )

        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            print(filename, "request success")
            download_url = response.json()["url"]
            download_data[filename] = [
                download_url,
                os.path.join(output_dir, filename),
                md5,
            ]
        else:
            print(f"request failed : {response.status_code}")
            print(response.text)

    print("Downloading files...")
    os.makedirs(output_dir, exist_ok=True)
    for output_name, (download_url, save_file, md5) in download_data.items():
        save_file = download_file(download_url, save_file, md5)
        download_data[output_name] = [download_url, save_file, md5]

    print("Extracting files...")
    for output_name, (download_url, save_file, md5) in download_data.items():
        if output_name.endswith(".tgz"):
            extract_tgz_to_original_folder(save_file)
        elif output_name.endswith(".tar"):
            extract_tar_to_original_folder(save_file)
        else:
            print("unknown file type", output_name)

    print("Done!")

if __name__ == "__main__":
    main()