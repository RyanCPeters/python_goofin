import os
from pathlib import Path
import json
from src import code_root, caching_paths, cfg_path
from datetime import datetime as dt
from datetime import timedelta as td
import base64 as b64
import requests


v:str
for k,v in caching_paths.items():
    if v.startswith("../"):
        v = v[3:]
    _v = Path(v).resolve()
    if not _v.is_absolute():
        root = Path(code_root)
        while root.name:
            if root.name == str(_v.parts[0]):
                break
            root = root.parent
        else:
            err = FileNotFoundError(f"Unable to derive correct relative path to caching_paths['{k}'] = '{v}'")
            err.args += k,v
            raise err
        caching_paths[k] = str(root.parent.joinpath(_v))
creds = json.loads(cfg_path.joinpath("credentials.json").read_text())
os.environ["SPOTIPY_CLIENT_ID"] = creds["id"]
os.environ["SPOTIPY_CLIENT_SECRET"] = creds["secret"]
# os.environ["SPOTIPY_REDIRECT_URI"] = creds["redirect_uri"]
issued_time = dt.now()
expires_in = -1
expiration_time = issued_time


def get_auth_token():
    global expires_in, issued_time, expiration_time
    tdelta = dt.now() - issued_time
    if tdelta.seconds>expires_in:
        ccreds = f"{creds['id']}:{creds['secret']}"
        b64ccreds = b64.b64encode(ccreds.encode())
        token_url = "https://accounts.spotify.com/api/token"
        method = "POST"
        token_data = {
            "grant_type": "client_credentials"
        }
        token_headers = {
            "Authorization": f"Basic {b64ccreds.decode()}"  # <base64 encoded client_id:client_secret>
        }
        issued_dt = dt.now()
        r = requests.post(token_url, data=token_data, headers=token_headers)
        temp_token = None
        if 200<=r.status_code<300:
            token_response_data = r.json()
            expires_in = token_response_data["expires_in"]
            issued_time = issued_dt
            temp_token = token_response_data["access_token"]
            expiration_time = issued_time + td(seconds=expires_in)

        creds["temp_token"] = temp_token
        creds["expires_in"] = expires_in
        creds["expiration_time"] = expiration_time

get_auth_token()

