from src.spotify_music_access import *
from urllib.parse import urlencode

SCOPE = 'user-library-read'
SCOPE2= "user-read-playback-state"

def test_tutorial():
    if creds["expiration_time"]<=dt.now():
        get_auth_token()
    headers = {
        "Authorization":f"Bearer {creds['temp_token']}"
    }
    from pprint import pprint
    pprint(headers)
    end_point = f'{creds["cur_song_meta"]["end_point"]}?{urlencode({"market":"US"})}' # GET https://api.spotify.com/v1/me/player/currently-playing
    print(f"{end_point=}")
    req = requests.get(end_point,headers=headers)
    pprint(req.json())


if __name__ == '__main__':
    test_targets = []
    test_targets.append(test_tutorial)
    # test_targets.append(test_stackOverflow_solution)
    # test_targets.append(test_prompt_for_user_token)
    # test_targets.append(test_headless)
    # test_targets.append(test_another_bad_explanation)
    for f in test_targets:
        try:
            f()
            print(f"{__name__}.{f.__name__} -> success")
        except BaseException as be:
            print(f"{__name__}.{f.__name__} -> {type(be)}: {be.args}")
