from src.spotify_music_access import creds, caching_paths
from pathlib import Path
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from src.spotify_music_access.sdl_youtube_wrappers import spotify2youtube as s2y


def get_oauth(cid:str,secret:str):
    return SpotifyClientCredentials(client_id=cid, client_secret=secret)


def get_current_track_meta_data(cfg, sp:spotipy.Spotify):
    end_point = cfg["cur_song_meta"]["end_point"]
    access_token = sp.auth_manager.get_access_token()
    sp._get(end_point)#,{"Authorization":f"Bearer {access_token}"})
    response = requests.get(end_point,
                            headers={"Authorization":f"Bearer {access_token}"}).json()
    markets = strip_markets_data(response)
    # print(pformat(response,indent=4))
    return response


def strip_markets_data(response:dict):
    ret = []
    items = tuple(response.items())
    for k,v in items:
        if k == "available_markets":
            ret.append((k,response.pop(k)))
        if isinstance(v,dict):
            result = strip_markets_data(v)
            ret.append((k,result))
    return ret


def get_track_analysis(cfg, track_data):
    track_id = track_data["item"]["id"]
    end_point = cfg["end_point"].format(id=track_id)
    access_token = cfg["access_token"]
    response = requests.get(end_point,
                            headers={"Authorization": f"Bearer {access_token}"}).json()
    strip_markets_data(response)
    return response

def get_current_track_and_convert(skip_mp3=False):
    auth_mngr = get_oauth(creds["id"],creds["secret"])
    sp = spotipy.Spotify(auth_manager=auth_mngr)
    track_meta = get_current_track_meta_data(creds,sp)
    song_itm = track_meta["item"]
    album = song_itm.get("album",{})
    artists = song_itm.get("artists",[{"name":"N/A"}])
    # spotify_song_id = song_itm["id"]
    song_name = song_itm["name"]
    album_name = album.get("name","")
    artists = "feat. ".join(a["name"] for a in artists)
    out_path = Path(caching_paths["yt_music"])
    s2y(out_path,
        album_name,
        artists,
        song_name,
        song_itm.get("track_number",0),
        album.get("total_tracks",0),
        album.get("release_date","XXXX-XX-XX").split("-")[0],
        album.get("images",[{"url":None}])[0].get("url"),
        skip_mp3=skip_mp3)

def follow_demo():
    # auth_mngr = get_oauth(creds["id"],creds["secret"])
    # sp = spotipy.Spotify(auth_manager=auth_mngr)
    # track_meta = get_current_track_meta_data(creds["cur_song_meta"])
    # track_analysis = get_track_analysis(creds["track_analysis"], track_meta)
    get_current_track_and_convert()
    # print(pformat(track_analysis,indent=2))




if __name__ == '__main__':
    follow_demo()