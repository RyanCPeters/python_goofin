import urllib
import urllib.request

from mutagen.easyid3 import EasyID3
from mutagen.id3 import APIC, ID3
from mutagen.mp3 import MP3

from spotify_dl import youtube as yt
from src import *


def spotify2youtube(out_path:Path,album:str,artist:str,name:str,num:int=None,num_tracks:int=None,year:str=None, cover:str=None,genre:str=None,skip_mp3=False):
    num = 0 if num is None else num
    num_tracks = 0 if num_tracks is None else num_tracks
    year = "6666" if year is None else year
    cover = "" if cover is None else cover
    genre = "" if genre is None else genre
    songs = [{'album': album,
              'artist': artist,
              'cover': cover,
              'genre': genre,
              'name': name,
              'num': num,
              'num_tracks': num_tracks,
              'year': year}]
    out_path.mkdir(parents=True,exist_ok=True)
    try:
        yt.download_songs(songs,download_directory=str(out_path),format_string='best',skip_mp3=skip_mp3)
    except ValueError as ve:
        from pprint import pformat
        print(f"{type(ve)}: {ve.args}\n\traised when trying to download/convert to mp3 the following song:\n\t\t{pformat(songs)}\n\t\t{out_path=}")
        raise ve


def test_download_one_false_skip():
    songs = [{'album': 'Hell Freezes Over (Remaster 2018)',
              'artist': 'Eagles',
              'cover': 'https://i.scdn.co/image/ab67616d0000b27396d28597a5ae44ab66552183',
              'genre': 'album rock',
              'name': 'Hotel California - Live On MTV, 1994',
              'num': 6,
              'num_tracks': 15,
              'year': '1994'}]
    dl_path = cache_path.joinpath("test_sdl_yt")
    yt.download_songs(songs, download_directory=str(dl_path),
                      format_string='best',
                      skip_mp3=False)
    expected_fname = "Eagles - Hotel California - Live On MTV, 1994.mp3"
    fname_path = dl_path.joinpath(expected_fname)
    music = MP3(str(fname_path), ID3=EasyID3)
    tags = ID3(str(fname_path))
    assert (music['artist'][0] == 'Eagles')
    assert (music['album'][0] == 'Hell Freezes Over (Remaster 2018)')
    assert (music['genre'][0] == 'album rock')
    assert (tags.getall("APIC")[0].data == APIC(encoding=3,
                                                mime='image/jpeg',
                                                type=3, desc=u'Cover',
                                                data=urllib.request.urlopen(songs[0].get('cover')).read()
                                                ))


def test_download_two_false_skip():
    songs = [{'album': 'Hell Freezes Over (Remaster 2018)',
              'artist': 'Eagles',
              'cover': 'https://i.scdn.co/image/ab67616d0000b27396d28597a5ae44ab66552183',
              'genre': 'album rock',
              'name': 'Hotel California - Live On MTV, 1994',
              'num': 6,
              'num_tracks': 15,
              'year': '1994'}]
    dl_path = cache_path.joinpath("test_sdl_yt")
    yt.download_songs(songs, download_directory=str(dl_path),
                      format_string='best',
                      skip_mp3=False)
    expected_fname = "Eagles - Hotel California - Live On MTV, 1994.mp3"
    fname_path = dl_path.joinpath(expected_fname)
    music = MP3(str(fname_path), ID3=EasyID3)
    tags = ID3(str(fname_path))
    assert (music['artist'][0] == 'Eagles')
    assert (music['album'][0] == 'Hell Freezes Over (Remaster 2018)')
    assert (music['genre'][0] == 'album rock')
    assert (tags.getall("APIC")[0].data == APIC(encoding=3,
                                                mime='image/jpeg',
                                                type=3, desc=u'Cover',
                                                data=urllib.request.urlopen(songs[0].get('cover')).read()
                                                ))


def test_download_one_true_skip():
    songs = [
        {'album': 'Hell Freezes Over (Remaster 2018)',
         'artist': 'Eagles',
         'cover': 'https://i.scdn.co/image/ab67616d0000b27396d28597a5ae44ab66552183',
         'genre': 'album rock',
         'name': 'Hotel California - Live On MTV, 1994',
         'num': 6,
         'num_tracks': 15,
         'year': '1994'}]
    dl_path = cache_path.joinpath("Downloads")
    yt.download_songs(songs, download_directory=str(dl_path), format_string='best',
                      skip_mp3=False)


def test_download_cover_none():
    songs = [
        {'album': 'Queen II (Deluxe Remastered Version)',
         'artist': 'Queen',
         'cover': None,
         'genre': 'classic rock',
         'name': "The Fairy Feller's Master-Stroke - Remastered 2011",
         'num': 7,
         'num_tracks': 16,
         'year': '1974'}]
    dl_path = cache_path.joinpath("Downloads")
    yt.download_songs(songs, download_directory=str(dl_path),
                      format_string='best',
                      skip_mp3=False)
    expected_fname = "Queen - The Fairy Feller's Master-Stroke - Remastered 2011.mp3"
    expected_fname_path = dl_path.joinpath(expected_fname)
    music = MP3(str(expected_fname_path), ID3=EasyID3)
    tags = ID3(str(expected_fname_path))
    assert (music['artist'][0] == 'Queen')
    assert (music['album'][0] == 'Queen II (Deluxe Remastered Version)')
    assert (music['genre'][0] == 'classic rock')
    assert (len(tags.getall("APIC")) == 0)


if __name__ == '__main__':
    test_download_one_false_skip()
    test_download_two_false_skip()
    test_download_one_true_skip()
    test_download_cover_none()
